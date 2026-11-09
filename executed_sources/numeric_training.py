"""Language-free SERA training and matched two/three-expert decoder evaluation."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "s_architecture_20261106"
os.environ["RESEARCH_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONUTF8"] = "1"
os.environ.pop("LLM_API_KEY", None)
sys.path.insert(0, str(SOURCE))
import run as legacy
import dlinear_s_model as linear
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from src.models.official_forecasting import OfficialForecasterBranch
from threadpoolctl import threadpool_limits

SPLITS = ("train", "val", "cal", "test")
SETS = {"HES": (0, 1, 2), "HE": (0, 1), "HS": (0, 2), "ES": (1, 2)}
STATUS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    for attempt in range(12):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(.025*2**attempt, .5))


def npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **values)
    temp.replace(path)


def status(**values):
    STATUS.update(values, updated_unix=time.time())
    atomic_json(HERE / "status.json", STATUS)


class Residual(nn.Module):
    def __init__(self, context_width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1 + context_width, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, reference, context):
        return reference + self.net(torch.cat((reference, context), dim=1))


class Encoder(nn.Module):
    def __init__(self, length, d):
        super().__init__()
        self.hist_branch = OfficialForecasterBranch("official_patchtst", length, 1, 1)
        self.ext_expert = Residual(d)
        self.context_expert = Residual(0)
        self.gate = nn.Sequential(nn.Linear(3+d, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, history, context, unused=None):
        h = self.hist_branch(history)
        e = self.ext_expert(h, context)
        c = self.context_expert(h, context[:, :0])
        experts = torch.cat((h, e, c), dim=1)
        weights = torch.softmax(self.gate(torch.cat((experts, context), dim=1)), dim=1)
        return {"mean": (weights*experts).sum(1, keepdim=True), "experts": experts, "alphas": weights}


class Decoder(nn.Module):
    def __init__(self, d, anchor=False, keep=(0, 1, 2)):
        super().__init__()
        self.d, self.anchor, self.keep = d, anchor, tuple(keep)
        if 1 in keep:
            self.ext_expert = Residual(d)
        if 2 in keep:
            self.context_expert = Residual(d+32)
        self.gate = nn.Sequential(nn.Linear(len(keep)+d+32, 64), nn.GELU(), nn.Linear(64, len(keep)))

    def forward(self, features, context, unused=None):
        h, z, d = features[:, 0, :1], features[:, 0, 1:33], features[:, 0, 33:34]
        values = {0: h}
        if 1 in self.keep:
            values[1] = self.ext_expert(d if self.anchor else h, context)
        if 2 in self.keep:
            values[2] = self.context_expert(h, torch.cat((context, z), dim=1))
        experts = torch.cat([values[i] for i in self.keep], dim=1)
        weights = torch.softmax(self.gate(torch.cat((experts, context, z), dim=1)), dim=1)
        return {"mean": (weights*experts).sum(1, keepdim=True), "experts": experts, "alphas": weights}


def initialize_decoder(model, source):
    """Transfer shared numeric weights; every newly introduced input is zero-initialized."""
    target = model.state_dict()
    old = source.state_dict()
    old_d = source.d if isinstance(source, Decoder) else source.ext_expert.net[0].in_features-1
    for name, value in target.items():
        previous = old[name]
        if name.endswith("net.0.weight") or name == "gate.0.weight":
            prefix = (3 if name == "gate.0.weight" else 1) + old_d
            value.zero_()
            if name == "context_expert.net.0.weight" and isinstance(source, Encoder):
                value[:, :1] = previous[:, :1]
            else:
                value[:, :prefix] = previous[:, :prefix]
                if previous.shape[1] > prefix:
                    value[:, prefix+model.d-old_d:] = previous[:, prefix:]
        else:
            value.copy_(previous)
    model.load_state_dict(target, strict=True)


def prune(source, keep):
    result = Decoder(source.d, anchor=source.anchor, keep=keep)
    state, original = result.state_dict(), source.state_dict()
    for name, value in state.items():
        if name == "gate.0.weight":
            value.copy_(original[name][:, list(keep)+list(range(3, original[name].shape[1]))])
        elif name in ("gate.2.weight", "gate.2.bias"):
            value.copy_(original[name][list(keep)])
        else:
            value.copy_(original[name])
    result.load_state_dict(state, strict=True)
    return result


def numeric_arrays(ds, h):
    """Load only numerical data fields, never cached predictions or learned representations."""
    cache = SOURCE / "cells" / ("latent_features" if ds in "CRSU" and h == 12 else "latent_full_features") / f"{ds}_h{h}_s41.npz"
    test_cache = SOURCE / "cells/latent_test_features" / f"{ds}_h{h}_s41.npz"
    arrays = {}
    for file, splits in ((cache, SPLITS[:3]), (test_cache, ("test",))):
        with np.load(file, allow_pickle=False) as data:
            for split in splits:
                for prefix in ("ext", "target", "semantic", "y_"+split+"_raw", "center", "scale"):
                    key = prefix if prefix.startswith("y_") else f"{prefix}_{split}"
                    arrays[key] = data[key].copy()
    frame, target, columns, cfg = legacy.exp.load_research_dataframe(ds, data_fraction=1., fraction_mode="full")
    values = frame[target].to_numpy(dtype=np.float32)
    raw = frame[columns].to_numpy(dtype=np.float32)
    ext_scaler = MinMaxScaler().fit(raw[arrays["semantic_train"]])
    ext = ext_scaler.transform(raw).astype(np.float32)
    histories = {}
    for split in SPLITS:
        origins = arrays[f"semantic_{split}"].astype(np.int64)
        np.testing.assert_array_equal(origins+h, arrays[f"target_{split}"])
        indices = origins[:, None]-np.arange(cfg["hist_len"]-1, -1, -1)[None, :]
        assert indices.min() >= 0 and (indices <= origins[:, None]).all()
        histories[split] = values[indices]
        np.testing.assert_allclose(ext[origins], arrays[f"ext_{split}"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(values[origins+h], arrays[f"y_{split}_raw"].reshape(-1), rtol=1e-6, atol=1e-6)
        c = ext[origins[:, None]-np.arange(23, -1, -1)[None, :]]
        arrays[f"context_{split}"] = np.column_stack((ext[origins], c.mean(1)-ext[origins], c[:, -1]-c[:, 0], c.std(1))).astype(np.float32)
    scaler = MinMaxScaler().fit(histories["train"].reshape(-1, 1))
    for split in SPLITS:
        hist = scaler.inverse_transform(scaler.transform(histories[split].reshape(-1, 1)).astype(np.float32)).reshape(histories[split].shape)
        center, scale = arrays[f"center_{split}"], arrays[f"scale_{split}"]
        np.testing.assert_allclose(hist.mean(1), center, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(np.maximum(hist.std(1), 1e-6), scale, rtol=1e-4, atol=1e-4)
        arrays[f"x_{split}"] = ((hist-center[:, None])/scale[:, None]).astype(np.float32)[:, :, None]
        arrays[f"y_{split}"] = ((arrays[f"y_{split}_raw"].reshape(-1)-center)/scale).astype(np.float32)[:, None]
        arrays[f"weight_{split}"] = (scale/arrays["scale_train"].mean()).astype(np.float32)[:, None]
    return arrays, cfg, {str(p): sha(p) for p in (cache, test_cache, legacy.exp.RESEARCH / str(legacy.exp.DATASETS[ds]["filepath"]))}


def fit(model, arrays, stage, seed, folder, identity, device, weighted=True):
    legacy.exp.set_seed(seed)
    model.to(device)
    checkpoint = folder / (stage+".pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003 if weighted else .00015, weight_decay=.0001)
    tensors = {key: torch.as_tensor(arrays[key], dtype=torch.float32, device=device)
               for split in ("train", "val") for key in (f"x_{split}", f"ext_{split}", f"y_{split}", f"weight_{split}")}
    def loss(split, start, stop):
        ix = slice(start, stop)
        error = abs(model(tensors[f"x_{split}"][ix], tensors[f"ext_{split}"][ix])["mean"]-tensors[f"y_{split}"][ix])
        return (error*tensors[f"weight_{split}"][ix]).mean() if weighted else error.mean()
    def validate():
        model.eval()
        with torch.no_grad():
            total = sum(float(loss("val", k, k+128))*len(tensors["y_val"][k:k+128]) for k in range(0, len(tensors["y_val"]), 128))
        return total/len(tensors["y_val"])
    best = copy.deepcopy(model.state_dict())
    best_val = validate() if weighted else float("inf")
    start, bad, best_epoch, spent = 0, 0, 0, 0.
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        assert saved["identity"] == identity
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        best, best_val, start, bad, best_epoch, spent = (saved[k] for k in ("best_state", "best_val", "epoch", "bad_epochs", "best_epoch", "training_seconds"))
        if saved["complete"]:
            model.load_state_dict(best)
            return model.cpu(), saved["metadata"]
        np.random.set_state(saved["numpy_random_state"])
        random.setstate(saved["python_random_state"])
        torch.set_rng_state(saved["torch_random_state"].cpu())
        if saved["cuda_random_state"] is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_random_state"]])
    timer = time.perf_counter()
    for epoch in range(start, 80):
        model.train()
        order = torch.randperm(len(tensors["y_train"]), device=device)
        for k in range(0, len(order), 128):
            ix = order[k:k+128]
            error = abs(model(tensors["x_train"][ix], tensors["ext_train"][ix])["mean"]-tensors["y_train"][ix])
            value = (error*tensors["weight_train"][ix]).mean() if weighted else error.mean()
            assert torch.isfinite(value)
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        score = validate()
        if score < best_val:
            best, best_val, bad, best_epoch = copy.deepcopy(model.state_dict()), score, 0, epoch+1
        else:
            bad += 1
        seconds = spent+time.perf_counter()-timer
        metadata = dict(epochs=epoch+1, best_epoch=best_epoch, validation_loss=best_val, training_seconds=seconds, parameters=sum(p.numel() for p in model.parameters()))
        saved = dict(identity=identity, model_state=model.state_dict(), best_state=best, optimizer_state=optimizer.state_dict(), epoch=epoch+1,
                     best_val=best_val, bad_epochs=bad, best_epoch=best_epoch, training_seconds=seconds, metadata=metadata,
                     numpy_random_state=np.random.get_state(), python_random_state=random.getstate(), torch_random_state=torch.get_rng_state(),
                     cuda_random_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, complete=bad >= 15 or epoch == 79)
        temp = checkpoint.with_suffix(".tmp.pt")
        torch.save(saved, temp)
        temp.replace(checkpoint)
        status(current_stage=stage, epoch=epoch+1, best_epoch=best_epoch, validation_loss=best_val)
        if time.time() > STATUS.get("deadline_unix", float("inf")):
            raise TimeoutError("Twelve-hour execution limit reached; epoch checkpoint saved for continuation")
        if saved["complete"]:
            break
    model.load_state_dict(best)
    return model.cpu(), metadata


def frozen_features(encoder, arrays, point_path, cfg, device):
    encoder.to(device).eval()
    point = linear.Point(cfg, arrays, "point")
    point.load_state_dict(torch.load(point_path, map_location="cpu", weights_only=False)["best_state"], strict=True)
    point.to(device).eval()
    captured, matrix = [], None
    def hook(module, args):
        nonlocal matrix
        features = args[0].detach().reshape(len(args[0]), -1)
        if matrix is None:
            matrix = torch.as_tensor(np.random.default_rng(17073).choice([-1., 1.], size=(features.shape[1], 32))/np.sqrt(32), dtype=features.dtype, device=device)
        captured.append((features@matrix).cpu().numpy())
    handle = encoder.hist_branch.model.model.head.linear.register_forward_pre_hook(hook)
    blocks = {}
    try:
        with torch.no_grad():
            for split in SPLITS:
                captured = []
                h, d = [], []
                for k in range(0, len(arrays[f"x_{split}"]), 256):
                    x = torch.as_tensor(arrays[f"x_{split}"][k:k+256], dtype=torch.float32, device=device)
                    h.append(encoder.hist_branch(x).cpu().numpy())
                    d.append(point(x, None, None)["mean"].cpu().numpy())
                blocks[split] = (np.concatenate(h), np.concatenate(captured), np.concatenate(d))
    finally:
        handle.remove()
    mean, scale = blocks["train"][1].mean(0), np.maximum(blocks["train"][1].std(0), 1e-6)
    result = {k: v for k, v in arrays.items() if not k.startswith("x_")}
    for split, (h, z, d) in blocks.items():
        result[f"x_{split}"] = np.column_stack((h, (z-mean)/scale, d)).astype(np.float32)[:, None, :]
    encoder.cpu()
    return result, dict(mean=mean.tolist(), scale=scale.tolist())


def predict(model, arrays, split):
    model.cpu().eval()
    outputs = {"mean": [], "experts": [], "alphas": []}
    with torch.no_grad():
        for start in range(0, len(arrays[f"x_{split}"]), 1024):
            output = model(torch.as_tensor(arrays[f"x_{split}"][start:start+1024]), torch.as_tensor(arrays[f"ext_{split}"][start:start+1024]))
            for key in outputs:
                outputs[key].append(output[key].numpy())
    result = {k: np.concatenate(v) for k, v in outputs.items()}
    for key in ("mean", "experts"):
        result[key] = result[key]*arrays[f"scale_{split}"][:, None]+arrays[f"center_{split}"][:, None]
    return result


def report(records):
    rows = [row for record in records for row in record["metrics"]]
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "cell_metrics.csv", index=False)
    tasks = table.groupby(["dataset", "horizon", "split", "model"], as_index=False).MAE.mean()
    tasks.to_csv(HERE / "task_metrics.csv", index=False)
    summaries = []
    for split in ("cal", "test"):
        scope = tasks[tasks.split == split]
        for reference in ("SERA_original", "numeric_HES", "numeric_H", "numeric_E", "numeric_S", "numeric_equal"):
            base = scope[scope.model == reference][["dataset", "horizon", "MAE"]]
            joined = scope.merge(base, on=["dataset", "horizon"], suffixes=("", "_reference"), validate="many_to_one")
            for model, group in joined.groupby("model"):
                gain = 100*(1-group.MAE/group.MAE_reference)
                summaries.append(dict(split=split, model=model, reference=reference, tasks=len(group), mean_reduction_pct=float(gain.mean()), wins=int((gain > 1e-4).sum())))
    atomic_json(HERE / "summary.json", dict(completed_cells=len(records), results=summaries, independent_test=False, new_llm_calls=0))


def run_cell(item, identity, device):
    ds, h, seed = item["dataset"], item["horizon"], item["seed"]
    stem = f"{ds}_h{h}_s{seed}"
    folder = HERE / "cells" / stem
    marker = folder / "complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        assert saved["identity"] == identity and sha(folder / "predictions.npz") == saved["predictions_sha256"]
        return saved
    status(current_cell=stem, current_stage="numeric_data", epoch=0)
    arrays, cfg, inputs = numeric_arrays(ds, h)
    assert sha(item["point"]) == json.loads(Path(item["point"]).with_suffix(".complete.json").read_text())["checkpoint_sha256"]
    inputs[item["point"]] = sha(item["point"])
    cell_identity = {**identity, "inputs": inputs, "cell": stem}
    d = arrays["ext_train"].shape[1]
    legacy.exp.set_seed(seed)
    encoder, enc_meta = fit(Encoder(cfg["hist_len"], d), arrays, "encoder", seed, folder, cell_identity, device, weighted=False)
    features, stats = frozen_features(encoder, arrays, item["point"], cfg, device)
    npz(HERE / "features" / (stem+".npz"), **{k: v for k, v in features.items() if k.startswith(("x_", "target_", "center_", "scale_"))})
    atomic_json(folder / "projection.json", stats)
    training = {"encoder": enc_meta}
    legacy.exp.set_seed(seed)
    model = Decoder(d)
    initialize_decoder(model, encoder)
    model, training["latent"] = fit(model, features, "latent", seed, folder, cell_identity, "cpu")
    context = dict(features)
    for split in SPLITS:
        context[f"ext_{split}"] = arrays[f"context_{split}"]
    legacy.exp.set_seed(seed)
    expanded = Decoder(4*d)
    initialize_decoder(expanded, model)
    model, training["context"] = fit(expanded, context, "context", seed, folder, cell_identity, "cpu")
    model.anchor = True
    model, training["anchor"] = fit(model, context, "anchor", seed, folder, cell_identity, "cpu")
    packed, metrics = {}, []
    def evaluate(name, current):
        for split in ("cal", "test"):
            out = predict(current, context, split)
            y = arrays[f"y_{split}_raw"].reshape(-1).astype(float)
            packed[f"{split}_{name}"] = out["mean"].reshape(-1)
            packed[f"{split}_{name}_experts"] = out["experts"]
            packed[f"{split}_{name}_weights"] = out["alphas"]
            predictions = {name: out["mean"].reshape(-1)}
            if name == "numeric_HES":
                predictions.update({f"numeric_{label}": out["experts"][:, i] for i, label in enumerate("HES")})
                predictions["numeric_equal"] = out["experts"].mean(1)
            for label, value in predictions.items():
                metrics.append(dict(dataset=ds, horizon=h, seed=seed, split=split, model=label, MAE=float(abs(value-y).mean()), samples=len(y)))
    evaluate("numeric_HES", model)
    for name, keep in SETS.items():
        legacy.exp.set_seed(seed)
        reduced = prune(model, keep)
        reduced, training["adapt_"+name] = fit(reduced, context, "adapt_"+name, seed, folder, cell_identity, "cpu")
        evaluate("adapt_"+name, reduced)
    with np.load(item["S"], allow_pickle=False) as original:
        for split in ("cal", "test"):
            y = arrays[f"y_{split}_raw"].reshape(-1).astype(float)
            np.testing.assert_array_equal(original[f"target_{split}"], arrays[f"target_{split}"])
            np.testing.assert_allclose(original[f"y_{split}"], y, rtol=1e-6, atol=1e-6)
            value = original[f"pred_{split}"]
            packed[f"{split}_SERA_original"] = value
            packed[f"{split}_y"], packed[f"{split}_target"] = y, arrays[f"target_{split}"]
            metrics.append(dict(dataset=ds, horizon=h, seed=seed, split=split, model="SERA_original", MAE=float(abs(value-y).mean()), samples=len(y)))
    assert not any(any(term in name for term in ("semantic", "reliability", "quantile")) for name, _ in model.named_parameters())
    npz(folder / "predictions.npz", **packed)
    saved = dict(identity=identity, inputs=inputs, training=training, metrics=metrics, language_trained_weights_used=False,
                 semantic_feature_arrays_loaded=False, new_llm_calls=0, predictions_sha256=sha(folder / "predictions.npz"))
    atomic_json(marker, saved)
    del arrays, features, context, encoder, model, reduced
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return saved


def self_test():
    legacy.exp.set_seed(41)
    encoder = Encoder(24, 4).eval()
    x, c = torch.randn(5, 24, 1), torch.randn(5, 4)
    out = encoder(x, c)
    out["mean"].square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters() if p.requires_grad)
    decoder = Decoder(4)
    initialize_decoder(decoder, encoder)
    features = torch.randn(5, 1, 34)
    features[:, 0, 0] = encoder.hist_branch(x).detach().reshape(-1)
    torch.testing.assert_close(decoder(features, c)["mean"], out["mean"], rtol=1e-6, atol=1e-6)
    expanded = Decoder(16)
    initialize_decoder(expanded, decoder)
    context = torch.cat((c, torch.randn(5, 12)), dim=1)
    torch.testing.assert_close(expanded(features, context)["mean"], decoder(features, c)["mean"], rtol=1e-6, atol=1e-6)
    for keep in SETS.values():
        reduced = prune(expanded, keep)
        value = reduced(features, context)
        assert value["experts"].shape == (5, len(keep))
        value["mean"].square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in reduced.parameters())
    arrays, cfg, inputs = numeric_arrays("C", 12)
    atomic_json(HERE / "preflight.json", dict(status="verified", zero_initialized_transfers_preserve_predictions=True,
                all_four_decoder_paths_have_finite_gradients=True, no_language_inputs_or_parameters=True,
                numerical_data_reconstruction_verified=True, samples={k: len(arrays[f"x_{k}"]) for k in SPLITS}, inputs=inputs))
    print("SELF TEST PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.self_test:
        self_test()
        return
    plan = json.loads((HERE / "PLAN.json").read_text())
    manifest_path = SOURCE / "S_external_anchor_candidates_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cells = sorted(manifest["cells"], key=lambda r: (not (r["dataset"] in "CRSU" and r["horizon"] == 12), r["dataset"], r["horizon"], r["seed"]))
    if args.limit:
        cells = cells[:args.limit]
    identity = dict(runner_sha256=sha(__file__), plan_sha256=sha(HERE / "PLAN.json"), manifest_sha256=sha(manifest_path))
    atomic_json(HERE / "run_identity.json", {**identity, "plan": plan})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    status(status="running", pid=os.getpid(), started_unix=time.time(), deadline_unix=time.time()+43200,
           completed_cells=0, total_cells=len(cells), device=device)
    records = []
    try:
        for index, item in enumerate(cells, 1):
            records.append(run_cell(item, identity, device))
            report(records)
            status(completed_cells=index)
            print(json.dumps(dict(completed_cells=index, total_cells=len(cells), last_cell=STATUS["current_cell"])), flush=True)
        status(status="complete", completed_unix=time.time())
    except Exception as error:
        status(status="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    with (HERE / ".run.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        with threadpool_limits(limits=1):
            main()
