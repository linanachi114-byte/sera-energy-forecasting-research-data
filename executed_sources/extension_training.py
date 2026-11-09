"""Extend the unchanged numerical SERA recipe to three recovered sequences."""
from __future__ import annotations

import argparse
import json
import msvcrt
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "sera_numeric_20261109"
ARCHIVE = ROOT / "s_extension_20261108"
for key in ("LLM_API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY"):
    os.environ.pop(key, None)
os.environ["RESEARCH_OFFLINE"] = "1"
sys.path.insert(0, str(BASE))
import run_experiment as n
sys.path.insert(0, str(HERE))

n.HERE = HERE
n.SOURCE = HERE / "inputs"
pd, np, torch = n.pd, n.np, n.torch
MODELS = ("numeric_HES", "adapt_HES", "adapt_HE", "adapt_HS", "adapt_ES")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked(path, expected):
    actual = n.sha(path)
    assert actual == expected, f"Source identity mismatch: {path}"
    return actual


def prepare():
    plan = read(HERE / "PLAN.json")
    cells, sources = [], {}
    paths = [Path(__file__), HERE / "PLAN.json", BASE / "run_experiment.py", BASE / "PLAN.json",
             BASE / "run_identity.json", BASE / "audit.json", BASE / "cell_metrics.csv",
             ARCHIVE / "protocol.json", ARCHIVE / "preflight.json", ARCHIVE / "data_source_audit.json",
             ARCHIVE / "verified_seed_metrics.csv", Path(n.legacy.__file__), Path(n.legacy.exp.__file__),
             Path(n.linear.__file__), n.linear.PATH, Path(n.legacy.exp.trainer.__file__),
             n.legacy.exp.RESEARCH / "src/models/official_forecasting.py"]
    sources.update({str(p): n.sha(p) for p in paths})
    assert read(BASE / "audit.json")["status"] == "verified"
    assert sources[str(BASE / "run_experiment.py")] == read(BASE / "run_identity.json")["runner_sha256"]
    audit = read(ARCHIVE / "data_source_audit.json")
    for row in audit["results"]:
        assert row["local_processing_reproduced"]
        sources[row["processed_file"]] = checked(row["processed_file"], row["processed_sha256"])
        for source in row["source_files"]:
            sources[source["path"]] = checked(source["path"], source["sha256"])
            if "provider" in source:
                assert source["matches_provider"] and source["sha256"] == source["provider"]["provider_sha256"]
    ett = next(row for row in read(ARCHIVE / "preflight.json")["datasets"] if row["key"] == "H")
    assert ett["official_source"]["byte_identical"]
    sources[ett["path"]] = checked(ett["path"], ett["official_source"]["sha256"])
    for ds in plan["datasets"]:
        for h in plan["native_leads"][ds]:
            source = ARCHIVE / "prepared" / f"{ds}_h{h}.npz"
            meta_path = source.with_suffix(".json")
            meta = read(meta_path)
            sources[str(meta_path)] = n.sha(meta_path)
            sources[str(source)] = checked(source, meta["sha256"])
            raw = (n.legacy.exp.RESEARCH / n.legacy.exp.DATASETS[ds]["filepath"]).resolve()
            sources[str(raw)] = checked(raw, meta["source_sha256"])
            for folder, splits in (("latent_full_features", n.SPLITS[:3]), ("latent_test_features", ("test",))):
                target = n.SOURCE / "cells" / folder / f"{ds}_h{h}_s41.npz"
                with np.load(source, allow_pickle=False) as data:
                    # The historical field named 'semantic' contains integer origin indices only.
                    arrays = {key: data[key].copy() for split in splits for key in
                              (f"ext_{split}", f"target_{split}", f"semantic_{split}",
                               f"y_{split}_raw", f"center_{split}", f"scale_{split}")}
                if target.exists():
                    with np.load(target, allow_pickle=False) as saved:
                        assert set(saved.files) == set(arrays)
                        for key, value in arrays.items():
                            np.testing.assert_array_equal(saved[key], value)
                else:
                    n.npz(target, **arrays)
                sources[str(target)] = n.sha(target)
            for seed in plan["seeds"]:
                stem = f"{ds}_h{h}_s{seed}"
                point = ARCHIVE / "cells" / f"{stem}_DLinear.pt"
                marker = point.with_suffix(".json")
                point_meta = read(marker)
                sources[str(marker)] = n.sha(marker)
                sources[str(point)] = checked(point, point_meta["checkpoint_sha256"])
                assert point_meta["inputs"][str(source)] == meta["sha256"]
                target = HERE / "references" / point.name
                target.parent.mkdir(exist_ok=True)
                if not target.exists():
                    shutil.copy2(point, target)
                checked(target, sources[str(point)])
                n.atomic_json(target.with_suffix(".complete.json"), point_meta)
                reference = ARCHIVE / "cells" / f"{stem}_SERA.npz"
                reference_meta = read(reference.with_suffix(".json"))
                sources[str(reference)] = checked(reference, reference_meta["sha256"])
                cells.append(dict(dataset=ds, horizon=h, seed=seed, point=str(target), S=str(reference)))
    manifest = dict(cells=cells, sources=sources, new_llm_calls=0)
    path = HERE / "manifest.json"
    if path.exists():
        assert read(path) == manifest, "Frozen manifest mismatch; do not resume changed inputs."
    else:
        n.atomic_json(path, manifest)
    identity = dict(runner_sha256=n.sha(__file__), base_runner_sha256=n.sha(BASE / "run_experiment.py"),
                    plan_sha256=n.sha(HERE / "PLAN.json"), manifest_sha256=n.sha(path))
    identity_path = HERE / "run_identity.json"
    if identity_path.exists():
        assert read(identity_path) == identity
    else:
        n.atomic_json(identity_path, identity)
    return manifest, identity


def comparison_table(task, baselines, references):
    rows = []
    for scope, datasets in (("original_ten", set("CGMNOPRSTU")), ("additional_three", set("HLQ")),
                             ("combined_thirteen", set("CGHLMNOPQRSTU"))):
        selected = task[task.dataset.isin(datasets)]
        for split in ("cal", "test"):
            for model, reference in references:
                current = selected[(selected.model == model) & (selected.split == split)]
                base = (selected[(selected.model == reference) & (selected.split == split)]
                        if reference in set(task.model) else baselines[baselines.model == reference])
                if reference not in set(task.model) and split != "test":
                    continue
                joined = current.merge(base[["dataset", "horizon", "MAE"]], on=["dataset", "horizon"],
                                       suffixes=("", "_reference"), validate="one_to_one")
                if joined.empty:
                    continue
                gain = 100*(1-joined.MAE/joined.MAE_reference)
                rows.append(dict(scope=scope, split=split, model=model, reference=reference,
                                 datasets=int(joined.dataset.nunique()), tasks=len(joined),
                                 mean_reduction_pct=float(gain.mean()), wins=int((gain > 1e-4).sum())))
    return pd.DataFrame(rows)


def report(records):
    n.report(records)
    original = pd.read_csv(BASE / "cell_metrics.csv")
    added = pd.read_csv(HERE / "cell_metrics.csv")
    combined = pd.concat((original, added), ignore_index=True)
    assert not combined.duplicated(["dataset", "horizon", "seed", "split", "model"]).any()
    task = combined.groupby(["dataset", "horizon", "split", "model"], as_index=False).MAE.mean()
    task.to_csv(HERE / "combined_task_metrics.csv", index=False)
    baselines = pd.read_csv(ARCHIVE / "verified_seed_metrics.csv")
    baselines = baselines.groupby(["dataset", "horizon", "model"], as_index=False).MAE.mean()
    external = comparison_table(task, baselines, [(m, b) for m in MODELS for b in ("N", "PatchTST", "TimeXer", "TQNet")])
    external.to_csv(HERE / "external_comparisons.csv", index=False)
    pairs = [(m, "SERA_original") for m in MODELS] + [(m, "adapt_HES") for m in MODELS[1:]]
    comparison_table(task, baselines, pairs).to_csv(HERE / "architecture_comparisons.csv", index=False)
    n.atomic_json(HERE / "progress.json", dict(completed_cells=len(records), total_cells=18,
                  completed_added_tasks=int(added[added.split == "test"].groupby(["dataset", "horizon"]).ngroups),
                  new_llm_calls=0, manuscript_changed=False))


def preflight(manifest):
    original_source = n.SOURCE
    try:
        n.SOURCE = ROOT / "s_architecture_20261106"
        n.self_test()
    finally:
        n.SOURCE = original_source
    samples = {}
    for ds, h in sorted({(v["dataset"], v["horizon"]) for v in manifest["cells"]}):
        arrays, cfg, _ = n.numeric_arrays(ds, h)
        samples[f"{ds}_h{h}"] = {split: len(arrays[f"x_{split}"]) for split in n.SPLITS}
        for item in (v for v in manifest["cells"] if v["dataset"] == ds and v["horizon"] == h):
            model = n.linear.Point(cfg, arrays, "point")
            model.load_state_dict(torch.load(item["point"], map_location="cpu", weights_only=False)["best_state"], strict=True)
            with np.load(item["S"], allow_pickle=False) as ref:
                for split in ("cal", "test"):
                    np.testing.assert_array_equal(ref[f"target_{split}"], arrays[f"target_{split}"])
                    np.testing.assert_allclose(ref[f"y_{split}"], arrays[f"y_{split}_raw"], rtol=1e-6, atol=1e-6)
    n.atomic_json(HERE / "preflight.json", dict(status="verified", cells=18, tasks=9, samples=samples,
                  model_and_gradients_verified=True, all_numeric_inputs_reconstructed=True,
                  archived_truth_and_target_indices_matched=True, point_checkpoint_layouts_verified=True,
                  BDG2_provider_hashes_verified=True, semantic_features_loaded=False, new_llm_calls=0))
    print("EXTENSION PREFLIGHT PASSED: 18 cells, numerical inputs and references verified", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest, identity = prepare()
    if args.preflight:
        preflight(manifest)
        return
    check = read(HERE / "preflight.json")
    assert check["status"] == "verified" and check["cells"] == 18 and len(check["samples"]) == 9
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n.status(status="running", pid=os.getpid(), started_unix=time.time(), deadline_unix=time.time()+43200,
             completed_cells=0, total_cells=18, device=device, new_llm_calls=0)
    records = []
    try:
        for item in manifest["cells"]:
            record = n.run_cell(item, identity, device)
            stem = f"{item['dataset']}_h{item['horizon']}_s{item['seed']}"
            record["feature_sha256"] = n.sha(HERE / "features" / (stem+".npz"))
            n.atomic_json(HERE / "cells" / stem / "complete.json", record)
            records.append(record)
            report(records)
            n.status(completed_cells=len(records))
            print(json.dumps(dict(completed_cells=len(records), total_cells=18, last_cell=stem)), flush=True)
        n.status(status="auditing", current_stage="verify_saved_predictions")
        from audit_results import main as audit
        audit()
        n.status(status="complete", completed_unix=time.time())
    except Exception as error:
        n.status(status="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    with (HERE / ".run.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        with n.threadpool_limits(limits=1):
            main()
