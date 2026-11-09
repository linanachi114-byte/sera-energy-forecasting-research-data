"""Use a distinct numerical anchor in only one existing S expert."""
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import history_anchor as shared
import run as s

STAGE = "split_anchor_pilot"
VARIANTS = ["history_only", "external_only", "semantic_only"]


class Candidate(shared.exo.Candidate):
    def __init__(self, cfg, arrays, variant):
        super().__init__(cfg, arrays, variant)
        assert variant in VARIANTS+["all_anchor", "warm_control"]
        self.variant = variant

    def forward(self, x_hist, x_ext, x_sem):
        h, z, linear = x_hist[:, 0, :1], x_hist[:, 0, 1:33], x_hist[:, 0, 33:34]
        content, rho = x_sem[:, :9], x_sem[:, 9:10]
        hp = linear if self.variant in ("history_only", "all_anchor") else h
        he = linear if self.variant in ("external_only", "all_anchor") else h
        hs = linear if self.variant in ("semantic_only", "all_anchor") else h
        ext = self.ext_expert(he, x_ext)
        semantic = self.semantic_expert(hs, s.torch.cat((x_ext, z, content, s.torch.zeros_like(rho)), dim=1))
        candidates = s.torch.cat((hp, ext, semantic), dim=1)
        logits = self.gate(s.torch.cat((candidates, x_ext, content, z, s.torch.zeros_like(rho)), dim=1))
        logits = s.torch.cat((logits[:, :2], logits[:, 2:]+s.F.softplus(self.reliability_slope_raw)*(rho-.5)), dim=1)
        weights = s.torch.softmax(logits, dim=1)
        return {"mean": (weights*candidates).sum(1, keepdim=True), "alphas": weights}


def compare():
    new = s.pd.read_csv(s.HERE/f"{STAGE}_metrics.csv")
    old = s.pd.read_csv(s.HERE/"exogenous_full_metrics.csv").drop(columns="variant")
    control = s.pd.read_csv(s.HERE/"expert_pruning_pilot_metrics.csv")
    control = control[(control.variant == "warm_control") & (control.seed == 41)].drop(columns="variant")
    joined = new.merge(old, on=["dataset", "horizon", "seed"], suffixes=("_new", "_context"), validate="many_to_one")
    joined = joined.merge(control[["dataset", "horizon", "seed", "S_MAE", "pair_MAE"]], on=["dataset", "horizon", "seed"], validate="many_to_one")
    assert len(joined) == 12
    for key, label in (("S_MAE", "candidate"), ("pair_MAE", "pair")):
        joined[f"{label}_gain_pct"] = 100*(1-joined[f"{key}_new"]/joined[f"{key}_context"])
        joined[f"{label}_vs_control_pct"] = 100*(1-joined[f"{key}_new"]/joined[key])
    joined.to_csv(s.HERE/"split_anchor_comparison_tasks.csv", index=False)
    results = []
    for variant, group in joined.groupby("variant"):
        values = {k: float(group[k].mean()) for k in ("candidate_gain_pct", "pair_gain_pct", "candidate_vs_control_pct", "pair_vs_control_pct")}
        wins = {k: int((group[f"{k}_gain_pct"] > .0001).sum()) for k in ("candidate", "pair")}
        passed = min(values["candidate_gain_pct"], values["pair_gain_pct"]) >= .1 and min(wins.values()) >= 3 and min(values["candidate_vs_control_pct"], values["pair_vs_control_pct"]) >= 0
        results.append(dict(variant=variant, **values, wins=wins, passes_replication_gate=bool(passed), score=(values["candidate_gain_pct"]+values["pair_gain_pct"])/2))
    output = dict(scope="CRSUh12seed41 calibration only", results=sorted(results, key=lambda r: -r["score"]), llm_calls=0, test_metrics=False)
    s.exp.atomic_json(s.HERE/"split_anchor_comparison.json", output)
    print(json.dumps(output, indent=2), flush=True)


def main():
    shared.safe_research_io.install()
    s.torch.set_num_threads(1)
    s.exp.DEVICE = s.torch.device("cpu")
    s.exp.PointOnlyLoss = shared.WeightedPointLoss
    s.exp.train_model = shared.head.train
    s.SemanticModel = Candidate
    source = json.loads((s.HERE/"latent_full_plan.json").read_text(encoding="utf-8"))
    paths = [Path(__file__), s.HERE/"SPLIT_ANCHOR_PLAN.md", s.HERE/"history_anchor.py", s.HERE/"history_anchor_comparison.json",
             s.HERE/"history_anchor_pilot_audit.json", s.HERE/"dlinear_s_model.py", shared.light.PATH, s.HERE/"joint_finetune.py",
             s.HERE/"exogenous_context.py", s.HERE/"exogenous_full.py", s.HERE/"latent_full.py", s.HERE/"latent_heads.py",
             s.HERE/"run.py", s.HERE/"safe_research_io.py", s.HERE/"latent_full_plan.json", s.HERE/"expert_pruning_pilot_metrics.csv",
             s.HERE/"expert_pruning_pilot_audit.json", s.HERE/"exogenous_full_metrics.csv"]
    identity = dict(run_sha256=s.exp.file_sha(Path(__file__)), sources={str(p): s.exp.file_sha(p) for p in paths})
    plan = dict(stage=STAGE, identity=identity, variants=VARIANTS, datasets=list("CRSU"), horizons={d: [12] for d in "CRSU"}, seeds=[41], llm_calls=0, test_metrics=False)
    path = s.HERE/f"{STAGE}_plan.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == plan
    else:
        s.exp.atomic_json(path, plan)
    status = s.HERE/"split_anchor_status.json"
    state = dict(status="running", pid=os.getpid(), completed_cells=0, total_cells=12, started_unix=time.time())
    s.exp.atomic_json(status, state)
    rows, checks = [], []
    try:
        for dataset in "CRSU":
            base, _, refs = shared.full.pack(dataset, 12, 41, source["identity"])
            observed, cfg = shared.history.observed(dataset, base)
            arrays = {**base, **shared.expanded.context(dataset, 12, base)}
            inverses = {k: s.norm.WindowInverse(base[f"center_{k}"], base[f"scale_{k}"]) for k in ("train", "val", "cal")}
            point_checkpoint = s.HERE/"cells"/"dlinear_point_pilot"/f"{dataset}_h12_s41_point.pt"
            point = shared.light.Point(cfg, arrays, "point")
            point.load_state_dict(s.torch.load(point_checkpoint, map_location="cpu", weights_only=False)["best_state"], strict=True)
            point.eval()
            with s.torch.no_grad():
                for split in ("train", "val", "cal"):
                    values = []
                    for start in range(0, len(observed[f"x_{split}"]), 512):
                        x = s.torch.as_tensor(observed[f"x_{split}"][start:start+512], dtype=s.torch.float32)
                        values.append(point(x, None, None)["mean"].cpu().numpy().reshape(-1))
                    h = s.np.concatenate(values)
                    arrays[f"x_{split}"] = s.np.concatenate((base[f"x_{split}"], h[:, None, None]), axis=2)
                    arrays[f"y_{split}"] = s.np.column_stack((base[f"y_{split}"].reshape(-1), base[f"scale_{split}"]/base["scale_train"].mean())).astype(s.np.float32)
            checkpoint = s.HERE/"cells"/"exogenous_full"/f"{dataset}_h12_s41_w24_stats.pt"
            shared.exo.START_STATE = s.torch.load(checkpoint, map_location="cpu", weights_only=False)["best_state"]
            shared.exo.BASE_D = arrays["ext_train"].shape[1]
            warm = Candidate({}, arrays, "warm_control")
            warm_pred, _ = s.exp.base.predict_point(warm, arrays, "cal", inverses["cal"], s.exp.DEVICE)
            with s.np.load(checkpoint.with_suffix(".npz"), allow_pickle=False) as f:
                s.np.testing.assert_allclose(warm_pred, f["pred_cal"], rtol=1e-6, atol=1e-6)
            prior = s.HERE/"cells"/"history_anchor_pilot"/f"{dataset}_h12_s41_linear_anchor.pt"
            replay = Candidate({}, arrays, "all_anchor")
            replay.load_state_dict(s.torch.load(prior, map_location="cpu", weights_only=False)["best_state"], strict=True)
            replay_pred, _ = s.exp.base.predict_point(replay, arrays, "cal", inverses["cal"], s.exp.DEVICE)
            with s.np.load(prior.with_suffix(".npz"), allow_pickle=False) as f:
                s.np.testing.assert_allclose(replay_pred, f["pred_cal"], rtol=1e-6, atol=1e-6)
            checks.append(dict(dataset=dataset, unchanged_warm_start=True, all_anchor_source_reproduced=True,
                               trainable_parameters=sum(p.numel() for p in warm.parameters())))
            s.exp.atomic_json(s.HERE/"split_anchor_feature_checks.json", checks)
            raw = (s.exp.RESEARCH/str(s.exp.DATASETS[dataset]["filepath"])).resolve()
            inputs = {**identity["sources"], **refs, **{str(p): s.exp.file_sha(p) for p in (point_checkpoint, point_checkpoint.with_suffix(".complete.json"), checkpoint, checkpoint.with_suffix(".complete.json"), prior, prior.with_suffix(".npz"), raw)}}
            for variant in VARIANTS:
                state.update(current_cell=f"{dataset}_{variant}", updated_unix=time.time())
                s.exp.atomic_json(status, state)
                rows.append(s.fit(dataset, 12, 41, {}, arrays, inverses, variant, STAGE, dict(run_sha256=identity["run_sha256"], sources=inputs)))
                s.summarize(rows, STAGE)
                state.update(completed_cells=len(rows), updated_unix=time.time())
                s.exp.atomic_json(status, state)
            del arrays, base, observed, point, warm, replay
            gc.collect()
        subprocess.run([sys.executable, str(s.HERE/"audit.py"), "--stage", STAGE], check=True, cwd=s.HERE, timeout=900)
        compare()
        state.update(status="complete", completed_unix=time.time())
        s.exp.atomic_json(status, state)
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        s.exp.atomic_json(status, state)
        raise


if __name__ == "__main__":
    with s.threadpool_limits(limits=1):
        main()
