"""Build thirteen-series DRA tables and redistributable errors from frozen outputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "research_data"
GEN = HERE / "manuscript/generated"
ORDER = list("CGHLMNOPQRSTU")
NAMES = dict(zip(ORDER, ["Appliances", "ETTm1", "ETTm2", "GEFCom price", "GEFCom load", "OPSD load",
                       "Household power", "Electricity total", "BDG2 Hog", "Steel energy", "Victoria demand",
                       "Alabama solar", "Tetouan load"]))
CITES = dict(zip(ORDER, ["candanedo2017appliances", "zhou2021informer", "zhou2021informer",
                       "hong2016probabilistic", "hong2016probabilistic", "wiese2019opsd", "hebrail2006household",
                       "trindade2015eld", "miller2020bdg2", "sathishkumar2021steel", "godahewa2020elecdemand",
                       "godahewa2020solar,nrelSolarIntegration", "salam2018tetouan"]))
BASELINES = ("N", "PatchTST", "TimeXer", "TQNet")
DECODERS = {"adapt_HES": "H+E+T", "adapt_HE": "H+E", "adapt_HS": "H+T", "adapt_ES": "E+T"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def intervals(values, datasets):
    # Shared source collections stay together; the point estimate weights all 39 tasks equally.
    groups = {d: ("ETT" if d in "GH" else "GEFCom" if d in "LM" else d) for d in ORDER}
    cluster = np.array([groups[d] for d in datasets])
    labels = sorted(set(cluster))
    chunks = [np.asarray(values)[cluster == key] for key in labels]
    sums, counts = np.array([v.sum() for v in chunks]), np.array([len(v) for v in chunks])
    draws = np.random.default_rng(7112026).integers(0, len(labels), size=(20000, len(labels)))
    means = sums[draws].sum(1)/counts[draws].sum(1)
    return np.quantile(means, [.025, .975]).tolist()


def main():
    for folder in (DATA / "errors", DATA / "figure_data", GEN):
        folder.mkdir(parents=True, exist_ok=True)
    for folder in ("sera_numeric_20261109", "sera_numeric_extension_20261109"):
        assert read(ROOT / folder / "audit.json")["status"] == "verified"
    registry = {row["key"]: row for row in read(ROOT / "segan_submission_20261109/public_repository/dataset_registry.json")}
    extras = dict(H=("ETTm2", 69680, 96, "OT", ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
                     "https://github.com/zhouhaoyi/ETDataset", "ETTm2.csv", "native 15-minute observations"),
                  L=("GEFCom2014 price", 25944, 24, "zonal_price", ["forecast_total_load", "forecast_zonal_load"],
                     "https://doi.org/10.1016/j.ijforecast.2016.02.001", "gefcom_price.csv",
                     "Task 15 price records retained in competition order, represented by monotonic hourly slots"),
                  Q=("BDG2 Hog", 17544, 168, "site_electricity", ["airTemperature", "dewTemperature", "seaLvlPressure", "windSpeed"],
                     "https://github.com/buds-lab/building-data-genome-project-2", "bdg2_hog.csv",
                     "sum of 49 fixed Hog-site electricity meters complete across the source period; weather forward-filled at most three hours before contiguous-block selection"))
    for ds, (name, length, lookback, target, ext, url, filename, aggregation) in extras.items():
        meta = read(ROOT / "s_extension_20261108/prepared" / f"{ds}_h{4 if ds == 'H' else 1}.json")
        horizons = [4, 48, 96] if ds == "H" else [1, 12, 24]
        counts = []
        for h in horizons:
            with np.load(ROOT / "s_extension_20261108/prepared" / f"{ds}_h{h}.npz", allow_pickle=False) as data:
                counts.append([len(data[f"target_{split}"]) for split in ("train", "val", "cal", "test")])
        registry[ds] = dict(key=ds, name=name, input_sha256=meta["source_sha256"], rows_after_loader=length,
                            interval_minutes=15 if ds == "H" else 60, hist_len=lookback, target=target,
                            target_unit=None, unit_status="native recorded numerical scale retained", source_url=url,
                            covariates=ext, aggregation=aggregation, native_horizons=horizons,
                            split_counts=counts, processed_input_filename=filename)
    save(DATA / "dataset_registry.json", [registry[d] for d in ORDER])
    main_rows, controls, costs, provenance, examples = [], [], [], [], []
    old_baselines = pd.read_csv(ROOT / "s_extension_20261108/verified_seed_metrics.csv")
    for ds in ORDER:
        folder = ROOT / ("sera_numeric_extension_20261109" if ds in extras else "sera_numeric_20261109")
        for h in registry[ds]["native_horizons"]:
            hours = h//4 if ds in "GH" else h
            for seed in (41, 44):
                cell = folder / "cells" / f"{ds}_h{h}_s{seed}"
                marker = read(cell / "complete.json")
                source = cell / "predictions.npz"
                assert sha(source) == marker["predictions_sha256"]
                with np.load(source, allow_pickle=False) as saved:
                    truth = saved["test_y"].copy()
                    target = saved["test_target"].copy()
                    dra = saved["test_adapt_ES"].copy()
                    for split in ("cal", "test"):
                        for stored, label in DECODERS.items():
                            error = saved[f"{split}_{stored}"]-saved[f"{split}_y"]
                            path = DATA / "errors" / f"decoder_{stored}_{ds}_h{h}_s{seed}_{split}.npz"
                            np.savez_compressed(path, signed_error=error, absolute_error=abs(error),
                                                target_index=saved[f"{split}_target"], origin_index=saved[f"{split}_target"]-h)
                            controls.append(dict(dataset=ds, horizon=h, horizon_hours=hours, seed=seed, split=split,
                                model=label, MAE=float(abs(error).mean()), samples=len(error), error_archive=str(path.relative_to(DATA)).replace('\\','/')))
                for stage in ("encoder", "latent", "context", "anchor", "adapt_ES"):
                    costs.append(dict(dataset=ds, horizon=h, seed=seed, stage=stage, **marker["training"][stage]))
                selected = old_baselines[(old_baselines.dataset == ds) & (old_baselines.horizon == h) & (old_baselines.seed == seed)]
                predictions = {"DRA": (dra, source, marker["predictions_sha256"])}
                for row in selected[selected.model.isin(BASELINES)].itertuples():
                    assert sha(row.source) == row.source_sha256
                    with np.load(row.source, allow_pickle=False) as baseline:
                        value_key, truth_key = ("pred_test", "y_test") if "pred_test" in baseline else ("mean", "y_true")
                        np.testing.assert_array_equal(baseline["target_test"].reshape(-1), target)
                        np.testing.assert_array_equal(baseline[truth_key].reshape(-1), truth)
                        value = baseline[value_key].reshape(-1).astype(float)
                    np.testing.assert_allclose(abs(value-truth).mean(), row.MAE, rtol=1e-12, atol=1e-10)
                    predictions[row.model] = (value, Path(row.source), row.source_sha256)
                assert set(predictions) == {"DRA", *BASELINES}
                for model, (values, raw_path, digest) in predictions.items():
                    error = values-truth
                    path = DATA / "errors" / f"{model}_{ds}_h{h}_s{seed}.npz"
                    np.savez_compressed(path, signed_error=error, absolute_error=abs(error), target_index=target, origin_index=target-h)
                    main_rows.append(dict(model=model, dataset=ds, horizon=h, horizon_hours=hours, seed=seed,
                        MAE=float(abs(error).mean()), samples=len(error), error_archive=str(path.relative_to(DATA)).replace('\\','/')))
                    provenance.append(dict(error_archive=str(path.relative_to(DATA)).replace('\\','/'), original_source=str(raw_path), original_sha256=digest))
                if seed == 41 and (ds, hours) in (("O", 1), ("N", 24)):
                    for i in range(min(168, len(truth))):
                        examples.append(dict(dataset=ds, horizon_hours=hours, target_index=int(target[i]), observed=float(truth[i]),
                                             DRA=float(dra[i]), TQNet=float(predictions["TQNet"][0][i])))
    primary = pd.DataFrame(main_rows)
    primary.to_csv(DATA / "primary_seed_mae.csv", index=False)
    assert len(primary) == 390 and not primary.duplicated(["model", "dataset", "horizon", "seed"]).any()
    task = primary.groupby(["dataset", "horizon", "horizon_hours", "model"], as_index=False).MAE.mean()
    wide = task.pivot(index=["dataset", "horizon", "horizon_hours"], columns="model", values="MAE").reset_index()
    summary = []
    for baseline in BASELINES:
        wide[f"vs_{baseline}_pct"] = 100*(1-wide.DRA/wide[baseline])
        gain = wide[f"vs_{baseline}_pct"]
        low, high = intervals(gain, wide.dataset)
        summary.append(dict(comparator=baseline, reduction_pct=float(gain.mean()), lower=low, upper=high,
                            task_wins=int((gain > 1e-4).sum()), dataset_wins=int((wide.assign(gain=gain).groupby("dataset").gain.mean() > 0).sum())))
    wide.to_csv(DATA / "task_mae.csv", index=False)
    pd.DataFrame(summary).to_csv(DATA / "expected_primary_summary.csv", index=False)
    by_dataset = wide.groupby("dataset")[[f"vs_{b}_pct" for b in BASELINES]].mean().reindex(ORDER)
    by_dataset.to_csv(DATA / "dataset_reductions.csv")
    wide.groupby("horizon_hours")[[f"vs_{b}_pct" for b in BASELINES]].mean().to_csv(DATA / "horizon_reductions.csv")
    control = pd.DataFrame(controls)
    control.to_csv(DATA / "decoder_seed_mae.csv", index=False)
    control_task = control.groupby(["dataset", "horizon", "horizon_hours", "split", "model"], as_index=False).MAE.mean()
    control_task.to_csv(DATA / "decoder_task_mae.csv", index=False)
    control_summary = []
    for split in ("cal", "test"):
        p = control_task[control_task.split == split].pivot(index=["dataset", "horizon"], columns="model", values="MAE")
        for model in DECODERS.values():
            increase = 100*(p[model]/p["H+E+T"]-1)
            low, high = intervals(increase, p.index.get_level_values("dataset"))
            control_summary.append(dict(split=split, model=model, mean_MAE_increase_pct=float(increase.mean()),
                                        lower=low, upper=high, tasks=len(increase)))
    pd.DataFrame(control_summary).to_csv(DATA / "decoder_summary.csv", index=False)
    decoder_rows = [r"\begin{tabular}{lrrr}", r"\toprule",
                    r"Outputs & Development change (\%) & Test change (\%) & Test interval (\%) \\", r"\midrule"]
    for model in DECODERS.values():
        dev = next(v for v in control_summary if v["split"] == "cal" and v["model"] == model)
        test = next(v for v in control_summary if v["split"] == "test" and v["model"] == model)
        decoder_rows.append(f"${model}$ & {dev['mean_MAE_increase_pct']:.3f} & {test['mean_MAE_increase_pct']:.3f} & [{test['lower']:.2f}, {test['upper']:.2f}]"+r" \\")
    (GEN / "decoder_summary.tex").write_text("\n".join(decoder_rows+[r"\bottomrule", r"\end{tabular}"])+"\n", encoding="utf-8")
    top = by_dataset.sort_values("vs_N_pct", ascending=False).head(3)
    findings = "The largest series-average reductions relative to N occur in "+", ".join(
        f"{NAMES[ds]} ({row.vs_N_pct:.2f}\\%)" for ds, row in top.iterrows())+". "
    horizon = wide.groupby("horizon_hours")[["vs_N_pct", "vs_TQNet_pct"]].mean()
    findings += "At the 1-, 12- and 24-hour leads, the mean reductions against N are "+", ".join(
        f"{horizon.loc[h, 'vs_N_pct']:.2f}\\%" for h in (1, 12, 24))+", respectively. "
    findings += "The corresponding TQNet reductions are "+", ".join(
        f"{horizon.loc[h, 'vs_TQNet_pct']:.2f}\\%" for h in (1, 12, 24))+".\n"
    (GEN / "dataset_findings.tex").write_text(findings, encoding="utf-8")
    pd.DataFrame(costs).to_csv(DATA / "training_cost_cells.csv", index=False)
    save(HERE / "qa/local_prediction_provenance.json", provenance)
    pd.DataFrame(examples).to_csv(HERE / "qa/forecast_examples_with_observations.csv", index=False)
    figure_errors = pd.DataFrame(examples).drop(columns=["observed", "DRA", "TQNet"])
    figure_errors["DRA_signed_error"] = [v["DRA"]-v["observed"] for v in examples]
    figure_errors["TQNet_signed_error"] = [v["TQNet"]-v["observed"] for v in examples]
    figure_errors.to_csv(DATA / "figure_data/forecast_example_errors.csv", index=False)
    data_lines = [r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xrrrrrr@{}}", r"\toprule",
                  r"Series & Interval (min) & $L$ & $d_x$ & 1 h & 12 h & 24 h \\", r"\midrule"]
    for ds in ORDER:
        row = registry[ds]
        numbers = [row["interval_minutes"], row["hist_len"], len(row["covariates"]), *[v[3] for v in row["split_counts"]]]
        data_lines.append(NAMES[ds]+r" \citep{"+CITES[ds]+"} & "+" & ".join(map(str, numbers))+r" \\")
    (GEN / "datasets.tex").write_text("\n".join(data_lines+[r"\bottomrule", r"\end{tabularx}"])+"\n", encoding="utf-8")
    rows = [r"\begin{tabular}{lrrr}", r"\toprule", r"Comparator & MAE reduction (\%) & Task wins & Series wins \\", r"\midrule"]
    for row in summary:
        rows.append(f"{('Numerical gate N' if row['comparator'] == 'N' else row['comparator'])} & {row['reduction_pct']:.2f} & {row['task_wins']}/39 & {row['dataset_wins']}/13"+r" \\")
    (GEN / "model_summary.tex").write_text("\n".join(rows+[r"\bottomrule", r"\end{tabular}"])+"\n", encoding="utf-8")
    split_rows = [r"\begin{longtable}{llrrrr}", r"\caption{Chronological sample counts.}\label{tab:splits}\\",
                  r"\toprule Series & Lead (h) & Train & Validation & Development & Test \\ \midrule\endfirsthead",
                  r"\toprule Series & Lead (h) & Train & Validation & Development & Test \\ \midrule\endhead"]
    error_rows = [r"\begin{longtable}{llrrrrr}", r"\caption{Seed-mean test MAE in each target's native recorded scale.}\label{tab:task_mae}\\",
                  r"\toprule Series & Lead (h) & DRA & N & PatchTST & TimeXer & TQNet \\ \midrule\endfirsthead",
                  r"\toprule Series & Lead (h) & DRA & N & PatchTST & TimeXer & TQNet \\ \midrule\endhead"]
    for ds in ORDER:
        for i, hours in enumerate((1, 12, 24)):
            split_rows.append(f"{NAMES[ds]} & {hours} & "+" & ".join(map(str, registry[ds]["split_counts"][i]))+r" \\")
            row = wide[(wide.dataset == ds) & (wide.horizon_hours == hours)].iloc[0]
            values = [row[m] for m in ("DRA", *BASELINES)]
            cells = [f"{v:.5g}" for v in values]
            cells[int(np.argmin(values))] = r"\textbf{"+cells[int(np.argmin(values))]+"}"
            error_rows.append(f"{NAMES[ds]} & {hours} & "+" & ".join(cells)+r" \\")
    for filename, rows in (("split_counts", split_rows), ("task_mae", error_rows)):
        (GEN / (filename+".tex")).write_text("\n".join(rows+[r"\bottomrule\end{longtable}"])+"\n", encoding="utf-8")
    save(HERE / "qa/results_audit.json", dict(status="verified", series=13, tasks=39, seeds=[41, 44],
        primary_records=len(primary), decoder_records=len(control), matching_truth_and_target_indices=True,
        interval_draws=20000, interval_seed=7112026, bootstrap_source_groups=11, summary=summary,
        original_outputs_unchanged=True, new_llm_calls=0))
    print(json.dumps(dict(summary=summary, decoder_summary=control_summary), indent=2))


if __name__ == "__main__":
    main()
