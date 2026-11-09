"""Recompute manuscript statistics offline from the deposited numerical data."""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np

DATASETS = ("C", "G", "M", "N", "O", "P", "R", "S", "T", "U")


def read_csv(root, name):
    with (root / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["model"], row["dataset"], int(row["horizon"])].append(float(row["MAE"]))
    means = {key: float(np.mean(values)) for key, values in groups.items()}
    tasks = sorted({(ds, h) for model, ds, h in means})
    assert len(tasks) == 30 and {ds for ds, h in tasks} == set(DATASETS)
    output = []
    draws = np.random.default_rng(7112026).integers(0, 10, (20000, 10))
    for model in sorted({key[0] for key in means} - {"S_current"}):
        effects = {(ds, h): 100 * (1 - means["S_current", ds, h] / means[model, ds, h])
                   for ds, h in tasks}
        dataset_effects = np.array([
            np.mean([effects[ds, h] for d, h in tasks if d == ds]) for ds in DATASETS])
        low, high = np.percentile(dataset_effects[draws].mean(axis=1), [2.5, 97.5])
        output.append(dict(baseline=model, reduction_pct=float(np.mean(list(effects.values()))),
                           task_wins=int(sum(value > 0 for value in effects.values())),
                           dataset_wins=int(np.sum(dataset_effects > 0)),
                           descriptive_95_low=float(low), descriptive_95_high=float(high)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("reproduced_results.json"))
    args = parser.parse_args()
    root = args.data.resolve()
    manifest = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == expected, f"File hash mismatch: {name}"
    primary = read_csv(root, "primary_seed_mae.csv")
    five = read_csv(root, "five_seed_mae.csv")
    assert len(primary) == 540 and len(five) == 750
    checked = {}
    for row in primary + five:
        name = row["error_archive"]
        if name not in checked:
            with np.load(root / name, allow_pickle=False) as saved:
                signed = saved["signed_error"]
                absolute = saved["absolute_error"]
                target = saved["target_index"]
            assert np.isfinite(absolute).all() and np.isfinite(signed).all()
            assert np.array_equal(absolute, np.abs(signed))
            assert target.size == absolute.size == int(row["samples"])
            assert len(np.unique(target)) == len(target)
            checked[name] = float(np.mean(absolute))
        assert abs(checked[name] - float(row["MAE"])) < 1e-10
    primary_summary, five_summary = summarize(primary), summarize(five)
    for result, expected_name, metric in [
        (primary_summary, "expected_primary_summary.csv", "S_reduction_pct"),
        (five_summary, "expected_five_seed_summary.csv", "task_macro_reduction_from_seed_means_pct"),
    ]:
        expected = {r["baseline"]: r for r in read_csv(root, expected_name)}
        assert set(expected) == {r["baseline"] for r in result}
        for row in result:
            source = expected[row["baseline"]]
            assert abs(row["reduction_pct"] - float(source[metric])) < 1e-10
            assert row["task_wins"] == int(source["task_wins"])
            assert row["dataset_wins"] == int(source["dataset_wins"])
    components = read_csv(root, "component_seed_mae.csv")
    assert len(components) == 360
    by_condition = defaultdict(list)
    for row in components:
        by_condition[row["variant"], row["dataset"], int(row["horizon"])].append(float(row["MAE"]))
    component_means = {key: np.mean(values) for key, values in by_condition.items()}
    component_summary = {}
    for variant in sorted({key[0] for key in component_means} - {"reference"}):
        effects = {key[1:]: 100 * (value / component_means["reference", key[1], key[2]] - 1)
                   for key, value in component_means.items() if key[0] == variant}
        assert len(effects) == 30
        component_summary[variant] = dict(
            mean_increase_pct=float(np.mean(list(effects.values()))),
            reference_task_wins=int(sum(value > 0 for value in effects.values())))
    report = dict(status="verified", primary_records=len(primary), five_seed_records=len(five),
                  verified_error_archives=len(checked), component_records=len(components),
                  primary=primary_summary, five_seed=five_summary, components=component_summary,
                  new_training_runs=0, new_llm_calls=0)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
