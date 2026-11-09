"""Offline reconstruction of DRA's primary and matched-decoder MAE tables."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def intervals(values, datasets):
    groups = {d: ("ETT" if d in "GH" else "GEFCom" if d in "LM" else d) for d in "CGHLMNOPQRSTU"}
    cluster = np.array([groups[d] for d in datasets])
    chunks = [np.asarray(values)[cluster == key] for key in sorted(set(cluster))]
    sums, counts = np.array([v.sum() for v in chunks]), np.array([len(v) for v in chunks])
    indices = np.random.default_rng(7112026).integers(0, len(chunks), size=(20000, len(chunks)))
    means = sums[indices].sum(1)/counts[indices].sum(1)
    return np.quantile(means, [.025, .975])


def reconstruct(filename):
    records = pd.read_csv(HERE / filename)
    for row in records.itertuples():
        with np.load(HERE / row.error_archive, allow_pickle=False) as error:
            assert set(error.files) == {"signed_error", "absolute_error", "target_index", "origin_index"}
            signed = error["signed_error"]
            assert len(signed) == row.samples and np.isfinite(signed).all()
            np.testing.assert_array_equal(abs(signed), error["absolute_error"])
            np.testing.assert_array_equal(error["target_index"]-row.horizon, error["origin_index"])
            np.testing.assert_allclose(abs(signed).mean(), row.MAE, rtol=1e-12, atol=1e-12)
        key = (row.dataset, row.horizon, row.seed, getattr(row, "split", "test"))
        if key in ALIGNMENT:
            np.testing.assert_array_equal(ALIGNMENT[key], error_indices(HERE / row.error_archive))
        else:
            ALIGNMENT[key] = error_indices(HERE / row.error_archive)
    return records


def error_indices(path):
    with np.load(path, allow_pickle=False) as data:
        return data["target_index"].copy()


ALIGNMENT = {}


def main():
    manifest = HERE / "SHA256SUMS.json"
    if manifest.exists():
        for filename, digest in json.loads(manifest.read_text()).items():
            assert hashlib.sha256((HERE / filename).read_bytes()).hexdigest() == digest, filename
    primary = reconstruct("primary_seed_mae.csv")
    assert len(primary) == 390
    task = primary.groupby(["dataset", "horizon", "model"], as_index=False).MAE.mean()
    wide = task.pivot(index=["dataset", "horizon"], columns="model", values="MAE")
    expected = pd.read_csv(HERE / "expected_primary_summary.csv")
    for row in expected.itertuples():
        gain = 100*(1-wide.DRA/wide[row.comparator])
        np.testing.assert_allclose(gain.mean(), row.reduction_pct, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(intervals(gain, wide.index.get_level_values("dataset")), [row.lower, row.upper], rtol=1e-12, atol=1e-12)
        assert int((gain > 1e-4).sum()) == row.task_wins
    controls = reconstruct("decoder_seed_mae.csv")
    assert len(controls) == 624
    task = controls.groupby(["dataset", "horizon", "split", "model"], as_index=False).MAE.mean()
    expected = pd.read_csv(HERE / "decoder_summary.csv")
    for split in ("cal", "test"):
        wide = task[task.split == split].pivot(index=["dataset", "horizon"], columns="model", values="MAE")
        for row in expected[expected.split == split].itertuples():
            increase = 100*(wide[row.model]/wide["H+E+T"]-1)
            np.testing.assert_allclose(increase.mean(), row.mean_MAE_increase_pct, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(intervals(increase, wide.index.get_level_values("dataset")), [row.lower, row.upper], rtol=1e-12, atol=1e-12)
    print(json.dumps(dict(status="verified", primary_records=390, decoder_records=624,
                          series=13, tasks=39, seeds=[41, 44], bootstrap_draws=20000), indent=2))


if __name__ == "__main__":
    main()
