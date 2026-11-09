"""Train/validation/calibration features for the immutable verified S manifest."""
import json
from pathlib import Path

import latent_full as full
import exogenous_full as expanded
import split_anchor as pilot
import resume_dlinear_screen as windows
import run as s


def pack(ds, h, seed):
    mp = s.HERE/"S_external_anchor_candidates_manifest.json"
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    item = next(r for r in manifest["cells"] if (r["dataset"], r["horizon"], r["seed"]) == (ds, h, seed))
    original = json.loads((s.HERE/"latent_full_plan.json").read_text(encoding="utf-8"))
    base, _, refs = full.pack(ds, h, seed, original["identity"])
    observed, cfg = windows.observed(ds, base)
    arrays = {**base, **expanded.context(ds, h, base)}
    inverses = {k: s.norm.WindowInverse(base[f"center_{k}"], base[f"scale_{k}"]) for k in ("train", "val", "cal")}
    point_path, cp = Path(item["point"]), Path(item["decoder"])
    for path in (point_path, cp):
        marker = path.with_suffix(".complete.json")
        assert s.exp.file_sha(path) == json.loads(marker.read_text(encoding="utf-8"))["checkpoint_sha256"]
        refs.update({str(p): s.exp.file_sha(p) for p in (path, marker)})
    point = pilot.shared.light.Point(cfg, arrays, "point")
    point.load_state_dict(s.torch.load(point_path, map_location="cpu", weights_only=False)["best_state"], strict=True)
    point.eval()
    with s.torch.no_grad():
        for split in ("train", "val", "cal"):
            values = []
            for start in range(0, len(observed[f"x_{split}"]), 512):
                batch = s.torch.as_tensor(observed[f"x_{split}"][start:start+512], dtype=s.torch.float32)
                values.append(point(batch, None, None)["mean"].numpy().reshape(-1))
            arrays[f"x_{split}"] = s.np.concatenate((base[f"x_{split}"], s.np.concatenate(values)[:, None, None]), axis=2)
            arrays[f"y_{split}"] = s.np.column_stack((base[f"y_{split}"].reshape(-1), base[f"scale_{split}"]/base["scale_train"].mean())).astype(s.np.float32)
    pilot.shared.exo.START_STATE = s.torch.load(cp, map_location="cpu", weights_only=False)["best_state"]
    pilot.shared.exo.BASE_D = arrays["ext_train"].shape[1]
    model = pilot.Candidate(cfg, arrays, "external_only").eval()
    pred, _ = s.exp.base.predict_point(model, arrays, "cal", inverses["cal"], s.exp.DEVICE)
    source = Path(item["S"])
    assert s.exp.file_sha(source) == item["S_sha256"]
    with s.np.load(source, allow_pickle=False) as f:
        s.np.testing.assert_array_equal(f["y_cal"], base["y_cal_raw"].reshape(-1))
        s.np.testing.assert_array_equal(f["target_cal"], base["target_cal"])
        s.np.testing.assert_allclose(pred, f["pred_cal"], rtol=1e-6, atol=1e-6)
    raw = (s.exp.RESEARCH/str(s.exp.DATASETS[ds]["filepath"])).resolve()
    refs.update({str(p): s.exp.file_sha(p) for p in (Path(__file__), mp, source, raw, Path(windows.__file__), Path(expanded.__file__))})
    return arrays, inverses, refs
