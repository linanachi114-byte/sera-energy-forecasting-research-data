"""Expand the lagged-covariate S decoder using cached full-scope encodings."""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import exogenous_context as exo
import latent_full as full
import latent_heads as head
import run as s
from cpu_heads import WeightedPointLoss
from sklearn.preprocessing import MinMaxScaler


def context(dataset,horizon,base,splits=("train","val","cal")):
    df,_,columns,_=s.exp.load_research_dataframe(dataset,data_fraction=1.,fraction_mode="full")
    raw=df[columns].to_numpy(dtype=s.np.float32)
    scaler=MinMaxScaler().fit(raw[base["semantic_train"]])
    values=scaler.transform(raw).astype(s.np.float32)
    result={}
    for split in splits:
        origins=base[f"semantic_{split}"].astype(s.np.int64)
        s.np.testing.assert_array_equal(origins,base[f"target_{split}"]-horizon)
        s.np.testing.assert_array_equal(values[origins],base[f"ext_{split}"])
        indices=s.np.maximum(origins[:,None]-s.np.arange(23,-1,-1)[None,:],0)
        assert (indices<=origins[:,None]).all()
        history=values[indices]
        result[f"ext_{split}"]=s.np.column_stack((base[f"ext_{split}"],history.mean(1)-values[origins],history[:,-1]-history[:,0],history.std(1))).astype(s.np.float32)
    return result


def main():
    s.torch.set_num_threads(1)
    s.exp.DEVICE=s.torch.device("cpu")
    s.exp.PointOnlyLoss=WeightedPointLoss
    s.exp.train_model=head.train
    s.SemanticModel=exo.Candidate
    stage="exogenous_full"
    state=dict(status="waiting_for_full_calibration",pid=os.getpid(),started_unix=time.time())
    s.exp.atomic_json(s.HERE/f"{stage}_status.json",state)
    deadline=time.monotonic()+18000
    while True:
        source=json.loads((s.HERE/"latent_full_status.json").read_text(encoding="utf-8"))
        audit=s.HERE/"latent_full_audit.json"
        if source["status"]=="failed":
            raise RuntimeError(source)
        if source["status"]=="complete" and audit.exists():
            assert json.loads(audit.read_text(encoding="utf-8"))["status"]=="verified"
            break
        assert time.monotonic()<deadline,"Source wait expired"
        time.sleep(15)
    source_plan=json.loads((s.HERE/"latent_full_plan.json").read_text(encoding="utf-8"))
    primary=json.loads((s.HERE/"latent_full_summary.json").read_text(encoding="utf-8"))["results"][0]
    selected=primary["variant"]
    table=s.pd.read_csv(s.HERE/"latent_full_metrics.csv")
    table=table[table.variant==selected]
    assert len(table)==60 and primary["tasks"]==30
    paths=[Path(__file__),s.HERE/"EXOGENOUS_FULL_PLAN.md",s.HERE/"exogenous_context.py",s.HERE/"latent_full.py",s.HERE/"latent_heads.py",s.HERE/"run.py",
           s.HERE/"latent_full_metrics.csv",s.HERE/"latent_full_summary.json",s.HERE/"latent_full_plan.json",s.HERE/"latent_full_audit.json",
           s.HERE/"exogenous_context_pilot_summary.json"]
    common=dict(run_sha256=s.exp.file_sha(Path(__file__)),sources={str(p):s.exp.file_sha(p) for p in paths})
    plan=dict(stage=stage,identity=common,selected=selected,variants=["w24_stats"],datasets=source_plan["datasets"],
              horizons=source_plan["horizons"],seeds=[41,44],llm_calls=0,test_metrics=False)
    frozen=s.HERE/f"{stage}_plan.json"
    if frozen.exists():
        assert json.loads(frozen.read_text(encoding="utf-8"))==plan
    else:
        s.exp.atomic_json(frozen,plan)
    state.update(status="running",selected=selected,completed_cells=0,total_cells=60,updated_unix=time.time())
    s.exp.atomic_json(s.HERE/f"{stage}_status.json",state)
    rows=[]
    try:
        for row in table.itertuples():
            base,_,refs=full.pack(row.dataset,int(row.horizon),int(row.seed),source_plan["identity"])
            checkpoint=s.HERE/"cells/latent_full"/f"{row.dataset}_h{row.horizon}_s{row.seed}_{selected}.pt"
            marker=checkpoint.with_suffix(".complete.json")
            old=json.loads(marker.read_text(encoding="utf-8"))
            assert old["checkpoint_sha256"]==s.exp.file_sha(checkpoint)
            raw_path=(s.exp.RESEARCH/str(s.exp.DATASETS[row.dataset]["filepath"])).resolve()
            exo.START_STATE=s.torch.load(checkpoint,map_location="cpu",weights_only=False)["best_state"]
            exo.BASE_D=base["ext_train"].shape[1]
            arrays={**base,**context(row.dataset,int(row.horizon),base)}
            for split in ("train","val","cal"):
                arrays[f"sem_{split}"]=base[f"sem_{split}"].copy()
                if selected=="context_raw":
                    arrays[f"sem_{split}"][:,1:-1]=0
                arrays[f"y_{split}"]=s.np.column_stack((base[f"y_{split}"].reshape(-1),base[f"scale_{split}"]/base["scale_train"].mean())).astype(s.np.float32)
            inverses={split:s.norm.WindowInverse(base[f"center_{split}"],base[f"scale_{split}"]) for split in ("train","val","cal")}
            initial=exo.Candidate({},arrays,"w24_stats")
            replay,_=s.exp.base.predict_point(initial,arrays,"cal",inverses["cal"],s.exp.DEVICE)
            with s.np.load(checkpoint.with_suffix(".npz"),allow_pickle=False) as f:
                s.np.testing.assert_allclose(replay,f["pred_cal"],rtol=1e-6,atol=1e-6)
            identity=dict(run_sha256=common["run_sha256"],sources={**common["sources"],**refs,**{str(p):s.exp.file_sha(p) for p in (checkpoint,marker,raw_path)}})
            state.update(current_cell=f"{row.dataset}_h{row.horizon}_s{row.seed}",updated_unix=time.time())
            s.exp.atomic_json(s.HERE/f"{stage}_status.json",state)
            rows.append(s.fit(row.dataset,int(row.horizon),int(row.seed),{},arrays,inverses,"w24_stats",stage,identity))
            s.summarize(rows,stage)
            state.update(completed_cells=len(rows),updated_unix=time.time())
            s.exp.atomic_json(s.HERE/f"{stage}_status.json",state)
            del arrays,base,initial
            gc.collect()
        state.update(status="complete",completed_unix=time.time())
        s.exp.atomic_json(s.HERE/f"{stage}_status.json",state)
    except Exception as error:
        state.update(status="failed",error=f"{type(error).__name__}: {error}")
        s.exp.atomic_json(s.HERE/f"{stage}_status.json",state)
        raise


if __name__=="__main__":
    with s.threadpool_limits(limits=1):
        main()
