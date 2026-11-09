"""Fast frozen-S head hypotheses using cached features, no GPU model fitting."""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import run as s
import run_reasons as reasons

SPECS={
    "zero_norm":(False,False,0.,1.5e-4),
    "zero_raw":(False,False,1.,1.5e-4),
    "jointzero_norm":(True,False,0.,1.5e-4),
    "jointzero_raw":(True,False,1.,1.5e-4),
    "text_norm":(False,True,0.,1.5e-4),
    "text_raw":(False,True,1.,1.5e-4),
    "jointtext_norm":(True,True,0.,1.5e-4),
    "jointtext_raw":(True,True,1.,1.5e-4),
    "jointtext_half":(True,True,.5,1.5e-4),
    "jointtext_raw_lrhalf":(True,True,1.,7.5e-5),
    "jointzero_raw_lrhalf":(True,False,1.,7.5e-5),
    "zero_raw_lrhalf":(False,False,1.,7.5e-5),
}
ACTIVE_LR=1.5e-4
ORIGINAL_TRAIN=s.exp.train_model


class WeightedPointLoss:
    def __init__(self,kind):
        pass

    def __call__(self,prediction,target):
        assert target.ndim==2 and target.shape[1]==2
        loss=(s.torch.abs(prediction["mean"]-target[:,:1])*target[:,1:2]).mean()
        return {"total":loss}


class Candidate(reasons.ReasonCandidate):
    def __init__(self,cfg,arrays,variant):
        joint,_,_,_=SPECS[variant]
        super().__init__(cfg,arrays,"joint_text" if joint else "text")


def train(*args,**kwargs):
    kwargs["lr"]=ACTIVE_LR
    kwargs["batch_size"]=128
    return ORIGINAL_TRAIN(*args,**kwargs)


def wait_for_source(dataset):
    marker=s.HERE/"cells/reason_pilot"/f"{dataset}_h12_s41_text.complete.json"
    deadline=time.monotonic()+1800
    while not marker.exists():
        status=s.HERE/"reason_status.json"
        if status.exists():
            state=json.loads(status.read_text(encoding="utf-8"))
            assert state["status"]!="failed",state
        assert time.monotonic()<deadline,f"Source feature wait expired: {dataset}"
        time.sleep(3)
    return marker


def main():
    global ACTIVE_LR
    s.torch.set_num_threads(1)
    s.exp.DEVICE=s.torch.device("cpu")
    s.exp.PointOnlyLoss=WeightedPointLoss
    s.exp.train_model=train
    s.SemanticModel=Candidate
    # Verify the weighted target layout and gradient before using cached labels.
    probe=s.torch.tensor([[2.],[4.]],requires_grad=True)
    loss=WeightedPointLoss("mae")({"mean":probe},s.torch.tensor([[1.,.5],[1.,1.5]]))["total"]
    assert float(loss.detach())==2.5
    loss.backward()
    s.torch.testing.assert_close(probe.grad,s.torch.tensor([[.25],[.75]]))
    sources=[Path(__file__),s.HERE/"CPU_HEAD_PLAN.md",s.HERE/"run.py",s.HERE/"run_reasons.py",
        Path(s.exp.__file__),Path(s.norm.__file__),Path(s.exp.base.__file__),
        Path(s.exp.trainer.__file__),s.exp.RESEARCH/"src/models/official_forecasting.py"]
    common=dict(run_sha256=s.exp.file_sha(Path(__file__)),sources={str(p):s.exp.file_sha(p) for p in sources})
    stage="cpu_head"
    plan=dict(identity=common,stage=stage,variants=list(SPECS),specifications={k:list(v) for k,v in SPECS.items()},
        datasets=["C","R","S","U"],seeds=[41],device="cpu",llm_calls=0,test_metrics=False)
    frozen=s.HERE/f"{stage}_plan.json"
    if frozen.exists():
        assert json.loads(frozen.read_text(encoding="utf-8"))==plan
    else:
        s.exp.atomic_json(frozen,plan)
    state=dict(status="running",pid=os.getpid(),stage=stage,completed_cells=0,
               total_cells=48,started_unix=time.time())
    s.exp.atomic_json(s.HERE/"cpu_head_status.json",state)
    rows=[]
    try:
        for dataset in plan["datasets"]:
            marker=wait_for_source(dataset)
            bundle=s.HERE/"cells/head_features"/f"{dataset}_h12.npz"
            text_path=bundle.with_name(f"{dataset}_h12_text.npz")
            checkpoint=s.ROOT/"ns_instance_normalization_20261023/all_cal/revin_mean_std/checkpoints"/f"{dataset}_h12_rasg_s41.pt"
            with s.np.load(bundle,allow_pickle=False) as f:
                base={k:f[k].copy() for k in f.files}
            with s.np.load(text_path,allow_pickle=False) as f:
                text={k:f[k].copy() for k in f.files}
            saved=s.torch.load(checkpoint,map_location="cpu",weights_only=False)
            assert saved["complete"]
            reasons.OLD_STATE=saved["best_state"]
            inverses={split:s.norm.WindowInverse(base[f"center_{split}"],base[f"scale_{split}"])
                      for split in ("train","val","cal")}
            identity=dict(run_sha256=common["run_sha256"],sources={**common["sources"],
                **{str(p):s.exp.file_sha(p) for p in (marker,bundle,text_path,checkpoint)}})
            cfg={"hist_len":24 if dataset=="C" else 168}
            for variant,(_,use_text,power,lr) in SPECS.items():
                ACTIVE_LR=lr
                arrays=dict(base)
                divisor=float(s.np.mean(base["scale_train"]**power))
                for split in ("train","val","cal"):
                    sem=text[f"sem_{split}"].copy()
                    if not use_text:
                        sem[:,1:-1]=0
                    arrays[f"sem_{split}"]=sem
                    weight=(base[f"scale_{split}"]**power/divisor).astype(s.np.float32)
                    arrays[f"y_{split}"]=s.np.column_stack((base[f"y_{split}"].reshape(-1),weight))
                state.update(current_cell=f"{dataset}_h12_s41_{variant}",updated_unix=time.time())
                s.exp.atomic_json(s.HERE/"cpu_head_status.json",state)
                metric=s.fit(dataset,12,41,cfg,arrays,inverses,variant,stage,identity)
                rows.append(metric)
                s.summarize(rows,stage)
                state.update(completed_cells=len(rows),updated_unix=time.time())
                s.exp.atomic_json(s.HERE/"cpu_head_status.json",state)
                print(json.dumps(metric),flush=True)
            del base,text,saved,arrays
            gc.collect()
        s.summarize(rows,stage)
        state.update(status="complete",completed_unix=time.time())
        s.exp.atomic_json(s.HERE/"cpu_head_status.json",state)
    except Exception as error:
        state.update(status="failed",error=f"{type(error).__name__}: {error}")
        s.exp.atomic_json(s.HERE/"cpu_head_status.json",state)
        raise


if __name__=="__main__":
    with s.threadpool_limits(limits=1):
        main()
