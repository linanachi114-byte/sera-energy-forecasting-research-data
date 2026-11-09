"""Give contextual S strictly lagged covariate dynamics, using a warm decoder."""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import latent_heads as head
import run as s
from cpu_heads import WeightedPointLoss
from sklearn.preprocessing import MinMaxScaler

SPECS={f"w{window}_{mode}":(window,mode) for window in (6,24,168) for mode in ("trend","stats")}
START_STATE=None
BASE_D=0


class Candidate(s.nn.Module):
    def __init__(self,cfg,arrays,variant):
        super().__init__()
        d=arrays["ext_train"].shape[1]
        self.ext_expert=s.exp.SnapshotCandidateExpert(d)
        self.semantic_expert=s.exp.SnapshotCandidateExpert(d+32+9+1)
        self.gate=s.nn.Sequential(s.nn.Linear(3+d+9+32+1,64),s.nn.GELU(),s.nn.Linear(64,3))
        self.reliability_slope_raw=s.nn.Parameter(START_STATE["reliability_slope_raw"].clone())
        values=self.state_dict()
        for name,value in values.items():
            old=START_STATE[name]
            if name in ("ext_expert.net.0.weight","semantic_expert.net.0.weight","gate.0.weight"):
                prefix=3+BASE_D if name=="gate.0.weight" else 1+BASE_D
                value.zero_()
                value[:,:prefix]=old[:,:prefix]
                if name!="ext_expert.net.0.weight":
                    value[:,prefix+d-BASE_D:]=old[:,prefix:]
            else:
                value.copy_(old)
        self.load_state_dict(values,strict=True)

    def forward(self,x_hist,x_ext,x_sem):
        h,z=x_hist[:,0,:1],x_hist[:,0,1:]
        content,rho=x_sem[:,:9],x_sem[:,9:10]
        ext=self.ext_expert(h,x_ext)
        semantic=self.semantic_expert(h,s.torch.cat((x_ext,z,content,s.torch.zeros_like(rho)),dim=1))
        candidates=s.torch.cat((h,ext,semantic),dim=1)
        logits=self.gate(s.torch.cat((candidates,x_ext,content,z,s.torch.zeros_like(rho)),dim=1))
        logits=s.torch.cat((logits[:,:2],logits[:,2:]+s.F.softplus(self.reliability_slope_raw)*(rho-.5)),dim=1)
        weights=s.torch.softmax(logits,dim=1)
        return {"mean":(weights*candidates).sum(1,keepdim=True),"alphas":weights}


def context(dataset,base,window,mode):
    df,_,columns,_=s.exp.load_research_dataframe(dataset,data_fraction=1.,fraction_mode="full")
    raw=df[columns].to_numpy(dtype=s.np.float32)
    scaler=MinMaxScaler().fit(raw[base["semantic_train"]])
    values=scaler.transform(raw).astype(s.np.float32)
    result={}
    for split in ("train","val","cal"):
        origins=base[f"semantic_{split}"].astype(s.np.int64)
        s.np.testing.assert_array_equal(origins,base[f"target_{split}"]-12)
        s.np.testing.assert_array_equal(values[origins],base[f"ext_{split}"])
        indices=s.np.maximum(origins[:,None]-s.np.arange(window-1,-1,-1)[None,:],0)
        assert (indices<=origins[:,None]).all()
        history=values[indices]
        parts=[history.mean(1)-values[origins],history[:,-1]-history[:,0]]
        if mode=="stats":
            parts.append(history.std(1))
        result[f"ext_{split}"]=s.np.column_stack((base[f"ext_{split}"],*parts)).astype(s.np.float32)
    return result


def main():
    global START_STATE,BASE_D
    s.torch.set_num_threads(1)
    s.exp.DEVICE=s.torch.device("cpu")
    s.exp.PointOnlyLoss=WeightedPointLoss
    s.exp.train_model=head.train
    s.SemanticModel=Candidate
    paths=[Path(__file__),s.HERE/"EXOGENOUS_CONTEXT_PLAN.md",s.HERE/"run.py",s.HERE/"latent_heads.py",s.HERE/"cpu_heads.py",
           Path(s.exp.__file__),Path(s.exp.base.__file__),Path(s.exp.trainer.__file__),Path(s.norm.__file__),s.exp.RESEARCH/"src/datasets/loaders.py"]
    common=dict(run_sha256=s.exp.file_sha(Path(__file__)),sources={str(p):s.exp.file_sha(p) for p in paths})
    stage="exogenous_context_pilot"
    plan=dict(stage=stage,identity=common,variants=list(SPECS),specifications=SPECS,datasets=list("CRSU"),seeds=[41],llm_calls=0,test_metrics=False)
    frozen=s.HERE/f"{stage}_plan.json"
    if frozen.exists():
        assert json.loads(frozen.read_text(encoding="utf-8"))==json.loads(json.dumps(plan))
    else:
        s.exp.atomic_json(frozen,plan)
    state=dict(status="running",pid=os.getpid(),completed_cells=0,total_cells=24,started_unix=time.time())
    s.exp.atomic_json(s.HERE/"exogenous_context_status.json",state)
    rows=[]
    try:
        for dataset in plan["datasets"]:
            bundle=s.HERE/"cells/latent_features"/f"{dataset}_h12_s41.npz"
            checkpoint=s.HERE/"cells/latent_pilot"/f"{dataset}_h12_s41_text_raw.pt"
            marker=checkpoint.with_suffix(".complete.json")
            raw_path=(s.exp.RESEARCH/str(s.exp.DATASETS[dataset]["filepath"])).resolve()
            old=json.loads(marker.read_text(encoding="utf-8"))
            assert old["checkpoint_sha256"]==s.exp.file_sha(checkpoint)
            with s.np.load(bundle,allow_pickle=False) as f:
                base={k:f[k].copy() for k in f.files}
            START_STATE=s.torch.load(checkpoint,map_location="cpu",weights_only=False)["best_state"]
            BASE_D=base["ext_train"].shape[1]
            inverses={split:s.norm.WindowInverse(base[f"center_{split}"],base[f"scale_{split}"]) for split in ("train","val","cal")}
            identity=dict(run_sha256=common["run_sha256"],sources={**common["sources"],**{str(p):s.exp.file_sha(p) for p in (bundle,checkpoint,marker,raw_path)}})
            for variant,(window,mode) in SPECS.items():
                arrays={**base,**context(dataset,base,window,mode)}
                for split in ("train","val","cal"):
                    arrays[f"y_{split}"]=s.np.column_stack((base[f"y_{split}"].reshape(-1),base[f"scale_{split}"]/base["scale_train"].mean())).astype(s.np.float32)
                initial=Candidate({},arrays,variant)
                reproduced,_=s.exp.base.predict_point(initial,arrays,"cal",inverses["cal"],s.exp.DEVICE)
                with s.np.load(checkpoint.with_suffix(".npz"),allow_pickle=False) as f:
                    s.np.testing.assert_allclose(reproduced,f["pred_cal"],rtol=1e-6,atol=1e-6)
                state.update(current_cell=f"{dataset}_h12_s41_{variant}",updated_unix=time.time())
                s.exp.atomic_json(s.HERE/"exogenous_context_status.json",state)
                rows.append(s.fit(dataset,12,41,{},arrays,inverses,variant,stage,identity))
                s.summarize(rows,stage)
                state.update(completed_cells=len(rows),updated_unix=time.time())
                s.exp.atomic_json(s.HERE/"exogenous_context_status.json",state)
            gc.collect()
        state.update(status="complete",completed_unix=time.time())
        s.exp.atomic_json(s.HERE/"exogenous_context_status.json",state)
    except Exception as error:
        state.update(status="failed",error=f"{type(error).__name__}: {error}")
        s.exp.atomic_json(s.HERE/"exogenous_context_status.json",state)
        raise


if __name__=="__main__":
    with s.threadpool_limits(limits=1):
        main()
