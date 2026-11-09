"""Frozen encoder context for S experts and gates, with epoch-zero selection."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path

import run as s
from cpu_heads import WeightedPointLoss

SPECS={f"{kind}_{loss}":(kind,power) for kind in
       ("point","context","text","film","filmzero","filmshuffle")
       for loss,power in (("norm",0.),("raw",1.))}
OLD_STATE=None
ORIGINAL_TRAIN=s.exp.train_model


class Candidate(s.nn.Module):
    def __init__(self,cfg,arrays,variant):
        super().__init__()
        self.kind=SPECS[variant][0]
        d=arrays["ext_train"].shape[1]
        self.ext_dim=d
        self.ext_expert=s.exp.SnapshotCandidateExpert(d)
        self.semantic_expert=s.exp.SnapshotCandidateExpert(d+32+9+1)
        self.gate=s.nn.Sequential(s.nn.Linear(3+d+9+32+1,64),s.nn.GELU(),s.nn.Linear(64,3))
        self.reliability_slope_raw=s.nn.Parameter(OLD_STATE["reliability_slope_raw"].clone())
        if self.kind.startswith("film"):
            self.modulation=s.nn.Linear(10,64)
            s.nn.init.zeros_(self.modulation.weight)
            s.nn.init.zeros_(self.modulation.bias)
        state=self.state_dict()
        for name,value in state.items():
            if name.startswith("modulation."):
                continue
            old=OLD_STATE[name]
            if name=="semantic_expert.net.0.weight":
                value.zero_()
                value[:,0]=old[:,0]
                value[:,1+d+32]=old[:,1]
                value[:,-1]=old[:,-1]
            elif name=="gate.0.weight":
                value.zero_()
                value[:,:old.shape[1]-1]=old[:,:-1]
                value[:,-1]=old[:,-1]
            else:
                value.copy_(old)
        self.load_state_dict(state,strict=True)

    def forward(self,x_hist,x_ext,x_sem):
        h=x_hist[:,0,:1]
        z=x_hist[:,0,1:]
        sem=x_sem[:,:10]
        content,rho=sem[:,:9],sem[:,9:10]
        if self.kind in ("point","context"):
            content=s.torch.cat((content[:,:1],s.torch.zeros_like(content[:,1:])),dim=1)
        if self.kind=="point":
            z=s.torch.zeros_like(z)
        if self.kind.startswith("film"):
            guide=x_sem[:,10:]
            gamma,beta=self.modulation(guide).chunk(2,dim=1)
            z=z*(1+.1*s.torch.tanh(gamma))+.1*s.torch.tanh(beta)
        ext=self.ext_expert(h,x_ext)
        semantic=self.semantic_expert(h,s.torch.cat((x_ext,z,content,s.torch.zeros_like(rho)),dim=1))
        candidates=s.torch.cat((h,ext,semantic),dim=1)
        logits=self.gate(s.torch.cat((candidates,x_ext,content,z,s.torch.zeros_like(rho)),dim=1))
        logits=s.torch.cat((logits[:,:2],logits[:,2:]+s.F.softplus(self.reliability_slope_raw)*(rho-.5)),dim=1)
        weights=s.torch.softmax(logits,dim=1)
        return {"mean":(weights*candidates).sum(1,keepdim=True),"alphas":weights}


def train(model,arrays,device,**kwargs):
    kwargs["lr"]=.0003
    checkpoint=Path(kwargs["checkpoint_path"])
    initial=None
    if not checkpoint.exists():
        model.eval()
        losses=[]
        with s.torch.no_grad():
            for start in range(0,len(arrays["y_val"]),128):
                stop=start+128
                batch=[s.torch.as_tensor(arrays[k][start:stop],dtype=s.torch.float32,device=device)
                       for k in ("x_val","ext_val","sem_val","y_val")]
                loss=WeightedPointLoss("mae")(model(*batch[:3]),batch[3])["total"]
                losses.append((float(loss),len(batch[3])))
        initial=sum(a*b for a,b in losses)/sum(b for _,b in losses)
        optimizer=s.torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
        state=dict(checkpoint_key=kwargs["checkpoint_key"],model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),best_state=copy.deepcopy(model.state_dict()),
            best_val=initial,epoch=0,best_epoch=0,bad_epochs=0,
            numpy_random_state=s.np.random.get_state(),python_random_state=random.getstate(),
            torch_random_state=s.torch.get_rng_state(),cuda_random_state=(
                s.torch.cuda.get_rng_state_all() if s.torch.cuda.is_available() else None),
            effective_gradient_parameter_count=0,early_stopped=False,complete=False)
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        temporary=checkpoint.with_suffix(".initial.tmp.pt")
        s.torch.save(state,temporary)
        temporary.replace(checkpoint)
    result,metadata=ORIGINAL_TRAIN(model,arrays,device,**kwargs)
    metadata.update(epoch_zero_included=True,initial_validation_loss=initial)
    return result,metadata


def features(dataset,common):
    folder=s.HERE/"cells/latent_features"
    path=folder/f"{dataset}_h12_s41.npz"
    marker=path.with_suffix(".json")
    checkpoint=s.ROOT/"ns_instance_normalization_20261023/all_cal/revin_mean_std/checkpoints"/f"{dataset}_h12_rasg_s41.pt"
    text_path=s.HERE/"cells/head_features"/f"{dataset}_h12_text.npz"
    dependencies={**common["sources"],str(checkpoint):s.exp.file_sha(checkpoint),str(text_path):s.exp.file_sha(text_path)}
    if marker.exists():
        meta=json.loads(marker.read_text(encoding="utf-8"))
        assert meta["sources"]==dependencies and meta["prediction_sha256"]==s.exp.file_sha(path)
        with s.np.load(path,allow_pickle=False) as f:
            return {k:f[k].copy() for k in f.files},checkpoint,{str(path):s.exp.file_sha(path),str(marker):s.exp.file_sha(marker)}
    device=s.torch.device("cuda" if s.torch.cuda.is_available() else "cpu")
    _,cfg,shared,scalers,_,_=s.exp.prepare(dataset,12)
    arrays,inverses=s.norm.normalize(shared,scalers,"revin_mean_std")
    saved=s.torch.load(checkpoint,map_location=device,weights_only=False)
    model=s.exp.build_model("mae_loss_tuned","rasg",cfg,arrays).to(device)
    model.load_state_dict(saved["best_state"],strict=True)
    model.eval()
    captured={"latent":[],"h":[]}
    projection=None
    def latent_hook(module,inputs):
        nonlocal projection
        value=inputs[0].detach().reshape(len(inputs[0]),-1)
        if projection is None:
            rng=s.np.random.default_rng(17073)
            projection=s.torch.as_tensor(rng.choice([-1.,1.],size=(value.shape[1],32))/s.np.sqrt(32),dtype=value.dtype,device=value.device)
        captured["latent"].append((value@projection).cpu().numpy())
    def h_hook(module,inputs,output):
        captured["h"].append(output.detach().cpu().numpy())
    handles=[model.hist_branch.model.model.head.linear.register_forward_pre_hook(latent_hook),
             model.hist_branch.register_forward_hook(h_hook)]
    blocks={}
    try:
        with s.torch.no_grad():
            for split in ("train","val","cal"):
                captured={"latent":[],"h":[]}
                means=[]
                for start in range(0,len(arrays[f"y_{split}"]),256):
                    inputs=[s.torch.as_tensor(arrays[f"{key}_{split}"][start:start+256],dtype=s.torch.float32,device=device)
                            for key in ("x","ext","sem")]
                    means.append(model(*inputs)["mean"].cpu().numpy())
                blocks[split]={k:s.np.concatenate(v) for k,v in captured.items()}
                blocks[split]["S0"]=s.np.concatenate(means)
    finally:
        for handle in handles:
            handle.remove()
    old,source_refs,_=s.load_reference(dataset,12,"rasg",41,arrays)
    error=float(s.np.max(s.np.abs(inverses["cal"].inverse_transform(blocks["cal"]["S0"]).reshape(-1)-old)))
    assert error<=max(1e-3,1e-5*float(s.np.std(arrays["y_cal_raw"])))
    mean=blocks["train"]["latent"].mean(0)
    scale=s.np.maximum(blocks["train"]["latent"].std(0),1e-6)
    packed={}
    with s.np.load(text_path,allow_pickle=False) as text:
        for split in ("train","val","cal"):
            latent=(blocks[split]["latent"]-mean)/scale
            packed[f"x_{split}"]=s.np.column_stack((blocks[split]["h"],latent)).astype(s.np.float32)[:,None,:]
            for key in ("ext","y","target","semantic"):
                packed[f"{key}_{split}"]=arrays[f"{key}_{split}"]
            packed[f"y_{split}_raw"]=arrays[f"y_{split}_raw"]
            packed[f"sem_{split}"]=text[f"sem_{split}"].copy()
            s.np.testing.assert_array_equal(packed[f"sem_{split}"][:,0],arrays[f"sem_{split}"][:,0])
            s.np.testing.assert_array_equal(packed[f"sem_{split}"][:,-1],arrays[f"sem_{split}"][:,-1])
            packed[f"center_{split}"]=inverses[split].center
            packed[f"scale_{split}"]=inverses[split].scale
    folder.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp.npz")
    s.np.savez_compressed(temporary,**packed)
    temporary.replace(path)
    s.exp.atomic_json(marker,dict(sources=dependencies,original_candidate_sources=source_refs,
        prediction_sha256=s.exp.file_sha(path),mean=mean.tolist(),scale=scale.tolist(),max_cal_error=error,
        normalized_input_sha256={f"{key}_{split}":hashlib.sha256(s.np.ascontiguousarray(arrays[f"{key}_{split}"]).tobytes()).hexdigest()
                                for split in ("train","val","cal") for key in ("x","ext","sem","y")}))
    del model,saved,shared,arrays,blocks
    s.exp.base.PREPARED_CASES.clear()
    gc.collect()
    if s.torch.cuda.is_available():
        s.torch.cuda.empty_cache()
    return packed,checkpoint,{str(path):s.exp.file_sha(path),str(marker):s.exp.file_sha(marker)}


def main():
    global OLD_STATE
    s.torch.set_num_threads(1)
    s.exp.DEVICE=s.torch.device("cpu")
    s.exp.PointOnlyLoss=WeightedPointLoss
    s.exp.train_model=train
    s.SemanticModel=Candidate
    sources=[Path(__file__),s.HERE/"LATENT_PLAN.md",s.HERE/"run.py",s.HERE/"cpu_heads.py",
        Path(s.exp.__file__),Path(s.norm.__file__),Path(s.exp.base.__file__),Path(s.exp.trainer.__file__),
        s.exp.RESEARCH/"src/models/official_forecasting.py"]
    common=dict(run_sha256=s.exp.file_sha(Path(__file__)),sources={str(p):s.exp.file_sha(p) for p in sources})
    stage="latent_pilot"
    plan=dict(identity=common,stage=stage,variants=list(SPECS),datasets=["C","R","S","U"],seeds=[41],
              device="cpu",llm_calls=0,test_metrics=False,epoch_zero_included=True)
    frozen=s.HERE/f"{stage}_plan.json"
    if frozen.exists():
        assert json.loads(frozen.read_text(encoding="utf-8"))==plan
    else:
        s.exp.atomic_json(frozen,plan)
    state=dict(status="running",pid=os.getpid(),stage=stage,completed_cells=0,total_cells=48,started_unix=time.time())
    s.exp.atomic_json(s.HERE/"latent_status.json",state)
    rows=[]
    try:
        for dataset in plan["datasets"]:
            base,checkpoint,inputs=features(dataset,common)
            OLD_STATE=s.torch.load(checkpoint,map_location="cpu",weights_only=False)["best_state"]
            inverses={split:s.norm.WindowInverse(base[f"center_{split}"],base[f"scale_{split}"]) for split in ("train","val","cal")}
            identity=dict(run_sha256=common["run_sha256"],sources={**common["sources"],**inputs,str(checkpoint):s.exp.file_sha(checkpoint)})
            for variant,(kind,power) in SPECS.items():
                arrays=dict(base)
                divisor=float(s.np.mean(base["scale_train"]**power))
                rng=s.np.random.default_rng(20261106)
                for split in ("train","val","cal"):
                    content=base[f"sem_{split}"]
                    guide=content.copy()
                    if kind=="filmzero":
                        guide.fill(0)
                    elif kind=="filmshuffle":
                        guide=guide[rng.permutation(len(guide))]
                    arrays[f"sem_{split}"]=s.np.column_stack((content,guide)).astype(s.np.float32)
                    arrays[f"y_{split}"]=s.np.column_stack((base[f"y_{split}"].reshape(-1),base[f"scale_{split}"]**power/divisor)).astype(s.np.float32)
                initial=Candidate({},arrays,variant)
                prediction,_=s.exp.base.predict_point(initial,arrays,"cal",inverses["cal"],s.exp.DEVICE)
                old,_,_=s.load_reference(dataset,12,"rasg",41,arrays)
                assert float(s.np.max(s.np.abs(prediction-old)))<=max(1e-3,1e-5*float(s.np.std(arrays["y_cal_raw"])))
                state.update(current_cell=f"{dataset}_h12_s41_{variant}",updated_unix=time.time())
                s.exp.atomic_json(s.HERE/"latent_status.json",state)
                metric=s.fit(dataset,12,41,{},arrays,inverses,variant,stage,identity)
                rows.append(metric)
                s.summarize(rows,stage)
                state.update(completed_cells=len(rows),updated_unix=time.time())
                s.exp.atomic_json(s.HERE/"latent_status.json",state)
                print(json.dumps(metric),flush=True)
            del base,arrays
            gc.collect()
        state.update(status="complete",completed_unix=time.time())
        s.exp.atomic_json(s.HERE/"latent_status.json",state)
    except Exception as error:
        state.update(status="failed",error=f"{type(error).__name__}: {error}")
        s.exp.atomic_json(s.HERE/"latent_status.json",state)
        raise


if __name__=="__main__":
    with s.threadpool_limits(limits=1):
        main()
