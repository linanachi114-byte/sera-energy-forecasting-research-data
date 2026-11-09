"""Lower-capacity history branch; identical semantic-on/off S architectures."""
import importlib.util
from types import SimpleNamespace
import run as s

PATH=s.exp.RESEARCH/"official_baselines/vendor/PatchTST/PatchTST_supervised/models/DLinear.py"
spec=importlib.util.spec_from_file_location("local_s_dlinear",PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
POINT_STATE=None
STATS=None


def history(cfg):
    return module.Model(SimpleNamespace(seq_len=cfg["hist_len"],pred_len=1,enc_in=1,individual=False))


class Point(s.nn.Module):
    def __init__(self,cfg,arrays,variant):
        super().__init__();self.model=history(cfg)

    def forward(self,x_hist,x_ext,x_sem):
        point=self.model(x_hist)[:,-1,:1]
        return {"mean":point,"alphas":s.torch.ones_like(point)}


class Candidate(s.nn.Module):
    def __init__(self,cfg,arrays,variant):
        super().__init__();self.variant=variant
        self.history=history(cfg);self.history.load_state_dict(POINT_STATE,strict=True)
        d=arrays["ext_train"].shape[1]
        self.ext_expert=s.exp.SnapshotCandidateExpert(d)
        self.semantic_expert=s.exp.SnapshotCandidateExpert(d+32+9+1)
        self.gate=s.nn.Sequential(s.nn.Linear(3+d+9+32+1,64),s.nn.GELU(),s.nn.Linear(64,3))
        self.reliability_slope_raw=s.nn.Parameter(s.torch.zeros(()))
        for expert in (self.ext_expert,self.semantic_expert):
            s.nn.init.zeros_(expert.net[-1].weight);s.nn.init.zeros_(expert.net[-1].bias)
        projection=s.np.random.default_rng(17073).choice([-1.,1.],size=(cfg["hist_len"],32))/s.np.sqrt(32)
        self.register_buffer("projection",s.torch.as_tensor(projection,dtype=s.torch.float32))
        self.register_buffer("state_mean",s.torch.as_tensor(STATS["mean"],dtype=s.torch.float32))
        self.register_buffer("state_scale",s.torch.as_tensor(STATS["scale"],dtype=s.torch.float32))

    def forward(self,x_hist,x_ext,x_sem):
        h=self.history(x_hist)[:,-1,:1]
        z=(x_hist[:,:,0]@self.projection-self.state_mean)/self.state_scale
        content,rho=x_sem[:,:9],x_sem[:,9:10]
        if self.variant=="numeric_control":
            content=s.torch.zeros_like(content);rho=s.torch.full_like(rho,.5)
        ext=self.ext_expert(h,x_ext)
        semantic=self.semantic_expert(h,s.torch.cat((x_ext,z,content,s.torch.zeros_like(rho)),dim=1))
        candidates=s.torch.cat((h,ext,semantic),dim=1)
        logits=self.gate(s.torch.cat((candidates,x_ext,content,z,s.torch.zeros_like(rho)),dim=1))
        logits=s.torch.cat((logits[:,:2],logits[:,2:]+s.F.softplus(self.reliability_slope_raw)*(rho-.5)),dim=1)
        weights=s.torch.softmax(logits,dim=1)
        return {"mean":(weights*candidates).sum(1,keepdim=True),"alphas":weights}
