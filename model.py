"""DRA's normalized two-expert decoder; reference models are fitted separately."""
import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, context_width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1+context_width, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, reference, context):
        return reference+self.net(torch.cat((reference, context), dim=1))


class DRA(nn.Module):
    """p/d: normalized PatchTST/DLinear forecasts; c: 4*d_x context; z: 32-D state."""
    def __init__(self, context_width):
        super().__init__()
        self.ext_expert = Residual(context_width)
        self.context_expert = Residual(context_width+32)
        self.gate = nn.Sequential(nn.Linear(2+context_width+32, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, p, d, c, z):
        e = self.ext_expert(d, c)
        t = self.context_expert(p, torch.cat((c, z), dim=1))
        experts = torch.cat((e, t), dim=1)
        weights = torch.softmax(self.gate(torch.cat((experts, c, z), dim=1)), dim=1)
        return dict(mean=(weights*experts).sum(1, keepdim=True), experts=experts, alphas=weights)


def window_normalize(history):
    center = history.mean(1, keepdim=True)
    scale = history.std(1, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (history-center)/scale, center, scale


def covariate_context(last_24):
    current = last_24[:, -1]
    return torch.cat((current, last_24.mean(1)-current, current-last_24[:, 0],
                      last_24.std(1, unbiased=False)), dim=1)


def raw_unit_equivalent_loss(prediction, target, origin_scale, mean_training_scale):
    return (abs(prediction-target.reshape_as(prediction))*origin_scale.reshape_as(prediction)/mean_training_scale).mean()
