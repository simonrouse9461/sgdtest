from smartgd.common.jittools import jittable
from .linear_layer import LinearLayer

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@jittable
@dataclass(kw_only=True, eq=False)
class MLP(nn.Module):

    @dataclass(kw_only=True, frozen=True)
    class Params:
        in_dim: int
        out_dim: int
        hidden_dims: list[int]

    @dataclass(kw_only=True, frozen=True)
    class Config:
        hidden_act: str = "leaky_relu"
        out_act: Optional[str] = None
        bn: Optional[str] = "batch_norm"
        dp: float = 0.0
        residual: bool = False

    params: Params
    config: Config = Config()

    def __post_init__(self):
        super().__init__()

        self.linear_seq: nn.Sequential[LinearLayer] = nn.Sequential()

        in_dims = [self.params.in_dim] + self.params.hidden_dims
        out_dims = self.params.hidden_dims + [self.params.out_dim]
        for layer_index, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            if layer_index < len(self.params.hidden_dims):
                self.linear_seq.append(LinearLayer(
                    params=LinearLayer.Params(
                        in_dim=in_dim,
                        out_dim=out_dim
                    ),
                    config=LinearLayer.Config(
                        bn=self.config.bn,
                        act=self.config.hidden_act,
                        dp=self.config.dp,
                        residual=self.config.residual
                    )
                ))
            else:
                self.linear_seq.append(LinearLayer(
                    params=LinearLayer.Params(
                        in_dim=in_dim,
                        out_dim=out_dim
                    ),
                    config=LinearLayer.Config(
                        bn=None,
                        act=self.config.out_act,
                        dp=0.0,
                        residual=False
                    )
                ))

    def forward(self, feat: torch.FloatTensor) -> torch.FloatTensor:
        return self.linear_seq(feat)
