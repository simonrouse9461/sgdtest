from smartgd.common.jittools import jittable
from .mlp import MLP
from .nnconv_basic_layer import NNConvBasicLayer

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn


@jittable
@dataclass(kw_only=True, eq=False)
class NNConvLayer(nn.Module):

    @dataclass(kw_only=True, frozen=True)
    class NNConvConfig:
        dense: bool = False
        bn: Optional[str] = "pyg_batch_norm"
        act: Optional[str] = "leaky_relu"
        dp: float = 0.0
        residual: bool = False
        aggr: str = "mean"
        root_weight: bool = True

    @dataclass(kw_only=True, frozen=True)
    class EdgeNetConfig:
        hidden_dims: list[int] = field(default_factory=list)
        hidden_act: str = "leaky_relu"
        out_act: Optional[str] = None
        bn: Optional[str] = "batch_norm"
        dp: float = 0.0
        residual: bool = True

    params: NNConvBasicLayer.Params
    nnconv_config: NNConvConfig = NNConvConfig()
    edge_net_config: EdgeNetConfig = EdgeNetConfig()

    def __post_init__(self):
        super().__init__()

        self.nnconv_layer: NNConvBasicLayer = NNConvBasicLayer(
            params=self.params,
            config=NNConvBasicLayer.Config(
                edge_net=MLP(
                    params=MLP.Params(
                        in_dim=self.params.edge_feat_dim,
                        out_dim=self.params.in_dim * self.params.out_dim,
                        hidden_dims=self.edge_net_config.hidden_dims,
                    ),
                    config=MLP.Config(
                        hidden_act=self.edge_net_config.hidden_act,
                        out_act=self.edge_net_config.out_act,
                        bn=self.edge_net_config.bn,
                        dp=self.edge_net_config.dp,
                        residual=self.edge_net_config.residual
                    )
                ),
                dense=self.nnconv_config.dense,
                bn=self.nnconv_config.bn,
                act=self.nnconv_config.act,
                dp=self.nnconv_config.dp,
                residual=self.nnconv_config.residual,
                aggr=self.nnconv_config.aggr,
                root_weight=self.nnconv_config.root_weight
            )
        )

    def forward(self, *,
                node_feat: torch.FloatTensor,
                edge_feat: torch.FloatTensor,
                edge_index: torch.LongTensor,
                batch_index: torch.LongTensor) -> torch.FloatTensor:
        return self.nnconv_layer(
            node_feat=node_feat,
            edge_feat=edge_feat,
            edge_index=edge_index,
            batch_index=batch_index
        )
