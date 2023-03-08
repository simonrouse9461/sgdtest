from smartgd.constants import EPS
from smartgd.common.jittools import jittable
from ..common import NNConvLayer, NNConvBasicLayer, EdgeFeatureExpansion

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@jittable
@dataclass(kw_only=True, eq=False)
class GeneratorLayer(nn.Module):

    @dataclass(kw_only=True, frozen=True)
    class Config:
        in_dim: int
        out_dim: int
        node_feat_dim: int
        edge_feat_dim: int

    @dataclass(kw_only=True, frozen=True)
    class EdgeNetConfig:
        width: int
        depth: int
        hidden_act: str
        out_act: Optional[str]
        bn: Optional[str]
        dp: float
        residual: bool

    @dataclass(kw_only=True, frozen=True)
    class GNNConfig:
        aggr: str
        root_weight: bool
        dense: bool
        bn: Optional[str]
        act: Optional[str]
        dp: float

    config: Config
    edge_net_config: EdgeNetConfig = EdgeNetConfig(
        width=0,
        depth=0,
        hidden_act="leaky_relu",
        out_act="tanh",
        bn="batch_norm",
        dp=0.0,
        residual=False
    )
    gnn_config: GNNConfig = GNNConfig(
        aggr="mean",
        root_weight=True,
        dense=False,
        bn="pyg_batch_norm",
        act="leaky_relu",
        dp=0.0
    )
    edge_feat_expansion: EdgeFeatureExpansion.Expansions = EdgeFeatureExpansion.Expansions(),
    eps: float = EPS

    def __post_init__(self):
        super().__init__()

        self.edge_feat_provider: EdgeFeatureExpansion = EdgeFeatureExpansion(
            config=EdgeFeatureExpansion.Config(
                node_feat_dim=self.config.node_feat_dim,
                edge_attr_dim=self.config.edge_feat_dim
            ),
            expansions=self.edge_feat_expansion,
            eps=self.eps
        )

        self.gnn_layer: NNConvLayer = NNConvLayer(
            params=NNConvBasicLayer.Params(
                in_dim=self.config.in_dim,
                out_dim=self.config.out_dim,
                edge_feat_dim=self.edge_feat_provider.get_feature_channels()
            ),
            nnconv_config=NNConvLayer.NNConvConfig(
                dense=self.gnn_config.dense,
                bn=self.gnn_config.bn,
                act=self.gnn_config.act,
                dp=self.gnn_config.dp,
                residual=False,
                aggr=self.gnn_config.aggr,
                root_weight=self.gnn_config.root_weight
            ),
            edge_net_config=NNConvLayer.EdgeNetConfig(
                hidden_dims=[self.edge_net_config.width] * self.edge_net_config.depth,
                hidden_act=self.edge_net_config.hidden_act,
                out_act=self.edge_net_config.out_act,
                bn=self.edge_net_config.bn,
                dp=self.edge_net_config.dp,
                residual=self.edge_net_config.residual
            )
        )

    def forward(self, *,
                node_feat: torch.FloatTensor,
                edge_feat: torch.FloatTensor,
                edge_index: torch.LongTensor,
                batch_index: torch.LongTensor) -> torch.FloatTensor:
        return self.gnn_layer(
            node_feat=node_feat,
            edge_feat=self.edge_feat_provider(
                node_feat=node_feat,
                edge_index=edge_index,
                edge_attr=edge_feat
            ),
            edge_index=edge_index,
            batch_index=batch_index
        )
