from smartgd.common.jittools import jittable
from smartgd.common.nn import ModuleFactory
from .norm_wrappers import NormWrapper
from .skip_connection import SkipConnection

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch_geometric as pyg


@jittable
@dataclass(kw_only=True, eq=False)
class NNConvBasicLayer(nn.Module):

    @dataclass(kw_only=True, frozen=True)
    class Params:
        in_dim: int
        out_dim: int
        edge_feat_dim: int

    @dataclass(kw_only=True, frozen=True)
    class Config:
        edge_net: Optional[nn.Module] = None
        dense: bool = False
        bn: Optional[str] = "batch_norm"
        act: Optional[str] = "leaky_relu"
        dp: float = 0.0
        residual: bool = False
        aggr: str = "mean"
        root_weight: bool = True

    params: Params
    config: Config = Config()

    def __post_init__(self):
        super().__init__()

        # Define flags
        self.with_dense: bool = self.config.dense
        self.with_bn: bool = self.config.bn is not None
        self.with_act: bool = self.config.act is not None
        self.with_dp: bool = self.config.dp > 0.0
        self.residual: bool = self.config.residual

        # Define nn modules
        self.conv: nn.Module = pyg.nn.NNConv(
            self.params.in_dim, self.params.out_dim,
            nn=self.config.edge_net or ModuleFactory("linear")(
                self.params.edge_feat_dim,
                self.params.in_dim * self.params.out_dim
            ),
            aggr=self.config.aggr,
            root_weight=self.config.root_weight
        ).jittable()
        self.dense: nn.Module = nn.Linear(self.params.out_dim, self.params.out_dim)
        self.bn: nn.Module = NormWrapper(ModuleFactory(self.config.bn)(self.params.out_dim))
        self.act: nn.Module = ModuleFactory(self.config.act)()
        self.dp: nn.Module = nn.Dropout(self.config.dp)
        self.skip: SkipConnection = SkipConnection(in_dim=self.params.in_dim, out_dim=self.params.out_dim)

    def forward(self, *,
                node_feat: torch.FloatTensor,
                edge_feat: torch.FloatTensor,
                edge_index: torch.LongTensor,
                batch_index: torch.LongTensor) -> torch.FloatTensor:
        inputs = outputs = node_feat
        outputs = self.conv(x=outputs, edge_index=edge_index, edge_attr=edge_feat)
        if self.with_dense:
            outputs = self.dense(outputs)
        if self.with_bn:
            outputs = self.bn(node_feat=outputs, batch_index=batch_index)
        if self.with_act:
            outputs = self.act(outputs)
        if self.with_dp:
            outputs = self.dp(outputs)
        if self.residual:
            outputs = self.skip(block_input=inputs, block_output=outputs)
        return outputs
