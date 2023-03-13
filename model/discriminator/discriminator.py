from smartgd.constants import EPS
from smartgd.common.functools import default_kwargs
from smartgd.common.jittools import jittable
from smartgd.common.data import GraphLayout
from ..common import EdgeFeatureExpansion, NNConvLayer
from .discriminator_block import DiscriminatorBlock

from dataclasses import dataclass, field

import torch
from torch import nn
import torch_geometric as pyg


@default_kwargs
@jittable
@dataclass(kw_only=True, eq=False)
class Discriminator(nn.Module):

    @dataclass(kw_only=True, frozen=True)
    class Params:
        num_layers: int
        hidden_width: int
        edge_net_shared_depth: int
        edge_net_embedded_depth: int
        edge_net_width: int
        edge_attr_dim: int

    @dataclass(kw_only=True, frozen=True)
    class Config:
        pooling: str | list[str] = field(default_factory=lambda: ["sum", "mean", "max", "min"])

    params: Params = Params(
        num_layers=9,
        hidden_width=16,
        edge_net_shared_depth=8,
        edge_net_embedded_depth=8,
        edge_net_width=64,
        edge_attr_dim=2
    )
    config: Config = Config()
    edge_net_config: DiscriminatorBlock.EdgeNetConfig = DiscriminatorBlock.EdgeNetConfig()
    gnn_config: NNConvLayer.NNConvConfig = NNConvLayer.NNConvConfig(
        aggr="add",
        residual=True
    )
    edge_feat_expansion: EdgeFeatureExpansion.Expansions = EdgeFeatureExpansion.Expansions(
        src_feat=True,
        dst_feat=True,
        diff_vec=False,
        unit_vec=False,
        vec_norm=False,
        vec_norm_inv=False,
        vec_norm_square=False,
        vec_norm_inv_square=False,
        edge_attr_inv=False,
        edge_attr_square=False,
        edge_attr_inv_square=False
    )
    eps: float = EPS

    def __post_init__(self):
        super().__init__()

        self.block: DiscriminatorBlock = DiscriminatorBlock(
            params=DiscriminatorBlock.Params(
                in_dim=2,
                out_dim=self.params.hidden_width,
                hidden_width=self.params.hidden_width,
                hidden_depth=self.params.num_layers,
                edge_attr_dim=self.params.edge_attr_dim,
                node_attr_dim=2
            ),
            edge_net_params=DiscriminatorBlock.EdgeNetParams(
                shared_depth=self.params.edge_net_shared_depth,
                embedded_depth=self.params.edge_net_embedded_depth,
                hidden_width=self.params.edge_net_width
            ),
            edge_net_config=self.edge_net_config,
            gnn_config=self.gnn_config,
            edge_feat_expansion=self.edge_feat_expansion,
            eps=self.eps
        )

        self.readout = pyg.nn.aggr.MultiAggregation(
            aggrs=self.config.pooling,
            mode="proj",
            mode_kwargs=dict(
                in_channels=self.params.hidden_width,
                out_channels=1
            )
        )

    def forward(self, layout: GraphLayout) -> torch.Tensor:
        node_feat = self.block(
            node_feat=layout.pos,
            edge_index=layout.edge_idx.mp,
            edge_attr=layout.edge_attr.all,
            batch_index=layout.batch
        )
        outputs = self.readout(node_feat, layout.batch)
        return outputs.flatten()
