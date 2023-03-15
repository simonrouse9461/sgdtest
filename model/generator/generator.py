from smartgd.constants import EPS
from smartgd.common.functools import default_kwargs
from smartgd.common.jittools import jittable
from ..common import EdgeFeatureExpansion
from .generator_block import GeneratorBlock
from .generator_layer import GeneratorLayer

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@default_kwargs
@jittable
@dataclass(kw_only=True, eq=False)
class Generator(nn.Module):

    @dataclass(kw_only=True, frozen=True)
    class Params:
        num_blocks: int
        block_depth: int
        block_width: int
        block_output_dim: int
        edge_net_depth: int
        edge_net_width: int
        edge_attr_dim: int
        node_attr_dim: int

    @dataclass(kw_only=True, frozen=True)
    class BlockConfig:
        relative_edge_feat_mode: Optional[str] = "jump"
        residual: bool = True

    @dataclass(kw_only=True, frozen=True)
    class EdgeNetConfig:
        hidden_act: str = "leaky_relu"
        out_act: Optional[str] = "tanh"
        bn: Optional[str] = "batch_norm"
        dp: float = 0.2
        residual: bool = False

    @dataclass(kw_only=True, frozen=True)
    class GNNConfig:
        root_weight: bool = True
        bn: Optional[str] = "pyg_batch_norm"
        act: Optional[str] = "leaky_relu"
        dp: float = 0.1

    params: Params = Params(
        num_blocks=10,
        block_depth=2,
        block_width=8,
        block_output_dim=8,
        edge_net_depth=1,
        edge_net_width=16,
        edge_attr_dim=2,
        node_attr_dim=2,
    )
    block_config: BlockConfig = BlockConfig()
    edge_net_config: EdgeNetConfig = EdgeNetConfig()
    gnn_config: GNNConfig = GNNConfig()
    edge_feat_expansion: EdgeFeatureExpansion.Expansions = EdgeFeatureExpansion.Expansions(
        src_feat=False,
        dst_feat=False,
        diff_vec=False,
        unit_vec=True,
        vec_norm=True,
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

        main_block_config = GeneratorBlock.Config(
            in_dim=self.params.block_output_dim,
            hidden_dims=[self.params.block_width] * self.params.block_depth,
            out_dim=self.params.block_output_dim,
            edge_attr_dim=self.params.edge_attr_dim,
            node_attr_dim=self.params.node_attr_dim,
            dynamic_edge_feat_mode=self.block_config.relative_edge_feat_mode,
            residual=self.block_config.residual
        )
        main_edge_net_config = GeneratorLayer.EdgeNetConfig(
            width=self.params.edge_net_width,
            depth=self.params.edge_net_depth,
            hidden_act=self.edge_net_config.hidden_act,
            out_act=self.edge_net_config.out_act,
            bn=self.edge_net_config.bn,
            dp=self.edge_net_config.dp,
            residual=self.edge_net_config.residual
        )
        main_gnn_config = GeneratorLayer.GNNConfig(
            aggr="mean",
            root_weight=self.gnn_config.root_weight,
            dense=False,
            bn=self.gnn_config.bn,
            act=self.gnn_config.act,
            dp=self.gnn_config.dp
        )

        # TODO: Normalization
        #   * add [Center, Standardization] before input
        #   * add [Center, ParametricScaling] after output

        self.block_list: nn.ModuleList = nn.ModuleList()
        self.block_list.append(
            GeneratorBlock(
                config=GeneratorBlock.Config(
                    in_dim=self.params.node_attr_dim,
                    hidden_dims=[],
                    out_dim=self.params.block_output_dim,
                    edge_attr_dim=self.params.edge_attr_dim,
                    node_attr_dim=self.params.node_attr_dim,
                    dynamic_edge_feat_mode=None,
                    residual=False
                ),
                edge_net_config=main_edge_net_config,
                gnn_config=main_gnn_config,
                eps=self.eps
            )
        )
        self.block_list.extend([
            GeneratorBlock(
                config=main_block_config,
                edge_net_config=main_edge_net_config,
                gnn_config=main_gnn_config,
                edge_feat_expansion=self.edge_feat_expansion,
                eps=self.eps
            )
            for _ in range(self.params.num_blocks)
        ])
        self.block_list.append(
            GeneratorBlock(
                config=GeneratorBlock.Config(
                    in_dim=self.params.block_output_dim,
                    hidden_dims=[],
                    out_dim=2,
                    edge_attr_dim=self.params.edge_attr_dim,
                    node_attr_dim=self.params.node_attr_dim,
                    dynamic_edge_feat_mode=None,
                    residual=False,
                ),
                edge_net_config=main_edge_net_config,
                gnn_config=GeneratorLayer.GNNConfig(
                    aggr="mean",
                    root_weight=True,
                    dense=False,
                    bn=None,
                    act=None,
                    dp=0.0,
                ),
                eps=self.eps
            )
        )

    def forward(self,
                init_pos: torch.FloatTensor,
                edge_index: torch.LongTensor,
                edge_attr: torch.FloatTensor,
                batch_index: torch.LongTensor) -> torch.Tensor:
        inputs = outputs = init_pos
        for block in self.block_list:
            outputs = block(node_feat=outputs,
                            node_attr=inputs,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            batch_index=batch_index)

        # TODO: Use input as initial layout
        #   outputs += normalized_inputs
        #   outputs = parametric_scaling(outputs)

        # TODO: ScaleNet - learn best scaling

        return outputs
