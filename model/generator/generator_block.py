from .generator_layer import GeneratorLayer
from .generator_feature_router import GeneratorFeatureRouter
from smartgd.model.common import SkipConnection, EdgeFeatureExpansion
from smartgd.common.decorators import jittable
from smartgd.global_constants import EPS

from dataclasses import dataclass
from collections import defaultdict
from typing import Optional
from functools import cached_property

import torch
from torch import nn


@jittable
@dataclass(kw_only=True, eq=False)
class GeneratorBlock(nn.Module):

    @dataclass(kw_only=True, frozen=True)
    class Config:
        in_dim: int
        out_dim: int
        edge_attr_dim: int
        node_attr_dim: int
        residual: bool
        hidden_dims: list[int]  # TODO: use width and depth instead
        dynamic_edge_feat_mode: Optional[str]
        """dynamic_edge_feat_mode: {
            "simple": block input to first layer, 
            "jump": block input to each layer,
            "chain": every layer output to next layer,
            "raw_simple": raw attribute to first layer,
            "raw_jump": raw attribute to each layer,
            None: no relative edge features
        }
        """

    config: Config
    edge_net_config: GeneratorLayer.EdgeNetConfig
    gnn_config: GeneratorLayer.GNNConfig
    edge_feat_expansion: EdgeFeatureExpansion.Expansions = EdgeFeatureExpansion.Expansions()
    eps: float = EPS

    def __post_init__(self):
        super().__init__()

        self.residual = self.config.residual

        self.skip: SkipConnection = SkipConnection(
            in_dim=self.config.in_dim,
            out_dim=self.config.out_dim
        )

        self.first_layer_feature_router: GeneratorFeatureRouter = self._make_feature_router(
            input_source=self.first_layer_input_source
        )

        self.rest_layers_feature_router: GeneratorFeatureRouter = self._make_feature_router(
            input_source=self.rest_layers_input_source
        )

        dims = [self.config.in_dim] + self.config.hidden_dims + [self.config.out_dim]

        self.first_layer: GeneratorLayer = self._make_layer(
            in_dim=dims[0],
            out_dim=dims[1],
            router=self.first_layer_feature_router
        )

        self.rest_layers: nn.ModuleList[GeneratorLayer] = nn.ModuleList()
        for layer_index, (in_dim, out_dim) in enumerate(zip(dims[1:-1], dims[2:])):
            self.rest_layers.append(self._make_layer(
                in_dim=in_dim,
                out_dim=out_dim,
                router=self.rest_layers_feature_router
            ))

    @cached_property
    def first_layer_input_source(self) -> Optional[str]:
        return defaultdict(
            lambda: "null",
            simple="block",
            jump="block",
            chain="block",
            raw_simple="raw",
            raw_jump="raw",
        )[self.config.dynamic_edge_feat_mode]

    @cached_property
    def rest_layers_input_source(self) -> Optional[str]:
        return defaultdict(
            lambda: "null",
            simple="null",
            jump="block",
            chain=None,
            raw_simple="null",
            raw_jump="raw",
        )[self.config.dynamic_edge_feat_mode]

    def _make_feature_router(self, *, input_source: Optional[str]) -> GeneratorFeatureRouter:
        return GeneratorFeatureRouter(
            config=GeneratorFeatureRouter.Config(
                input_source=input_source,
                block_input_dim=self.config.in_dim,
                raw_input_dim=self.config.node_attr_dim,
                edge_attr_dim=self.config.edge_attr_dim
            ),
            edge_feat_expansion=self.edge_feat_expansion,
            eps=self.eps
        )

    def _make_layer(self, *,
                    in_dim: int,
                    out_dim: int,
                    router: GeneratorFeatureRouter) -> GeneratorLayer:
        return GeneratorLayer(
            config=GeneratorLayer.Config(
                in_dim=in_dim,
                out_dim=out_dim,
                node_feat_dim=in_dim,
                edge_feat_dim=(router.get_output_channels()
                               if router.input_source else self.config.edge_attr_dim)
            ),
            edge_net_config=self.edge_net_config,
            gnn_config=self.gnn_config,
            edge_feat_expansion=(EdgeFeatureExpansion.Expansions() if router.input_source
                                 else self.edge_feat_expansion),
            eps=self.eps
        )

    def forward(self, *,
                node_feat: torch.FloatTensor,
                node_attr: torch.FloatTensor,
                edge_attr: torch.FloatTensor,
                edge_index: torch.LongTensor,
                batch_index: torch.LongTensor) -> torch.FloatTensor:

        outputs = self.first_layer(
            node_feat=node_feat,
            edge_feat=self.first_layer_feature_router(
                block_input=node_feat,
                raw_input=node_attr,
                edge_attr=edge_attr,
                edge_index=edge_index
            ),
            edge_index=edge_index,
            batch_index=batch_index
        )

        rest_layers_edge_feat = self.rest_layers_feature_router(
            block_input=node_feat,
            raw_input=node_attr,
            edge_attr=edge_attr,
            edge_index=edge_index
        )

        for layer in self.rest_layers:
            outputs = layer(
                node_feat=outputs,
                edge_feat=rest_layers_edge_feat,
                edge_index=edge_index,
                batch_index=batch_index
            )

        if self.residual:
            outputs = self.skip(
                block_input=node_feat,
                block_output=outputs
            )
        return outputs
