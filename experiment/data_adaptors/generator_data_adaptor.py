from smartgd.common.data.graph_layout import GraphLayout

import torch
from torch import nn


class GeneratorDataAdaptor(nn.Module):

    def __init__(self, generator: nn.Module):
        super().__init__()
        self.model = generator

    def forward(self, layout: GraphLayout) -> GraphLayout:
        pos = self.model(
            init_pos=layout.pos,
            edge_index=layout.edge_mp.idx,
            edge_attr=layout.edge_mp.attr.all,
            batch_index=layout.batch
        )
        return layout(pos)
