from smartgd.common.data.graph_struct import GraphStruct

import torch
from torch import nn


class GeneratorDataAdaptor(nn.Module):

    def __init__(self, generator: nn.Module):
        super().__init__()
        self.model = generator

    def forward(self, layout: GraphStruct) -> GraphStruct:
        pos = self.model(
            init_pos=layout.pos,
            edge_index=layout.aggr_index,
            edge_attr=torch.cat([layout.aggr_attr, layout.aggr_weight.unsqueeze(dim=1)], dim=1),
            batch_index=layout.batch
        )
        return layout(pos)
