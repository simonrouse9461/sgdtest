from smartgd.common.data.graph_struct import GraphStruct

import torch
from torch import nn


class DiscriminatorDataAdaptor(nn.Module):

    def __init__(self, discriminator: nn.Module):
        super().__init__()
        self.model = discriminator

    def forward(self, layout: GraphStruct) -> torch.Tensor:
        score = self.model(
            pos=layout.pos,
            edge_index=layout.aggr_index,
            edge_attr=torch.cat([layout.aggr_attr, layout.aggr_weight.unsqueeze(dim=1)], dim=1),
            batch_index=layout.batch
        )
        return score
