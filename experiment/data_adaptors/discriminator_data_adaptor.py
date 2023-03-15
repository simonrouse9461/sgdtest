from smartgd.common.data.graph_layout import GraphLayout

import torch
from torch import nn


class DiscriminatorDataAdaptor(nn.Module):

    def __init__(self, discriminator: nn.Module):
        super().__init__()
        self.model = discriminator

    def forward(self, layout: GraphLayout) -> torch.Tensor:
        score = self.model(
            pos=layout.pos,
            edge_index=layout.edge_mp.idx,
            edge_attr=layout.edge_mp.attr.all,
            batch_index=layout.batch
        )
        return score
