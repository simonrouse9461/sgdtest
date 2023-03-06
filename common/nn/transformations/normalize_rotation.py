from smartgd.data.graph_layout import GraphLayout

from .base_transformation import BaseTransformation

import numpy as np
import torch
from torch import nn
import torch_scatter


class NormalizeRotation(BaseTransformation):

    def __init__(self, base_angle: float = 0):
        super().__init__()
        sin = np.sin(base_angle)
        cos = np.cos(base_angle)
        self.base_rotation: torch.FloatTensor = torch.tensor(
            [[-sin, +cos],
             [+cos, +sin]]
        ).float()

    def forward(self, layout: GraphLayout) -> GraphLayout:
        outer = torch.einsum('ni,nj->nij', layout.pos, layout.pos)
        cov = torch_scatter.scatter(outer, layout.batch, dim=0, reduce='mean')
        components = torch.linalg.eigh(cov).eigenvectors
        return layout(torch.einsum('ij,njk,nk->ni',
                                   self.base_rotation.to(layout.pos.device),
                                   components[layout.batch],
                                   layout.pos).float())
