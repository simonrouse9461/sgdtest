from smartgd.data.graph_layout import GraphLayout

from .base_transformation import BaseTransformation

import torch
from torch import nn


# TODO
class ParametricScaling(BaseTransformation):

    def __init__(self):
        super().__init__()

    def forward(self, layout: GraphLayout) -> GraphLayout:
        return NotImplemented
