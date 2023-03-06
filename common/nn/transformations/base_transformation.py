from smartgd.data.graph_layout import GraphLayout

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseTransformation(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, layout: GraphLayout) -> GraphLayout:
        return NotImplemented
