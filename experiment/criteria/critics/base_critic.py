from smartgd.common.nn import Reduce
from smartgd.common.data import GraphLayout

from typing import Optional
from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseCritic(nn.Module, ABC):

    def __init__(self, *, batch_reduce: Optional[str] = "mean"):
        super().__init__()
        self.reduce: Reduce = Reduce(method=batch_reduce)

    @abstractmethod
    def compute(self, layout: GraphLayout) -> torch.Tensor:
        return NotImplemented

    def forward(self, layout: GraphLayout) -> torch.Tensor:
        return self.reduce(self.compute(layout))
