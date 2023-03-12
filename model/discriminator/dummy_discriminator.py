from smartgd.common.nn.criteria.critics import Stress
from smartgd.common.data import GraphLayout
from smartgd.common.jittools import jittable

from dataclasses import dataclass

import torch
from torch import nn


@jittable
@dataclass(kw_only=True, eq=False)
class DummyDiscriminator(nn.Module):
    def __post_init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.stress = Stress(batch_reduce=None)

    def forward(self, layout: GraphLayout) -> torch.Tensor:
        outputs = self.stress(layout)
        outputs = torch.log(outputs)
        return self.dummy * 0 - outputs
