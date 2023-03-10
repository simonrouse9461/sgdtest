from smartgd.common.data import GraphLayout
from .base_critic import BaseCritic
from .composite_critic import CompositeCritic

from typing import Optional

import torch
import torch_scatter


# TODO: scale invariance
@CompositeCritic.register_critic("stress")
class Stress(BaseCritic):

    def __init__(self, *, batch_reduce: Optional[str] = "mean"):
        super().__init__(batch_reduce=batch_reduce)

    def compute(self, layout: GraphLayout) -> torch.Tensor:
        dist = torch.norm(layout.full_dst_pos - layout.full_src_pos, 2, 1)
        edge_stress = dist.sub(layout.edge_attr.shortest_path).abs().div(layout.edge_attr.shortest_path).square()
        return torch_scatter.scatter(edge_stress,  layout.full_batch_index)
