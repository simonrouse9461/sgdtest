from smartgd.common.data import GraphStruct
from ..base_layout_metric import BaseLayoutMetric
from ..composite_critic import CompositeCritic

from typing import Optional

import torch
import torch_scatter


@CompositeCritic.register_critic("ring")
class Occlusion(BaseLayoutMetric):

    def __init__(self, *, gamma: float = 1., batch_reduce: Optional[str] = "mean"):
        super().__init__(batch_reduce=batch_reduce)
        self.gamma: float = gamma

    def compute(self, layout: GraphStruct) -> torch.Tensor:
        dist = layout.perm_dst_pos.sub(layout.perm_src_pos).norm(dim=1)
        edge_occlusion = dist.mul(-self.gamma).exp()
        return torch_scatter.scatter(edge_occlusion, layout.perm_batch_index)
