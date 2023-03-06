from smartgd.common.nn.ops import Reduce
from smartgd.data.graph_layout import GraphLayout
from .base_critic import BaseCritic
from .composite_critic import CompositeCritic

from typing import Optional

import torch
from torch import nn
import torch_scatter


@CompositeCritic.register_critic("ring")
class Occlusion(BaseCritic):

    def __init__(self, *, gamma: float = 1., batch_reduce: Optional[str] = "mean"):
        super().__init__(batch_reduce=batch_reduce)
        self.gamma: float = gamma

    def compute(self, layout: GraphLayout) -> torch.Tensor:
        dist = layout.full_dst_pos.sub(layout.full_src_pos).norm(dim=1)
        edge_occlusion = dist.mul(-self.gamma).exp()
        return torch_scatter.scatter(edge_occlusion, layout.full_batch_index)
