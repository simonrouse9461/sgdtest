from smartgd.common.nn.ops import Reduce
from smartgd.data.graph_layout import GraphLayout
from .base_critic import BaseCritic
from .composite_critic import CompositeCritic

from typing import Optional

import torch
from torch import nn
import torch_scatter


@CompositeCritic.register_critic("edge", abs_edge_len=1.)
@CompositeCritic.register_critic("edgevar")
class EdgeVar(BaseCritic):

    def __init__(self, *,
                 abs_edge_len: Optional[float] = None,
                 batch_reduce: Optional[str] = "mean"):
        super().__init__(batch_reduce=batch_reduce)
        self.abs_edge_len: Optional[float] = abs_edge_len

    def compute(self, layout: GraphLayout) -> torch.Tensor:
        dist = layout.adj_dst_pos.sub(layout.adj_src_pos).norm(dim=1)
        edge_var = dist.sub(self.abs_edge_len or dist.mean()).square()
        return torch_scatter.scatter(edge_var, layout.adj_batch_index, reduce="mean")
