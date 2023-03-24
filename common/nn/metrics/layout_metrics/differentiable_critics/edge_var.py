from smartgd.common.data import GraphStruct
from ..base_layout_metric import BaseLayoutMetric
from ..composite_critic import CompositeCritic

from typing import Optional

import torch
import torch_scatter


@CompositeCritic.register_critic("absedge", abs_edge_len=1.)
@CompositeCritic.register_critic("edge")
class EdgeVar(BaseLayoutMetric):

    def __init__(self, *,
                 abs_edge_len: Optional[float] = None,
                 batch_reduce: Optional[str] = "mean"):
        super().__init__(batch_reduce=batch_reduce)
        self.abs_edge_len: Optional[float] = abs_edge_len

    def compute(self, layout: GraphStruct) -> torch.Tensor:
        dist = layout.edge_dst_pos.sub(layout.edge_src_pos).norm(dim=1)
        edge_var = dist.sub(self.abs_edge_len or dist.mean()).square()
        return torch_scatter.scatter(edge_var, layout.edge_batch_index, reduce="mean")
