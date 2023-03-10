from smartgd.common.data import GraphLayout
from .base_critic import BaseCritic
from .composite_critic import CompositeCritic

from typing import Optional

import torch
import torch_scatter


# TODO: scale invariance
@CompositeCritic.register_critic("tsne")
class TSNEScore(BaseCritic):

    def __init__(self, *,
                 sigma: float = 1.,
                 batch_reduce: Optional[str] = "mean"):
        super().__init__(batch_reduce=batch_reduce)
        self.sigma: float = sigma

    def compute(self, layout: GraphLayout) -> torch.Tensor:
        p = layout.edge_attr.shortest_path.div(-2 * self.sigma ** 2).exp()
        sum_src = torch_scatter.scatter(p, layout.edge_idx.full_src)[layout.edge_idx.full_src]
        sum_dst = torch_scatter.scatter(p, layout.edge_idx.full_dst)[layout.edge_idx.full_dst]
        p = (p / sum_src + p / sum_dst) / (2 * layout.n[layout.full_batch_index])
        dist = layout.full_dst_pos.sub(layout.full_src_pos).norm(dim=1)
        index = layout.full_batch_index
        q = 1 / (1 + dist.square())
        q /= torch_scatter.scatter(q, index)[index]
        edge_kl = (p.log() - q.log()).mul(p)
        return torch_scatter.scatter(edge_kl, index)
