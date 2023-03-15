from smartgd.common.data import GraphLayout
from ..base_critic import BaseCritic
from ..composite_critic import CompositeCritic
from ..utils.shape_metric_utils import delaunay, rng, jaccard_index

from typing import Optional

import torch


@CompositeCritic.register_critic("rng")
class RNGJaccardDistance(BaseCritic):

    def __init__(self, *, batch_reduce: Optional[str] = "mean"):
        super().__init__(batch_reduce=batch_reduce)

    def compute(self, layout: GraphLayout) -> torch.Tensor:
        jaccard_dist = []
        for l in layout.split():
            raw_edges = l.edge_adj.idx.T.cpu().numpy()
            delaunay_edges = delaunay(l.pos.detach().cpu().numpy())
            shape_edges = rng(l.pos.detach().cpu().numpy(), delaunay_edges)
            jaccard_dist.append(torch.tensor(1 - jaccard_index(raw_edges, shape_edges)).to(l.pos))
        return torch.stack(jaccard_dist)
