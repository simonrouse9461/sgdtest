from smartgd.common.data import GraphLayout
from ..base_critic import BaseCritic
from ..composite_critic import CompositeCritic
from ..utils.shape_metric_utils import simple_shape_score

from typing import Optional

import torch


@CompositeCritic.register_critic("sss")
class SimpleShapeScore(BaseCritic):

    def __init__(self, *, batch_reduce: Optional[str] = "mean"):
        super().__init__(batch_reduce=batch_reduce)

    def compute(self, layout: GraphLayout) -> torch.Tensor:
        shape_score = []
        for l in layout.split():
            raw_edges = l.edge_adj.idx.T.cpu().numpy()
            sss = simple_shape_score(l.pos.detach().cpu().numpy(), raw_edges, k=self.k)
            shape_score.append(torch.tensor(sss).to(l.pos))
        return torch.stack(shape_score)
