from smartgd.constants import EPS
from smartgd.common.data import GraphStruct
from smartgd.common.nn.ops import SparseSort
from ..base_layout_metric import BaseLayoutMetric
from ..composite_critic import CompositeCritic

from typing import Optional

import torch
import torch.nn.functional as F
import torch_scatter


@CompositeCritic.register_critic("xing")
class Crossings(BaseLayoutMetric):

    def __init__(self, *, eps: float = EPS, batch_reduce: Optional[str] = "mean"):
        super().__init__(batch_reduce=batch_reduce)
        self.sparse_sort = SparseSort()
        self.eps = eps

    @staticmethod
    def _cross(v, u):
        return torch.cross(
            F.pad(v, (0, 1)),
            F.pad(u, (0, 1))
        )[:, -1]

    @staticmethod
    def _dot(v, u):
        return (v * u).sum(dim=-1)

    def compute(self, layout: GraphStruct) -> torch.Tensor:
        # get pqrs
        (s1, s2), (e1, e2) = layout.edge_pair_index
        p, q = layout.pos[s1], layout.pos[s2]
        r, s = layout.pos[e1] - p, layout.pos[e2] - q

        # shrink by eps
        p += self.eps * r
        q += self.eps * s
        r *= 1 - 2 * self.eps
        s *= 1 - 2 * self.eps

        # get intersection
        qmp = q - p
        qmpxs = self._cross(qmp, s)
        qmpxr = self._cross(qmp, r)
        rxs = self._cross(r, s)
        rdr = self._dot(r, r)
        t = qmpxs / rxs
        u = qmpxr / rxs
        t0 = self._dot(qmp, r) / rdr
        t1 = t0 + self._dot(s, r) / rdr

        # calculate bool
        zero = torch.zeros_like(rxs)
        parallel = rxs.isclose(zero)
        nonparallel = parallel.logical_not()
        collinear = parallel.logical_and(qmpxr.isclose(zero))

        xing = torch.logical_or(
            collinear.logical_and(
                torch.logical_and(
                    (t0 > 0).logical_or(t1 > 0),
                    (t0 < 1).logical_or(t1 < 1),
                )
            ),
            nonparallel.logical_and(
                torch.logical_and(
                    (0 < t).logical_and(t < 1),
                    (0 < u).logical_and(u < 1),
                )
            )
        ).float()

        return torch_scatter.scatter(xing, layout.batch[s1], reduce="sum")
