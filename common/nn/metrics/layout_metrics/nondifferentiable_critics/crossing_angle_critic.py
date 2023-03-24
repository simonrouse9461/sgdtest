from smartgd.common.data import GraphStruct
from smartgd.common.nn.ops import SparseSort
from ..base_layout_metric import BaseLayoutMetric
from ..composite_critic import CompositeCritic

from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import torch_scatter


@CompositeCritic.register_critic("xangle")
class CrossingAngle(BaseLayoutMetric):

    def __init__(self, *, batch_reduce: Optional[str] = "mean"):
        super().__init__(batch_reduce=batch_reduce)
        self.sparse_sort = SparseSort()

    def compute(self, layout: GraphStruct) -> torch.Tensor:
        raise NotImplementedError
