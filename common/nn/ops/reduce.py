from typing import Optional

import torch
from torch import nn


class Reduce(nn.Module):

    def __init__(self, method: Optional[str] = None):
        super().__init__()
        self.reduce: str = method or "none"

    def forward(self, x: torch.Tensor, /, *, dim: int = 0) -> torch.Tensor:
        if self.reduce == "mean":
            return torch.mean(x, dim=dim)
        if self.reduce == "sum":
            return torch.sum(x, dim=dim)
        if self.reduce == "min":
            return torch.min(x, dim=dim)
        if self.reduce == "max":
            return torch.max(x, dim=dim)
        if self.reduce == "none":
            return x
