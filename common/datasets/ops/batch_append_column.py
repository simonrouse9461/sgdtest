from typing import Optional

import torch
from torch import nn
import torch_geometric as pyg


class BatchAppendColumn(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, *,
                batch: pyg.data.Batch,
                tensor: torch.Tensor,
                name: str,
                like: Optional[str] = None) -> list[torch.Tensor]:
        data_list = batch.to_data_list()
        if not like:
            for col, _ in data_list[0]:
                target = batch[col]
                if isinstance(target, torch.Tensor) and target.shape == tensor.shape:
                    like = col
        if like:
            shape = None
        else:
            shape = tensor.shape
            shape[0] /= len(batch)
        for data in data_list:
            data[name] = torch.zeros(*shape or data[like].shape)
        batch = pyg.data.Batch.from_data_list(data_list)
        assert batch[name].shape == tensor.shape
        batch[name] = tensor.to(batch[like])
        return batch
