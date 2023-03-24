from .graph_struct import GraphStruct
from .base_graph_drawing_data import BaseGraphDrawingData
from .mixins.drawing_mixin import DrawingMixin

from typing import Any, Optional, Mapping, Union, Iterable
from functools import singledispatchmethod
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.typing import OptTensor
import torch_geometric.transforms as T
from typing_extensions import Self
import networkx as nx


class GraphDrawingData(BaseGraphDrawingData, DrawingMixin):

    # TODO: make fields in other transform stages optional as well
    def struct(self, value: Any = None, /, *,
               post_transform: Union[str, Iterable[str], None] = None) -> GraphStruct:
        if post_transform is None:
            post_transform = []
        if isinstance(post_transform, str):
            post_transform = [post_transform]
        return self._struct(value, post_transform)

    @singledispatchmethod
    def _struct(self, value: Any, post_transform: Optional[list[str]]) -> GraphStruct:
        raise NotImplementedError

    @_struct.register
    def _(self, pos: None, post_transform: Optional[list[str]]) -> GraphStruct:
        assert self.pos is not None
        data = self.post_transform(post_transform)
        batch_index = data.batch if isinstance(data, Batch) else torch.ones(data.num_nodes).to(data.device)
        return GraphStruct(
            pos=data.pos,
            n=data.n,
            m=data.m,
            x=data.x,
            batch=batch_index,
            perm_index=data.perm_index,
            perm_attr=data.perm_attr,
            perm_weight=data.perm_weight,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            edge_weight=data.edge_weight,
            aggr_index=data.aggr_index,
            aggr_attr=data.aggr_attr,
            aggr_weight=data.aggr_weight,
            apsp_attr=data.apsp_attr,
            gabriel_index=data.gabriel_index,
            rng_index=data.rng_index,
            edge_pair_index=data.edge_pair_index
        )

    @_struct.register
    def _(self, pos: np.ndarray, post_transform: Optional[list[str]]) -> GraphStruct:
        return self._struct(torch.tensor(pos), post_transform)

    @_struct.register
    def _(self, pos: torch.Tensor, post_transform: Optional[list[str]]) -> GraphStruct:
        self.pos = pos.to(self.device).float()
        return self._struct(None, post_transform)

    @_struct.register
    def _(self, store: dict, post_transform: Optional[list[str]]) -> GraphStruct:
        names = self.name
        if isinstance(names, str):
            names = [names]
        pos = np.concatenate(list(map(store.__getitem__, names)), axis=0)
        return self._struct(pos, post_transform)

    @_struct.register
    def _(self, struct: GraphStruct, post_transform: Optional[list[str]]) -> GraphStruct:
        return self._struct(struct.pos, post_transform)
