from smartgd.common.decorators import jittable

import shelve
from dataclasses import dataclass
from typing import Mapping, Optional
from typing_extensions import Self

import torch
from torch import jit
import torch_geometric as pyg


@jit.script
@jittable
@dataclass(kw_only=True, eq=False, repr=False)
class GraphLayout:

    @jittable
    @dataclass(kw_only=True, eq=False, repr=False)
    class GraphAttribute:
        n: torch.LongTensor
        m: torch.LongTensor

    @jittable
    @dataclass(kw_only=True, eq=False, repr=False)
    class NodeAttribute:
        laplacian_pe: torch.FloatTensor
        random_walk_pe: torch.FloatTensor

        @property
        def all(self) -> torch.Tensor:
            return torch.stack([self.laplacian_pe, self.random_walk_pe], dim=-1)

    @jittable
    @dataclass(kw_only=True, eq=False, repr=False)
    class EdgeAttribute:
        shortest_path: torch.FloatTensor

        @property
        def k(self) -> torch.FloatTensor:
            return 1 / self.shortest_path ** 2

        @property
        def all(self) -> torch.Tensor:
            return torch.stack([self.shortest_path, self.k], dim=-1)

    @jittable
    @dataclass(kw_only=True, eq=False, repr=False)
    class EdgeIndex:
        mp: torch.LongTensor
        full: torch.LongTensor
        adj: torch.LongTensor

        @property
        def mp_src(self) -> torch.Tensor:
            return self.mp[0]

        @property
        def mp_dst(self) -> torch.Tensor:
            return self.mp[1]

        @property
        def full_src(self) -> torch.Tensor:
            return self.full[0]

        @property
        def full_dst(self) -> torch.Tensor:
            return self.full[1]

        @property
        def adj_src(self) -> torch.Tensor:
            return self.full[0]

        @property
        def adj_dst(self) -> torch.Tensor:
            return self.full[1]

    pos: torch.FloatTensor
    batch: torch.LongTensor
    graph_attr: GraphAttribute
    # node_attr: NodeAttribute  # TODO: add node attr
    edge_attr: EdgeAttribute
    edge_idx: EdgeIndex

    @property
    def full_src_pos(self) -> torch.FloatTensor:
        return self.pos[self.edge_idx.full_src]

    @property
    def full_dst_pos(self) -> torch.FloatTensor:
        return self.pos[self.edge_idx.full_dst]

    @property
    def adj_src_pos(self) -> torch.FloatTensor:
        return self.pos[self.edge_idx.full_src]

    @property
    def adj_dst_pos(self) -> torch.FloatTensor:
        return self.pos[self.edge_idx.full_dst]

    @property
    def full_batch_index(self) -> torch.LongTensor:
        return self.batch[self.edge_idx.full_src]

    @property
    def adj_batch_index(self) -> torch.LongTensor:
        return self.batch[self.edge_idx.adj_src]

    def __call__(self, pos: torch.FloatTensor):  # TODO: figure out how to use `-> Self`
        return GraphLayout(
            pos=pos,
            batch=self.batch,
            graph_attr=self.graph_attr,
            edge_attr=self.edge_attr,
            edge_idx=self.edge_idx
        )

    @classmethod
    @jit.unused
    def _from_pos(cls,
                  pos: torch.Tensor,
                  data: pyg.data.Data | pyg.data.Batch):  # TODO: figure out how to use `-> Self`
        if isinstance(data, pyg.data.Batch):
            batch_index = data.batch
        else:
            batch_index = torch.zeros(len(pos), device=pos.device).long()
        return cls(
            pos=pos,
            batch=batch_index,
            graph_attr=cls.GraphAttribute(
                n=data.n,
                m=data.m
            ),
            edge_attr=cls.EdgeAttribute(
                shortest_path=data.d_attr
            ),
            edge_idx=cls.EdgeIndex(
                mp=data.edge_index,
                full=data.full_index,
                adj=data.adj_index
            )
        )

    @classmethod
    @jit.unused
    def _from_kvstore(cls,
                      kvstore: Mapping,
                      data: pyg.data.Data | pyg.data.Batch):  # TODO: figure out how to use `-> Self`
        if isinstance(data, pyg.data.Batch):
            names = data.name
        else:
            names = [data.name]
        pos_list = map(lambda x: torch.tensor(x).to(data.pos), map(kvstore.__getitem__, names))
        pos = torch.cat(list(pos_list))
        return cls._from_pos(pos=pos, data=data)

    @classmethod
    @jit.unused
    def from_data(cls,
                  data: pyg.data.Data | pyg.data.Batch,
                  pos: Optional[torch.Tensor] = None,
                  kvstore: Optional[Mapping] = None):  # TODO: figure out how to use `-> Self`
        if pos:
            return cls._from_pos(pos=pos, data=data)
        if kvstore:
            return cls._from_kvstore(kvstore=kvstore, data=data)
        return cls._from_pos(pos=data.pos, data=data)
