from smartgd.common.jittools import jittable, jittable_recover

import re
from dataclasses import dataclass
from types import SimpleNamespace as namespace
from typing import Mapping, Optional, Union, Iterable
from typing_extensions import Self

import torch
from torch import jit
import torch_geometric as pyg


@jittable_recover
@jit.script
@jittable
@dataclass(eq=False, repr=False)
class GraphLayout:

    @jittable
    @dataclass(eq=False, repr=False)
    class GraphAttr:
        n: torch.LongTensor
        dataset_id: torch.LongTensor
        graph_id: torch.LongTensor

    @jittable
    @dataclass(eq=False, repr=False)
    class NodeAttr:
        laplacian_pe: torch.FloatTensor
        random_walk_pe: torch.FloatTensor

        @property
        def all(self) -> torch.Tensor:
            return torch.stack([self.laplacian_pe, self.random_walk_pe], dim=-1)

    @jittable
    @dataclass(eq=False, repr=False)
    class Edge:

        @jittable
        @dataclass(eq=False, repr=False)
        class Attr:
            shortest_path: torch.FloatTensor

            @property
            def k(self) -> torch.FloatTensor:
                return 1 / self.shortest_path ** 2

            @property
            def all(self) -> torch.Tensor:
                return torch.stack([self.shortest_path, self.k], dim=-1)

        idx: torch.LongTensor
        m: torch.LongTensor
        attr: Attr

        @property
        def src(self) -> torch.Tensor:
            return self.idx[0]

        @property
        def dst(self) -> torch.Tensor:
            return self.idx[1]

    pos: torch.FloatTensor
    batch: torch.LongTensor
    graph_attr: GraphAttr
    # node_attr: NodeAttr  # TODO: add node attr
    edge_mp: Edge
    edge_full: Edge
    edge_adj: Edge

    @property
    def full_src_pos(self) -> torch.FloatTensor:
        return self.pos[self.edge_full.src]

    @property
    def full_dst_pos(self) -> torch.FloatTensor:
        return self.pos[self.edge_full.dst]

    @property
    def adj_src_pos(self) -> torch.FloatTensor:
        return self.pos[self.edge_adj.src]

    @property
    def adj_dst_pos(self) -> torch.FloatTensor:
        return self.pos[self.edge_adj.dst]

    @property
    def full_batch_index(self) -> torch.LongTensor:
        return self.batch[self.edge_full.src]

    @property
    def adj_batch_index(self) -> torch.LongTensor:
        return self.batch[self.edge_adj.src]

    def __call__(self, pos: torch.FloatTensor):  # TODO: figure out how to use `-> Self`
        return GraphLayout(
            pos=pos,
            batch=self.batch,
            graph_attr=self.graph_attr,
            edge_mp=self.edge_mp,
            edge_full=self.edge_full,
            edge_adj=self.edge_adj
        )

    # TODO: define in superclass
    @jit.unused
    def __repr__(self) -> str:
        return type(self).__name__ + re.sub("namespace", "", repr(namespace(
            pos=list(self.pos.shape),
            batch=list(self.batch.shape),
            graph_attr=namespace(
                n=list(self.graph_attr.n.shape),
                dataset_id=list(self.graph_attr.dataset_id.shape),
                graph_id=list(self.graph_attr.graph_id.shape)
            ),
            # TODO: use loop
            edge_mp=namespace(
                idx=list(self.edge_mp.idx.shape),
                m=list(self.edge_mp.m.shape),
                attr=namespace(
                    shortest_path=list(self.edge_mp.attr.shortest_path.shape)
                )
            ),
            edge_full=namespace(
                idx=list(self.edge_full.idx.shape),
                m=list(self.edge_full.m.shape),
                attr=namespace(
                    shortest_path=list(self.edge_full.attr.shortest_path.shape)
                )
            ),
            edge_adj=namespace(
                idx=list(self.edge_adj.idx.shape),
                m=list(self.edge_adj.m.shape),
                attr=namespace(
                    shortest_path=list(self.edge_adj.attr.shortest_path.shape)
                )
            )
        )))

    @jit.unused
    def split(self):  # TODO: figure out how to use `-> Iterable[Self]`
        n_range_right = torch.cumsum(self.graph_attr.n, 0)
        n_range_left = n_range_right - self.graph_attr.n
        m_mp_range_right = torch.cumsum(self.edge_mp.m*2, 0)
        m_mp_range_left = m_mp_range_right - self.edge_mp.m*2
        m_full_range_right = torch.cumsum(self.edge_full.m*2, 0)
        m_full_range_left = m_full_range_right - self.edge_full.m*2
        m_adj_range_right = torch.cumsum(self.edge_adj.m*2, 0)
        m_adj_range_left = m_adj_range_right - self.edge_adj.m*2
        idx_range = torch.stack([
            n_range_left, n_range_right,
            m_mp_range_left, m_mp_range_right,
            m_full_range_left, m_full_range_right,
            m_adj_range_left, m_adj_range_right
        ]).T
        for i, (nl, nr, mml, mmr, mfl, mfr, mal, mar) in enumerate(idx_range):
            yield GraphLayout(
                pos=self.pos[nl:nr, :],
                batch=self.batch[nl:nr] - i,
                graph_attr=self.GraphAttr(
                    n=self.graph_attr.n[i:i+1],
                    dataset_id=self.graph_attr.dataset_id[i:i+1],
                    graph_id=self.graph_attr.graph_id[i:i+1]
                ),
                # TODO: use loop
                edge_mp=self.Edge(
                    idx=self.edge_mp.idx[:, mml:mmr] - nl,
                    m=self.edge_mp.m[i:i+1],
                    attr=self.Edge.Attr(
                        shortest_path=self.edge_mp.attr.shortest_path[mml:mmr]
                    )
                ),
                edge_full=self.Edge(
                    idx=self.edge_full.idx[:, mfl:mfr] - nl,
                    m=self.edge_full.m[i:i+1],
                    attr=self.Edge.Attr(
                        shortest_path=self.edge_full.attr.shortest_path[mfl:mfr]
                    )
                ),
                edge_adj=self.Edge(
                    idx=self.edge_adj.idx[:, mal:mar] - nl,
                    m=self.edge_adj.m[i:i+1],
                    attr=self.Edge.Attr(
                        shortest_path=self.edge_adj.attr.shortest_path[mal:mar]
                    )
                )
            )

    @classmethod
    @jit.unused
    def cat(cls, layouts: list):  # TODO: figure out how to use `-> Self`
        raise NotImplementedError

    @classmethod
    @jit.unused
    def _from_pos(cls,
                  pos: torch.Tensor,
                  data: Union[pyg.data.Data, pyg.data.Batch]):  # TODO: figure out how to use `-> Self`
        if isinstance(data, pyg.data.Batch):
            batch_index = data.batch
        else:
            batch_index = torch.zeros(len(pos), device=pos.device).long()
        return cls(
            pos=pos,
            batch=batch_index,
            graph_attr=cls.GraphAttr(
                n=data.n,
                dataset_id=data.dataset_id,
                graph_id=data.graph_id
            ),
            edge_mp=cls.Edge(
                idx=data.edge_index,
                m=data.edge_m,
                attr=cls.Edge.Attr(
                    shortest_path=data.edge_d_attr
                )
            ),
            edge_full=cls.Edge(
                idx=data.full_index,
                m=data.full_m,
                attr=cls.Edge.Attr(
                    shortest_path=data.full_d_attr
                )
            ),
            edge_adj=cls.Edge(
                idx=data.adj_index,
                m=data.adj_m,
                attr=cls.Edge.Attr(
                    shortest_path=data.adj_d_attr
                )
            )
        )

    @classmethod
    @jit.unused
    def _from_kvstore(cls,
                      kvstore: Mapping,
                      data: Union[pyg.data.Data, pyg.data.Batch]):  # TODO: figure out how to use `-> Self`
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
                  data: Union[pyg.data.Data, pyg.data.Batch],
                  pos: Optional[torch.Tensor] = None,
                  kvstore: Optional[Mapping] = None):  # TODO: figure out how to use `-> Self`
        if pos:
            return cls._from_pos(pos=pos, data=data)
        if kvstore:
            return cls._from_kvstore(kvstore=kvstore, data=data)
        return cls._from_pos(pos=data.pos, data=data)
