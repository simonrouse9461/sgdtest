from .transforms import (
    NormalizeGraph,
    AddAdjacencyInfo,
    CreateEdgePairs,
    ComputeShortestPath,
    Delaunay,
    GenerateRandomLayout,
    BatchAppendColumn,
    PopulateGraphAttrs,
    SampleAggregationEdges,
    GabrielGraph,
    RandomNeighborhoodGraph,
)

from typing import Any, Optional, Union, Iterable
from itertools import permutations
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.typing import OptTensor
import torch_geometric.transforms as T
from typing_extensions import Self
import networkx as nx


class BaseGraphDrawingData(Data):

    @dataclass
    class Field:
        stage: str
        transform: T.BaseTransform

    # Inputs
    G:                         nx.Graph

    # Initialization
    perm_index:                Tensor

    # After pre_transform (Storage Footprint -- Saved to disk)
    edge_metaindex:            Tensor = Field(stage="pre_transform",
                                              transform=AddAdjacencyInfo(attr_name="edge_metaindex"))
    apsp_attr:                 Tensor = Field(stage="pre_transform",
                                              transform=(compute_shortest_path := ComputeShortestPath(
                                                  cutoff=None,
                                                  attr_name="apsp_attr",
                                                  weight_name="perm_weight"
                                              )))
    perm_weight:               Tensor = Field(stage="pre_transform",
                                              transform=compute_shortest_path)
    laplacian_eigenvector_pe:  Tensor = Field(stage="pre_transform",
                                              transform=T.AddLaplacianEigenvectorPE(
                                                  k=3,
                                                  is_undirected=True,
                                                  attr_name="laplacian_eigenvector_pe"
                                              ))

    # After transform (Memory Footprint -- Generated everytime when being loaded to memory)
    name:                      str = Field(stage="transform", transform=PopulateGraphAttrs())
    dataset:                   str = Field(stage="transform", transform=PopulateGraphAttrs())
    n:                         Tensor = Field(stage="transform", transform=PopulateGraphAttrs())
    m:                         Tensor = Field(stage="transform", transform=PopulateGraphAttrs())
    aggr_metaindex:            Tensor = Field(stage="transform",
                                              transform=SampleAggregationEdges(attr_name="aggr_metaindex"))

    # After post_transform (Memory/CPU Footprint -- Generated everytime when needed)
    # TODO: raise exception if field is None, then catch the exception and perform corresponding transform
    pos:                       OptTensor = Field(stage="post_transform", transform=GenerateRandomLayout())
    face:                      OptTensor = Field(stage="post_transform", transform=Delaunay())
    edge_pair_metaindex:       OptTensor = Field(stage="post_transform",
                                                 transform=CreateEdgePairs(
                                                     edge_metaindex_name="edge_metaindex",
                                                     attr_name="edge_pair_metaindex"
                                                 ))
    gabriel_index:             OptTensor = Field(stage="post_transform",
                                                 transform=GabrielGraph(attr_name="gabriel_index"))
    rng_index:                 OptTensor = Field(stage="post_transform",
                                                 transform=RandomNeighborhoodGraph(attr_name="rng_index"))

    # Dynamic (CPU/GPU Footprint -- Generated on the fly when being accessed)
    # ------------------------------------ pre_transform
    x:                         Tensor
    perm_attr:                 Tensor
    edge_index:                Tensor
    edge_attr:                 Tensor
    edge_weight:               Tensor
    # ---------------------------------------- transform
    aggr_index:                Tensor
    aggr_attr:                 Tensor
    aggr_weight:               Tensor
    # ----------------------------------- post_transform
    edge_pair_index:           OptTensor

    @property
    def x(self) -> Tensor:
        assert self.laplacian_eigenvector_pe is not None
        return torch.cat([
            self.laplacian_eigenvector_pe,
        ], dim=1).float()

    @property
    def perm_attr(self) -> Tensor:
        assert self.apsp_attr is not None
        return torch.cat([self.apsp_attr.unsqueeze(1)], dim=1).float()

    @property
    def edge_index(self) -> Tensor:
        return NotImplemented  # override as NotImplemented to force dynamic generation

    @property
    def edge_attr(self) -> Tensor:
        return NotImplemented

    @property
    def edge_weight(self) -> Tensor:
        return NotImplemented

    @property
    def device(self) -> torch.device:
        return self.perm_index.device

    @classmethod
    def field_annotations(cls) -> dict[str, type]:
        return BaseGraphDrawingData.__annotations__

    @classmethod
    def fields(cls, stage: Optional[str] = None) -> dict[str, Field]:
        def generate_fields():
            for field_name in cls.field_annotations():
                if not hasattr(cls, field_name):
                    continue
                field_info = getattr(cls, field_name)
                if not isinstance(field_info, cls.Field):
                    continue
                if stage in [None, field_info.stage]:
                    yield field_name, field_info
        return dict(generate_fields())

    # noinspection PyPep8Naming
    @classmethod
    def new(cls, G: nx.Graph) -> Optional[Self]:
        data = cls(G=G)
        if data.pre_filter():
            return data.pre_transform().transform()
        return None

    # noinspection PyPep8Naming
    def __init__(self, G: Optional[nx.Graph] = None):
        super().__init__(G=G)
        if G is not None:  # Allow empty GraphDrawingData when constructing Batch
            self.num_nodes = n = self.G.number_of_nodes()
            self.perm_index = torch.tensor(np.array(list(permutations(range(n), 2)))).T

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if "metaindex" in key:
            return self.perm_index.shape[1]
        return super().__inc__(key, value, *args, **kwargs)

    def __getattr__(self, key: str) -> Any:
        if key not in self and key in self.field_annotations():
            for suffix in ["index", "attr", "weight"]:
                if key.endswith("_" + suffix):
                    metaindex_key = key.replace(suffix, "metaindex")
                    if metaindex_key not in self.field_annotations():
                        continue
                    indexer = getattr(self, metaindex_key)
                    if suffix == "index":
                        indexer = slice(None), indexer
                    return getattr(self, f"perm_{suffix}")[indexer]
            else:
                return None
        return super().__getattr__(key)

    def __getattribute__(self, key):
        value = object.__getattribute__(self, key)
        if isinstance(value, type(self).Field) or value is NotImplemented:
            value = self.__getattr__(key)
        return value

    def pre_filter(self) -> bool:
        assert not isinstance(self, Batch)
        return nx.is_connected(self.G)

    def pre_transform(self) -> Self:
        assert not isinstance(self, Batch)
        return T.Compose([
            NormalizeGraph(),
            AddAdjacencyInfo(attr_name="edge_metaindex"),
            ComputeShortestPath(
                cutoff=None,
                attr_name="apsp_attr",
                weight_name="perm_weight"
            ),
            T.AddLaplacianEigenvectorPE(
                k=3,
                is_undirected=True,
                attr_name="laplacian_eigenvector_pe"
            )
        ])(self)

    def transform(self) -> Self:
        assert not isinstance(self, Batch)
        return T.Compose([
            PopulateGraphAttrs(),
            SampleAggregationEdges(attr_name="aggr_metaindex")
        ])(self)

    def post_transform(self, field: Union[str, Iterable[str], None] = None) -> Self:
        field_dict = self.fields(stage="post_transform")
        if field is None:
            field = field_dict.keys()
        if isinstance(field, str):
            field = [field]
        transforms = T.Compose([field_dict[f].transform for f in field])
        if isinstance(self, Batch):
            return Batch.from_data_list(list(map(transforms, self.to_data_list())))
        return transforms(self)

    def append(self, tensor: torch.Tensor, *, name: str, like: Optional[str] = None):
        if like is not None:
            self[name] = tensor.to(self[like])
        else:
            self[name] = tensor
        if isinstance(self, Batch):
            return BatchAppendColumn(
                attr_name=name,
                like=like,
                dtype=tensor.dtype if like is None else None,
                device=tensor.device if like is None else None
            )(self)
        return self
