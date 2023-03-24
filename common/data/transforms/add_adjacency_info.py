import networkx as nx
import numpy as np
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


class AddAdjacencyInfo(BaseTransform):

    def __init__(self,
                 attr_name: str = "edge_metaindex"):
        super().__init__()
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        # FIXME: THIS IS A HACK!
        # TODO: generate edge based on perm_index edge order
        metaindex = [u * (data.num_nodes - 1) + v - (u < v) for u, v in data.G.edges]
        data[self.attr_name] = torch.tensor(np.array(metaindex)).to(data.device)
        return data
