from itertools import permutations

import networkx as nx
import numpy as np
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


class CreateEdgePairs(BaseTransform):

    def __init__(self,
                 edge_metaindex_name: str = "edge_metaindex",
                 attr_name: str = "edge_pair_metaindex"):
        super().__init__()
        self.edge_metaindex_name = edge_metaindex_name
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        metaindex = list(permutations(getattr(data, self.edge_metaindex_name), 2))
        data[self.attr_name] = torch.tensor(np.array(metaindex)).T.to(data.device)
        return data
