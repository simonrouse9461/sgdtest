from smartgd.constants import DATASET_ROOT
from .utils import s3_dataset_syncing

import os
import re
from typing import Callable, Optional

from tqdm.auto import tqdm
import numpy as np
import torch
import torch_geometric as pyg
import networkx as nx


@s3_dataset_syncing
class RomeDataset(pyg.data.InMemoryDataset):

    ROME_DEFAULT_URL: str = "https://www.graphdrawing.org/download/rome-graphml.tgz"

    def __init__(self, *,
                 url: str = ROME_DEFAULT_URL,
                 root: str = DATASET_ROOT,
                 name: str = "Rome",
                 layout_initializer: Optional[Callable] = None,  # TODO: use transform to do this
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.url = url
        self.name = name
        self.initializer = layout_initializer or nx.drawing.random_layout
        super().__init__(f"{root}/{name}", transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        meta_file = "rome/Graph.log"
        if os.path.exists(metadata_path := f"{self.raw_dir}/{meta_file}"):
            return list(map(lambda f: f"rome/{f}.graphml", self.get_graph_names(metadata_path)))
        else:
            return [meta_file]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @classmethod
    def get_graph_names(cls, logfile):
        with open(logfile) as fin:
            for line in fin.readlines():
                if match := re.search(r'name: (grafo\d+\.\d+)', line):
                    yield f'{match.group(1)}'

    def process_raw(self):
        name_regex = r"grafo(\d+)\.(\d+)"

        def key(path):
            match = re.search(name_regex, path)
            return int(match.group(1)), int(match.group(2))
        graphmls = sorted(self.raw_paths, key=key)
        for file in tqdm(graphmls, desc=f"Loading graphs"):
            G = nx.read_graphml(file)
            G.graph["name"] = re.search(name_regex, file).group(0)
            if nx.is_connected(G):  # TODO: use dataset filter
                yield nx.convert_node_labels_to_integers(G)

    def convert(self, G):
        apsp = dict(nx.all_pairs_shortest_path_length(G))
        init_pos = torch.tensor(np.array(list(self.initializer(G).values())))
        full_edges, attr_d = zip(*[((u, v), d) for u in apsp for v, d in apsp[u].items()])
        adj_index = pyg.utils.to_undirected(torch.tensor(list(G.edges)).T)
        full_index, d = pyg.utils.remove_self_loops(*pyg.utils.to_undirected(
            edge_index=torch.tensor(full_edges).T,
            edge_attr=torch.tensor(attr_d),
            reduce="mean"
        ))
        edge_index = full_index
        return pyg.data.Data(
            G=G,
            pos=init_pos,
            edge_index=edge_index,
            d_attr=d,
            full_index=full_index,
            adj_index=adj_index,
            n=G.number_of_nodes(),
            m=G.number_of_edges(),
            name=G.graph["name"],
        )

    def download(self):
        pyg.data.download_url(self.url, self.raw_dir)
        pyg.data.extract_tar(f'{self.raw_dir}/rome-graphml.tgz', self.raw_dir)

    def process(self):
        data_list = map(self.convert, self.process_raw())

        if self.pre_filter is not None:
            data_list = filter(self.pre_filter, data_list)

        if self.pre_transform is not None:
            data_list = map(self.pre_transform, data_list)

        data, slices = self.collate(list(data_list))
        torch.save((data, slices), self.processed_paths[0])
