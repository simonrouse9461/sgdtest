from smartgd.constants import DATASET_ROOT
from smartgd.common.data import GraphDrawingData
from .utils import s3_dataset_syncing

import os
import re
import hashlib
from typing import Callable, Optional, TypeVar, Iterator

from tqdm.auto import tqdm
import numpy as np
import torch
import torch_geometric as pyg
import networkx as nx


DATATYPE = TypeVar("DATATYPE", bound=pyg.data.Data)


@s3_dataset_syncing  # TODO: make it work with multiple dataloader workers
class RomeDataset(pyg.data.InMemoryDataset):

    DEFAULT_NAME = "Rome"
    DEFAULT_URL = "https://www.graphdrawing.org/download/rome-graphml.tgz"
    GRAPH_NAME_REGEX = re.compile(r"grafo(\d+)\.(\d+)")

    def __init__(self, *,
                 url: str = DEFAULT_URL,
                 root: str = DATASET_ROOT,
                 name: str = DEFAULT_NAME,
                 index: Optional[list[str]] = None,
                 datatype: type[DATATYPE] = GraphDrawingData):
        self.url: str = url
        self.name: str = name
        self.index: Optional[list[str]] = index
        self.datatype: type[DATATYPE] = datatype
        super().__init__(
            root=os.path.join(root, name),
            transform=self.datatype.transform,
            pre_transform=self.datatype.pre_transform,
            pre_filter=self.datatype.pre_filter
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    def _parse_metadata(self, logfile: str) -> Iterator[str]:
        with open(logfile) as fin:
            for line in fin.readlines():
                if match := self.GRAPH_NAME_REGEX.search(line):
                    yield match.group(0)

    @property
    def raw_file_names(self) -> list[str]:
        metadata_file = "rome/Graph.log"
        if os.path.exists(metadata_path := os.path.join(self.raw_dir, metadata_file)):
            return list(map(lambda f: f"rome/{f}.graphml", self._parse_metadata(metadata_path)))
        return [metadata_file]

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt", "index.txt"]

    def generate(self) -> Iterator[nx.Graph]:
        def key(path):
            match = self.GRAPH_NAME_REGEX.search(path)
            return int(match.group(1)), int(match.group(2))
        for file in tqdm(sorted(self.raw_paths, key=key), desc=f"Loading graphs"):
            G = nx.read_graphml(file)
            G.graph.update(dict(
                name=self.GRAPH_NAME_REGEX.search(file).group(0),
                dataset=self.name
            ))
            yield G

    def download(self) -> None:
        pyg.data.download_url(self.url, self.raw_dir)
        pyg.data.extract_tar(f'{self.raw_dir}/rome-graphml.tgz', self.raw_dir)

    def process(self) -> None:
        G_list = list(self.generate())
        if self.index is None:
            self.index = [G.graph["name"] for G in G_list]
        else:
            G_dict = {G.graph["name"]: G for G in G_list}
            G_list = [G_dict[name] for name in self.index]
        data_list = map(self.datatype, G_list)
        data_list = filter(self.pre_filter, data_list)
        data_list = map(self.pre_transform, data_list)
        data, slices = self.collate(list(data_list))
        torch.save((data, slices), self.processed_paths[0])
        with open(self.processed_paths[1], "w") as index_file:
            index_file.write("\n".join(self.index))
