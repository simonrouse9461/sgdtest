from smartgd.common.syncing import LayoutSyncer

from typing import Mapping, Optional

import numpy as np


class LazyLayoutStore(dict):

    def __init__(self, dataset_name: str, name: str, params: Optional[dict] = None):
        super().__init__()
        self.syncer = LayoutSyncer(dataset_name=dataset_name)
        self.name = name
        self.params = params

    def __missing__(self, key: str):
        self[key] = self.syncer.load_layout(name=self.name, params=self.params, graph_id=key)
        return self[key]
