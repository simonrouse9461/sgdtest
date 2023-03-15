from smartgd.constants import PREDICTION_S3_BUCKET
from smartgd.common.functools import normalize_args
from smartgd.common.collectiontools import compact_repr

import pickle
from typing import Optional
from typing_extensions import Self
from cachetools import cached

import numpy as np
from lightning_lite.utilities import cloud_io


class LayoutSyncer:

    @classmethod
    def get_default_syncer(cls, dataset: str) -> Self:
        return cls(root=f"s3://{PREDICTION_S3_BUCKET}", dataset=dataset)

    def __init__(self, *, root: str = ".", dataset: str):
        self.dirpath = f"{root}/{dataset}"
        self.fs = cloud_io.get_filesystem(self.dirpath)

    @staticmethod
    def _get_params_str(params: Optional[dict]) -> str:
        return "" if params is None else compact_repr(params, sort_list=True, type_agnostic=True)

    def _get_file_path(self, name: str, params: Optional[dict]) -> str:
        return f"{self.dirpath}/{name}{self._get_params_str(params)}.pkl"

    @normalize_args
    def save(self, layout_dict: dict, *, name: str, params: Optional[dict] = None):
        with self.fs.open(self._get_file_path(name, params), "wb") as fout:
            pickle.dump(layout_dict, fout)
        self.evict_cache(name=name, params=params)

    @normalize_args
    def exists(self, *, name: str, params: Optional[dict] = None) -> bool:
        return self.fs.exists(self._get_file_path(name, params))

    @normalize_args
    @cached({})
    def load(self, *, name: str, params: Optional[dict] = None) -> dict[str, np.ndarray]:
        with self.fs.open(self._get_file_path(name, params), "rb") as fin:
            return pickle.load(fin)

    @normalize_args
    def evict_cache(self, *, name: str, params: Optional[dict] = None) -> Optional[dict[str, np.ndarray]]:
        cache_key = self.load.cache_key(self=self, name=name, params=params)  # must be kw_only to match cache_key
        return self.load.cache.pop(cache_key, None)

    def clear_cache(self):
        self.load.cache.clear()
