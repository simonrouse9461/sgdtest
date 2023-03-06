from smartgd.global_constants import PREDICTION_S3_BUCKET
from smartgd.common.decorators import normalize_args

import pickle
from typing import Optional
from typing_extensions import Self
from cachetools import cached

import numpy as np
from lightning_lite.utilities import cloud_io


class LayoutManager:

    @classmethod
    def get_default_manager(cls, dataset: str) -> Self:
        return cls(root=f"s3://{PREDICTION_S3_BUCKET}", dataset=dataset)

    def __init__(self, *, root: str = ".", dataset: str):
        self.dirpath = f"{root}/{dataset}"
        self.fs = cloud_io.get_filesystem(self.dirpath)

    @staticmethod
    def _get_params_str(params: Optional[dict]) -> str:
        if params is None:
            return ""
        params_list = sorted(params.items(), key=lambda kv: kv[0])
        return "(" + ",".join([f"{k}={v}" for k, v in params_list]) + ")"

    def _get_file_path(self, name: str, params: Optional[dict]) -> str:
        return f"{self.dirpath}/{name}{self._get_params_str(params)}.pkl"

    def save(self, layout_dict: dict, *, name: str, params: Optional[dict] = None):
        with self.fs.open(self._get_file_path(name, params), "wb") as fout:
            pickle.dump(layout_dict, fout)
        self.evict_cache(name=name, params=params)

    @normalize_args
    @cached({})
    def load(self, *, name: str, params: Optional[dict] = None) -> dict[..., np.ndarray]:
        with self.fs.open(self._get_file_path(name, params), "rb") as fin:
            return pickle.load(fin)

    def evict_cache(self, *, name: str, params: Optional[dict] = None) -> Optional[dict[..., np.ndarray]]:
        cache_key = self.load.cache_key(self=self, name=name, params=params)  # must be kw_only to match cache_key
        return self.load.cache.pop(cache_key, None)

    def clear_cache(self):
        self.load.cache.clear()
