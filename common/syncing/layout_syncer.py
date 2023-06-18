from smartgd.constants import DYNAMODB_REGION, LAYOUTS_DB_PREFIX
from smartgd.common.functools import normalize_args
from .utils.identifier_utils import IdentifierUtils
from .model_alias_resolver import ModelAliasResolver

from collections import defaultdict
from decimal import Decimal
import pickle
from typing import Optional, Any, Union, Iterable, Mapping
from typing_extensions import Self
from cachetools import cached
import more_itertools

import boto3
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# TODO: use pynamodb
class LayoutSyncer:

    METHOD_KEY = "method"
    GRAPHID_KEY = "graph_id"
    POS_KEY = "__pos__"

    resolver = ModelAliasResolver()

    def __init__(self, dataset_name: str):
        self.db = boto3.resource("dynamodb", region_name=DYNAMODB_REGION)
        self.table_name = LAYOUTS_DB_PREFIX + dataset_name
        self.table = self.db.Table(self.table_name)

    # TODO: make a separate class for DB clients
    def _db_update(self, method: str, graph_id: str, **attrs) -> dict:
        set_expr = ",".join([f"#{key}=:{key}" for key, value in attrs.items() if value is not None])
        remove_expr = ",".join([f"#{key}" for key, value in attrs.items() if value is None])
        update_exprs = [f"set {set_expr}" if set_expr else "", f"remove {remove_expr}" if remove_expr else ""]
        return self.table.update_item(
            Key=dict(method=method, graph_id=graph_id),
            UpdateExpression=" ".join(filter(bool, update_exprs)),
            ExpressionAttributeNames={f"#{key}": key for key in attrs},
            ExpressionAttributeValues={f":{key}": value for key, value in attrs.items() if value is not None},
        )

    def _db_put(self, method: str, graph_id: str, _writer=None, **attrs) -> dict:
        _writer = _writer or self.table
        return _writer.put_item(Item=dict(
            method=method,
            graph_id=graph_id,
            **attrs
        ))

    def _db_delete(self, method: str, graph_id: str, _writer=None) -> dict:
        _writer = _writer or self.table
        return _writer.delete_item(Key=dict(
            method=method,
            graph_id=graph_id,
        ))

    def _db_get_item(self, method: str, graph_id: str) -> dict:
        return self.table.get_item(Key=dict(
            method=method,
            graph_id=graph_id
        ))["Item"]

    def _db_has_record(self, method: str, graph_id: str, attr: Optional[str] = None) -> bool:
        response = self.table.get_item(Key=dict(
            method=method,
            graph_id=graph_id
        ))
        if attr is None:
            return "Item" in response
        return "Item" in response and attr in response["Item"]

    def _db_get_batch(self, method: str, graph_ids: Iterable[str]) -> Iterable[dict]:
        for chunk in more_itertools.chunked(tqdm(graph_ids, desc=f"Fetching layout data '{method}'"), n=100):
            yield from self.db.batch_get_item(RequestItems={
                self.table_name: dict(Keys=[dict(
                    method=method,
                    graph_id=graph_id
                ) for graph_id in chunk])
            })['Responses'][self.table_name]

    def _db_get_all(self, method: str) -> Iterable[dict]:
        pagination_token = None
        query_params = dict(
            KeyConditionExpression="#method=:method",
            ExpressionAttributeNames={"#method": self.METHOD_KEY},
            ExpressionAttributeValues={":method": method},
        )
        with tqdm(desc=f"Fetching layout data '{method}'") as bar:
            while True:
                pagination_params = dict(ExclusiveStartKey=pagination_token) if pagination_token else {}
                response = self.table.query(**query_params | pagination_params)
                bar.update(len(items := response["Items"]))
                yield from items
                if (pagination_token := response.get('LastEvaluatedKey', None)) is None:
                    return

    def _db_get(self, method: str, graph_id: Union[None, Iterable[str], str]) -> dict | Iterable[dict]:
        if graph_id is None:
            return self._db_get_all(method)
        elif isinstance(graph_id, str):
            return self._db_get_item(method, graph_id)
        elif isinstance(graph_id, Iterable):
            return self._db_get_batch(method, graph_id)
        else:
            raise TypeError(f"Unknow type '{type(graph_id)}' for 'graph_id'!")

    # TODO: write a separate class for DynamoDB serialization
    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, float):
            value = Decimal(str(value))
            if value.is_nan() or value.is_infinite():
                value = str(value)
        return value

    def _deserialize_value(self, value: Any) -> Any:
        if isinstance(value, Decimal) or value in [str(Decimal("nan")), str(Decimal("inf"))]:
            value = float(value)
        return value

    def _serialize_layout(self, layout: Optional[np.ndarray]) -> Optional[list]:
        if isinstance(layout, np.ndarray):
            return np.vectorize(self._serialize_value)(layout.astype(float)).tolist()
        return None

    def _deserialize_layout(self, layout: list) -> np.ndarray:
        return np.array(layout, dtype=float)

    def _serialize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.applymap(self._serialize_value)
        if self.POS_KEY in data:
            data[self.POS_KEY] = data[self.POS_KEY].map(self._serialize_layout)
        return data

    @resolver.wrapper
    @normalize_args
    def exist(self, *,
              name: str,
              graph_id: str,
              params: Optional[dict] = None,
              attr: Optional[str] = None):
        return self._db_has_record(method=IdentifierUtils.get_identifier(name=name, params=params),
                                   graph_id=graph_id,
                                   attr=attr)

    @resolver.wrapper
    @normalize_args
    def exist_layout(self, *,
                     name: str,
                     graph_id: str,
                     params: Optional[dict] = None):
        return self._db_has_record(method=IdentifierUtils.get_identifier(name=name, params=params),
                                   graph_id=graph_id,
                                   attr=self.POS_KEY)

    @resolver.wrapper
    @normalize_args
    def update(self, *,
               name: str,
               graph_id: str,
               params: Optional[dict] = None,
               layout: Optional[np.ndarray] = None,
               # TODO: implement namespace
               metadata: Optional[dict] = None):
        if layout is not None:
            self._db_update(method=IdentifierUtils.get_identifier(name=name, params=params),
                            graph_id=graph_id,
                            __pos__=self._serialize_layout(layout))
        if metadata is not None:
            self._db_update(full_name=IdentifierUtils.get_identifier(name=name, params=params),
                            graph_id=graph_id,
                            **metadata)
        self.evict_cache(name=name, params=params)

    @resolver.wrapper
    @normalize_args
    def batch_update(self, *,
                     name: str,
                     params: Optional[dict] = None,
                     layout_dict: Optional[dict[str, np.ndarray]] = None,
                     # TODO: implement namespace
                     metadata: Union[None, dict[str, dict], pd.DataFrame] = None,
                     ignore_cache: bool = True):
        if layout_dict is None:
            layout_dict = {}
        if metadata is None:
            metadata_dict = {}
        elif isinstance(metadata, pd.DataFrame):
            metadata_dict = metadata.T.to_dict()
        elif isinstance(metadata, Mapping):
            metadata_dict = dict(metadata)
        else:
            raise TypeError(f"Unsupported data type {type(metadata)} for metadata!")
        data_dict = defaultdict(dict)
        index_list = list(set(layout_dict or {}) | set(metadata_dict or {}))
        if ignore_cache:
            self.evict_cache(name=name, params=params)
        data_dict.update(self.load(
            name=name,
            params=params,
            graph_ids=set(layout_dict) | set(metadata_dict)
        ).T.to_dict())
        if layout_dict is not None:
            for graph_id, layout in layout_dict.items():
                data_dict[graph_id][self.POS_KEY] = layout
        if metadata_dict is not None:
            for graph_id, metadata in metadata_dict.items():
                data_dict[graph_id].update(metadata.items())
                data_dict[graph_id] = {k: v for k, v in data_dict[graph_id].items() if v is not None}
        data_dict = self._serialize_data(pd.DataFrame(data_dict).T.loc[index_list]).T.to_dict()
        method = IdentifierUtils.get_identifier(name=name, params=params)
        with self.table.batch_writer() as writer:
            for graph_id, data in tqdm(data_dict.items(), desc=f"Batch updating '{method}'"):
                self._db_put(_writer=writer,
                             method=method,
                             graph_id=graph_id,
                             **data)
        self.evict_cache(name=name, params=params)

    @resolver.wrapper
    @normalize_args
    def batch_delete(self, *,
                     name: str,
                     params: Optional[dict] = None,
                     graph_ids: Optional[Iterable[str]] = None):
        raise NotImplementedError

    @resolver.wrapper
    @normalize_args
    def delete_all(self, *,
                   name: str,
                   params: Optional[dict] = None):
        self.evict_cache(name=name, params=params)
        data_dict = self.load(name=name, params=params).T.to_dict()
        method = IdentifierUtils.get_identifier(name=name, params=params)
        with self.table.batch_writer() as writer:
            for graph_id in tqdm(data_dict, desc=f"Batch deleting '{method}'"):
                self._db_delete(_writer=writer,
                                method=method,
                                graph_id=graph_id)
        self.evict_cache(name=name, params=params)

    # TODO: element-wise cache
    @resolver.wrapper
    @normalize_args(mapping_fns=dict(
        graph_ids=lambda x: set(x) if x is not None else x
    ))
    @cached(dict())
    def load(self, *,
             name: str,
             params: Optional[dict] = None,
             graph_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
        method = IdentifierUtils.get_identifier(name=name, params=params)
        data = list(self._db_get(method, graph_ids))
        df = pd.DataFrame(data) if len(data) > 0 else pd.DataFrame(columns=[self.METHOD_KEY, self.GRAPHID_KEY])
        df = df.drop(self.METHOD_KEY, axis=1).set_index(self.GRAPHID_KEY)
        df = df.applymap(self._deserialize_value)
        if self.POS_KEY in df:
            df[self.POS_KEY] = df[self.POS_KEY].map(self._deserialize_layout)
        elif len(data) > 0:
            print(f"Warning: '{method}' does not have '{self.POS_KEY}' attribute!")
        if graph_ids is not None and len(data) < len(list(graph_ids)):
            print(f"Warning: {len(data)} of {len(list(graph_ids))} records retrieved!")
        return df

    @resolver.wrapper
    @normalize_args
    def load_layout_dict(self, *,
                         name: str,
                         params: Optional[dict] = None,
                         graph_ids: Optional[Iterable[str]] = None) -> dict[str, np.ndarray]:
        df = self.load(name=name, params=params, graph_ids=graph_ids)
        return df[~df[self.POS_KEY].isna()][self.POS_KEY].to_dict()

    @resolver.wrapper
    @normalize_args
    def load_layout(self, *,
                    name: str,
                    params: Optional[dict] = None,
                    graph_id: str) -> np.ndarray:
        return self.load_layout_dict(name=name, params=params, graph_ids=[graph_id])[graph_id]

    @resolver.wrapper
    @normalize_args
    def load_metadata(self, *,
                      name: str,
                      params: Optional[dict] = None,
                      return_dict: bool = False,
                      graph_ids: Optional[Iterable[str]] = None) -> pd.DataFrame | dict:
        metadata = self.load(name=name, params=params, graph_ids=graph_ids).drop(self.POS_KEY, axis=1, errors="ignore")
        return metadata.T.to_dict() if return_dict else metadata

    @resolver.wrapper
    @normalize_args
    def evict_cache(self, *,
                    name: str,
                    params: Optional[dict] = None,
                    graph_ids: Optional[Iterable[str]] = None) -> Optional[dict[str, np.ndarray]]:
        cache_key = self.load.cache_key(self=self, name=name, params=params, graph_ids=graph_ids)  # must be kw_only
        return self.load.cache.pop(cache_key, None)

    def clear_cache(self):
        self.load.cache.clear()
