from smartgd.constants import DYNAMODB_REGION, ALIAS_DB_NAME

from typing import Optional
from functools import wraps

import boto3


class ModelAliasResolver:

    # TODO: add write capability

    def __init__(self):
        self.db = boto3.resource("dynamodb", region_name=DYNAMODB_REGION).Table(ALIAS_DB_NAME)

    def lookup(self, alias: str) -> Optional[dict]:
        method_spec = self.db.get_item(Key=dict(
            alias=alias,
        )).get("Item", None)
        if method_spec is not None:
            del method_spec["alias"]
            if "params" not in method_spec:
                method_spec["params"] = None
        return method_spec

    def list(self):
        return self.db.scan()["Items"]

    def resolve(self, *, name: str, params: Optional[dict] = None) -> dict:
        if (spec := self.lookup(name)) is not None and params is None:
            return spec
        return dict(name=name, params=params)

    # TODO: Be able to specify parameter names
    def wrapper(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs.update(self.resolve(name=kwargs["name"], params=kwargs.get("params", None)))
            return func(*args, **kwargs)
        return wrapper
