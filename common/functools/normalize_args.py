from smartgd.common.collectiontools import *

import inspect
from typing import Callable, Any, Optional
from functools import wraps, partial
from frozendict import frozendict


def normalize_args(func: Optional[Callable] = None, /, *, mapping_fns: dict[str, Callable] = None) -> Callable:
    """A decorator that normalizes function arguments.
    Helpful for cache mechanisms that uses function arguments as cache keys.
    1. Transform args and kwargs into canonical callargs dict
    2. Recursively transform mutable dict of mutable collections into immutable dict of immutable collections
    """
    if mapping_fns is None:
        mapping_fns = {}
    if func is None:
        return partial(normalize_args, mapping_fns=mapping_fns)

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        wrapped = func
        while hasattr(wrapped, "__wrapped__"):
            wrapped = wrapped.__wrapped__
        kwargs = inspect.getcallargs(wrapped, *args, **kwargs)
        kwargs = {k: mapping_fns.get(k, lambda x: x)(v) for k, v in kwargs.items()}
        return func(**freeze(kwargs))

    return wrapper
