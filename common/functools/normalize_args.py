from smartgd.common.collectiontools import *

import inspect
from typing import Callable, Any
from functools import wraps
from frozendict import frozendict


def normalize_args(func: Callable) -> Callable:
    """A decorator that normalizes function arguments.
    Helpful for cache mechanisms that uses function arguments as cache keys.
    1. Transform args and kwargs into canonical callargs dict
    2. Recursively transform mutable dict of mutable collections into immutable dict of immutable collections
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        wrapped = func
        while hasattr(wrapped, "__wrapped__"):
            wrapped = wrapped.__wrapped__
        kwargs = inspect.getcallargs(wrapped, *args, **kwargs)
        return func(**freeze(kwargs))

    return wrapper
