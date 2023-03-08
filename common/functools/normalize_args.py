import inspect
from typing import Callable, Any
from functools import wraps
from frozendict import frozendict


def normalize_args(func: Callable) -> Callable:
    """A decorator that normalizes function arguments.
    Helpful for cache mechanisms that uses function arguments as cache keys.
    1. Transform args and kwargs into canonical callargs dict
    2. Transform mutable dicts into immutable dicts
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        wrapped = func
        while hasattr(wrapped, "__wrapped__"):
            wrapped = wrapped.__wrapped__
        kwargs = inspect.getcallargs(wrapped, *args, **kwargs)
        kwargs = {k: frozendict(v) if isinstance(v, dict) else v for k, v in kwargs.items()}
        return func(**kwargs)

    return wrapper
