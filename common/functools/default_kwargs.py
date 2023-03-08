import inspect
from typing import Callable, Any, Optional, Union
from functools import wraps


def default_kwargs(_kwargs_or_func: Union[Optional[dict], Callable] = None, /, *,
                   undefined: Union[Any, Callable[[Any], bool]] = Ellipsis):

    if _kwargs_or_func is None:
        _kwargs_or_func = {}

    def v_defined(kv):
        if isinstance(undefined, Callable):
            return not undefined(kv[1])
        return kv[1] is not undefined

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            args_keys = list(dict(zip(inspect.signature(func).parameters.keys(), args)).keys())
            defined_args = dict(filter(v_defined, zip(args_keys, args)))
            defined_kwargs = dict(filter(v_defined, kwargs.items()))
            return func(**_kwargs_or_func | defined_args | defined_kwargs)
        return wrapper

    if isinstance(_kwargs_or_func, Callable):
        return default_kwargs()(_kwargs_or_func)
    return decorator
