import inspect
from typing import Optional, Callable
from functools import wraps


def save_init_args(_func: Optional[Callable] = None, /, *,
                   init_args_attr: str = "init_args"):
    def decorator(__init__: Callable):
        assert __init__.__name__ == "__init__"

        @wraps(__init__)
        def wrapper(self, *args, **kwargs):
            call_args = inspect.getcallargs(__init__, self, *args, **kwargs)
            del call_args["self"]
            setattr(self, init_args_attr, call_args)
            return __init__(self, *args, **kwargs)
        return wrapper
    if _func:
        return save_init_args()(_func)
    return decorator
