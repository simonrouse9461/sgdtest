import re
import inspect
import tempfile
import importlib.util
import importlib.machinery
from typing import TypeVar
from dataclasses import is_dataclass
from torch import nn
from torch.jit._dataclass_impls import synthesize__init__


T = TypeVar("T", bound=type)


# TODO: support __eq__ and __repr__
def jittable(cls: T) -> T:
    assert is_dataclass(cls)
    src = synthesize__init__(cls).source.replace(f"{cls.__module__}.{cls.__qualname__}", cls.__name__)
    # get `globals()` from the caller
    globals_dict = {k: v for k, v in inspect.stack()[1][0].f_globals.items()
                    if not re.match(r"__\w+__", k)}
    # This is to handle `from ... import`, where the imported names are already in `globals()`
    for param in inspect.signature(cls.__init__).parameters.values():
        if param.annotation.__module__.split(".")[0] not in globals_dict:
            src = src.replace(f"{param.annotation.__module__}.", "")
    # write source code into a temp file to allow `inspect.getsource` to load the source code
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write(src)
        tmp.flush()
    loader = importlib.machinery.SourceFileLoader(cls.__name__, tmp.name)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    for k, v in globals_dict.items():
        setattr(module, k, v)
    setattr(module, cls.__name__, cls)
    spec.loader.exec_module(module)
    cls.__init__ = module.__init__
    # This is to prevent `jit.script` from recursively scripting annotations
    cls.__annotations__.clear()
    if not issubclass(cls, nn.Module):
        # Let `@jit.script` treat `cls` as a normal class
        cls.__dataclass_fields_backup__ = cls.__dataclass_fields__
        del cls.__dataclass_fields__
    # Write `cls` back to caller's `globals()`, so that nested classes are visible to `@jit.script`
    inspect.stack()[1][0].f_globals[cls.__name__] = cls
    return cls
