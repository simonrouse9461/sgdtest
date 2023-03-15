from functools import singledispatch
from typing import Any, Set, Mapping, MutableSequence, Tuple

from frozendict import frozendict
from frozenlist import FrozenList as frozenlist


@singledispatch
def freeze(collection: Any) -> Any:
    return collection


@freeze.register(set)
@freeze.register(frozenset)
def _(collection: Set) -> frozenset:
    return frozenset({freeze(v) for v in collection})


@freeze.register(dict)
@freeze.register(frozendict)
def _(collection: Mapping) -> frozendict:
    return frozendict({k: freeze(v) for k, v in collection.items()})


@freeze.register(list)
@freeze.register(frozenlist)
def _(collection: MutableSequence) -> frozenlist:
    fl = frozenlist([freeze(v) for v in collection])
    fl.freeze()
    return fl


@freeze.register(tuple)
def _(collection: Tuple) -> tuple:
    return tuple(freeze(v) for v in collection)