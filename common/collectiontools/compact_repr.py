from functools import singledispatch
from typing import Any, Set, Mapping, MutableSequence, Tuple

from frozendict import frozendict
from frozenlist import FrozenList as frozenlist


@singledispatch
def compact_repr(collection: Any, sort_list: bool = False, type_agnostic: bool = False) -> Any:
    return str(collection)


@compact_repr.register(set)
@compact_repr.register(frozenset)
def _(collection: Set, sort_list: bool = False, type_agnostic: bool = False) -> str:
    prefix, suffix = ("(", ")") if type_agnostic else ("{", "}")
    normalized = map(lambda x: compact_repr(x, sort_list, type_agnostic), collection)
    normalized = sorted(normalized)
    return prefix + ",".join(normalized) + suffix


@compact_repr.register(dict)
@compact_repr.register(frozendict)
def _(collection: Mapping, sort_list: bool = False, type_agnostic: bool = False) -> str:
    prefix, suffix = ("(", ")") if type_agnostic else ("{", "}")
    normalized = (
        (compact_repr(k, sort_list, type_agnostic),
         compact_repr(v, sort_list, type_agnostic))
        for k, v in collection.items()
    )
    normalized = sorted(normalized)
    return prefix + ",".join(f"{k}={v}" for k, v in normalized) + suffix


@compact_repr.register(list)
@compact_repr.register(frozenlist)
def _(collection: MutableSequence, sort_list: bool = False, type_agnostic: bool = False) -> str:
    prefix, suffix = ("(", ")") if type_agnostic else ("[", "]")
    normalized = map(lambda x: compact_repr(x, sort_list, type_agnostic), collection)
    if sort_list:
        normalized = sorted(normalized)
    return prefix + ",".join(normalized) + suffix


@compact_repr.register(tuple)
def _(collection: Tuple, sort_list: bool = False, type_agnostic: bool = False) -> str:
    prefix, suffix = ("(", ")") if type_agnostic else ("(", ")")
    normalized = map(lambda x: compact_repr(x, sort_list, type_agnostic), collection)
    return prefix + ",".join(normalized) + suffix
