from typing import TypeVar


T = TypeVar("T", bound=type)


def jittable_recover(cls: T) -> T:
    if not hasattr(cls, "__dataclass_fields_backup__"):
        return cls
    cls.__dataclass_fields__ = cls.__dataclass_fields_backup__
    for name, field in cls.__dataclass_fields__.items():
        jittable_recover(field.type)
    return cls
