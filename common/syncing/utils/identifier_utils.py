from smartgd.common.collectiontools import compact_repr

from typing import Optional


class IdentifierUtils:

    @staticmethod
    def _get_params_repr(params: Optional[dict]) -> str:
        return "" if params is None else compact_repr(params, sort_list=True, type_agnostic=True)

    @classmethod
    def get_identifier(cls, *, name: str, params: Optional[dict]) -> str:
        return name + cls._get_params_repr(params)
