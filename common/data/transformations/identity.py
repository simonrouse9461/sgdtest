from ..graph_layout import GraphLayout
from .base_transformation import BaseTransformation


class Identity(BaseTransformation):

    def __init__(self):
        super().__init__()

    def forward(self, layout: GraphLayout) -> GraphLayout:
        return layout
