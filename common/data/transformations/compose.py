from ..graph_layout import GraphLayout
from .base_transformation import BaseTransformation

from torch import nn


class Compose(BaseTransformation):

    def __init__(self, *transformations: BaseTransformation):
        super().__init__()
        self.transformations: nn.ModuleList[BaseTransformation] = nn.ModuleList(transformations)

    def forward(self, layout: GraphLayout) -> GraphLayout:
        for transformation in self.transformations:
            layout = transformation(layout)
        return layout
