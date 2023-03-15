from ..graph_layout import GraphLayout
from .base_transformation import BaseTransformation

import torch_scatter


class RescaleByStress(BaseTransformation):

    def __init__(self):
        super().__init__()

    def forward(self, layout: GraphLayout) -> GraphLayout:
        dist = (layout.full_dst_pos - layout.full_src_pos).norm(dim=1)
        u_over_d = dist / layout.edge_full.attr.shortest_path
        scatterd_u_over_d_2 = torch_scatter.scatter(u_over_d ** 2, layout.full_batch_index)
        scatterd_u_over_d = torch_scatter.scatter(u_over_d, layout.full_batch_index)
        scale = scatterd_u_over_d_2 / scatterd_u_over_d
        return layout(layout.pos / scale[layout.batch][:, None])
