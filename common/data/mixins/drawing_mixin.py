from abc import ABC

import networkx as nx


class DrawingMixin(ABC):

    G: nx.Graph

    def draw(self):
        nx.draw(self.G, pos={i: self.pos[i] for i in range(len(self.pos))})
