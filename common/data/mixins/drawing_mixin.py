from abc import ABC

import networkx as nx
import matplotlib.pyplot as plt


class DrawingMixin(ABC):

    G: nx.Graph

    def draw(self):
        pos = {i: self.pos[i].tolist() for i in range(len(self.pos))}
        nx.draw(self.G.to_undirected(), pos=pos)
        plt.axis("equal")
