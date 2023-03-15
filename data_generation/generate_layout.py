import os
import random
from typing import Optional
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from fa2 import ForceAtlas2
import s_gd2


def _get_pmds_layout(G: nx.Graph, pmds_bin: str = os.path.join(os.path.dirname(__file__), "hde/pmds")):
    indot = str(nx.nx_pydot.to_pydot(G))
    outdot = subprocess.check_output([pmds_bin], text=True, input=indot, stderr=subprocess.DEVNULL)
    G = nx.nx_pydot.from_pydot(pydot.graph_from_dot_data(outdot)[0])
    raw_layout = nx.get_node_attributes(G, "pos")
    layout = {int(n): tuple(map(float, pos.replace('"', '').split(","))) for n, pos in raw_layout.items()}
    return dict(sorted(layout.items(), key=lambda pair: pair[0]))


# TODO: convert to class
def generate_layout(G: nx.Graph, method: str, seed: Optional[int] = None, draw: bool = False) -> nx.Graph:
    G = G.copy()
    manual_seed = random.randint(0, 2**16) if seed is None else seed
    match method:
        case "fa2":
            random.seed(manual_seed)
            layout = ForceAtlas2(verbose=False).forceatlas2_networkx_layout(G, iterations=2000)
        case "pmds":
            assert seed is None, f"Seed is not supported for {method}!"
            layout = _get_pmds_layout(G)
        case "sgd2":
            pos = s_gd2.layout(*zip(*G.edges), random_seed=manual_seed)
            layout = {i: p for i, p in zip(G.nodes, pos.tolist())}
        case "spring":
            layout = getattr(nx.drawing.layout, f'{method}_layout')(G, seed=manual_seed)
        case _:
            if hasattr(nx.drawing.layout, f'{method}_layout'):
                assert seed is None, f"Seed is not supported for {method}!"
                layout = getattr(nx.drawing.layout, f'{method}_layout')(G)
            else:
                G.graph["start"] = manual_seed
                layout = nx.drawing.nx_agraph.graphviz_layout(G, prog=method)
    nx.set_node_attributes(G, layout, name=method)
    if draw:
        nx.draw(G, pos=nx.get_node_attributes(G, name=method))  # TODO: use a unified customized drawing
        plt.axis("equal")
    return G


class LayoutGenerator:

    def __init__(self, method: str):
        pass

    def generate(self, G: nx.Graph, *, seed: Optional[int] = None):
        pass
