import random

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from fa2 import ForceAtlas2
import s_gd2


# def _get_pmds_layout(G, pmds_bin='hde/pmds'):
#     indot = str(nx.nx_pydot.to_pydot(G))
#     outdot = subprocess.check_output([pmds_bin], text=True, input=indot)
#     G = nx.nx_pydot.from_pydot(pydot.graph_from_dot_data(outdot)[0])
#     raw_layout = nx.get_node_attributes(G, 'pos')
#     layout = {int(n): tuple(map(float, pos.replace('"', '').split(','))) for n, pos in raw_layout.items()}
#     return dict(sorted(layout.items(), key=lambda pair: pair[0]))

# TODO: convert to class

# TODO: throw error if seed is not supported
def generate_layout(G: nx.Graph, method: str, seed: int = None, draw: bool = False) -> nx.Graph:
    G = G.copy()
    if seed is None:
        seed = random.randint(0, 2**16)
    match method:
        case "fa2":
            random.seed(seed)
            layout = ForceAtlas2(verbose=False).forceatlas2_networkx_layout(G, iterations=2000)
        # case "pmds":
        #     layout = get_pmds_layout(G)
        case "sgd2":
            pos = s_gd2.layout(*zip(*G.edges), random_seed=seed)
            layout = {i: p for i, p in zip(G.nodes, pos.tolist())}
        case "spring":
            layout = getattr(nx.drawing.layout, f'{method}_layout')(G, seed=seed)
        case _:
            if hasattr(nx.drawing.layout, f'{method}_layout'):
                layout = getattr(nx.drawing.layout, f'{method}_layout')(G)
            else:
                G.graph["start"] = seed
                layout = nx.drawing.nx_agraph.graphviz_layout(G, prog=method)
    nx.set_node_attributes(G, layout, name=method)
    if draw:
        nx.draw(G, pos=nx.get_node_attributes(G, name=method))  # TODO: use a unified customized drawing
        plt.axis("equal")
    return G
