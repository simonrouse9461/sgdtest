from .generate_layout import generate_layout

from functools import partial
import multiprocessing

import networkx as nx
from tqdm.auto import tqdm


def batch_generate_layouts(G_list: list[nx.Graph], method: str, seed: int = None) -> list[nx.Graph]:
    G_dict = {}
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    func = partial(generate_layout, method=method, seed=seed)
    for G in tqdm(pool.imap_unordered(func, G_list), total=len(G_list), desc=f"Generating {method} layouts..."):
        G_dict[G.graph["name"]] = G
    return [G_dict[old_G.graph["name"]] for old_G in G_list]
