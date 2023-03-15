from itertools import permutations
from typing import Optional

import numpy as np
from scipy import spatial, sparse
import networkx as nx


def delaunay(pos, eps=1e-5, retry=10):
    offset = 0
    for _ in range(retry + 1):
        try:
            tri = spatial.Delaunay(points=pos + offset)
            break
        except spatial.qhull.QhullError as e:
            offset = np.random.random(pos.shape) * eps
    else:
        raise e
    return np.unique(tri.simplices[:, list(permutations(range(3), 2))].reshape([-1, 2]), axis=0)


# TODO: torchfy
def gabriel(pos, edge_set, eps=1e-5):
    tree = spatial.KDTree(pos)
    c = pos[edge_set]
    m = c.mean(axis=1)
    d = np.linalg.norm(c[:, 0, :] - c[:, 1, :], axis=1)
    dm = tree.query(x=m, k=1)[0]
    return edge_set[dm >= d / 2 * (1 - eps)]


# TODO: torchfy
def rng(pos, edge_set, eps=1e-5):
    tree = spatial.KDTree(pos)
    c = pos[edge_set]
    d = np.linalg.norm(c[:, 0, :] - c[:, 1, :], axis=1)
    p0 = tree.query_ball_point(x=c[:, 0, :], r=d*(1 - eps))
    p1 = tree.query_ball_point(x=c[:, 1, :], r=d*(1 - eps))
    p0m = sparse.lil_matrix((len(edge_set), len(pos)))
    p0m.rows, p0m.data = p0, list(map(np.ones_like, p0))
    p1m = sparse.lil_matrix((len(edge_set), len(pos)))
    p1m.rows, p1m.data = p1, list(map(np.ones_like, p1))
    return edge_set[~(p0m.toarray().astype(bool) & p1m.toarray().astype(bool)).any(axis=1)]


# TODO: torchfy
def jaccard_index(edges, shape_edges):
    n = max(edges.max(), shape_edges.max()) + 1
    adj = sparse.coo_matrix((np.ones_like(edges[:, 0]), edges.T), (n, n)).astype(bool).toarray()
    shape_adj = sparse.coo_matrix((np.ones_like(shape_edges[:, 0]), shape_edges.T), (n, n)).astype(bool).toarray()
    assert np.all(adj.T == adj) and np.all(shape_adj.T == shape_adj)
    return np.mean((adj & shape_adj).sum(axis=1) / (adj | shape_adj).sum(axis=1))


# TODO: torchfy
# TODO: fix logic
def simple_shape_score(pos, edge_set, k=3):
    G = nx.Graph(edge_set.tolist())
    apsp = dict(nx.all_pairs_shortest_path_length(G))
    full_edges, attr_d = zip(*[((u, v), d) for u in apsp for v, d in apsp[u].items() if u != v])
    full_index, d = np.array(full_edges).T, np.array(attr_d)
    src, dst = pos[full_index[0]], pos[full_index[1]]
    u = np.linalg.norm(dst - src, axis=1)
    d_mat = sparse.coo_matrix((d, full_index)).toarray()  # TODO: specify shape
    u_mat = sparse.coo_matrix((u, full_index)).toarray()
    kth_dist = u_mat[np.indices(d_mat.shape)[0], d_mat.argsort(axis=1)][:, :k+1].max(axis=1)
    return np.mean(k / (np.sum(u_mat <= kth_dist[:, None], axis=1) - 1))
