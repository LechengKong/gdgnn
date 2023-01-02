import numpy as np
from gnnfree.utils.graph import get_k_hop_neighbors

from gnnfree.utils.utils import SmartTimer, var_size_repeat

from graph_tool.all import (
    shortest_distance,
    random_shortest_path,
    all_predecessors,
)


def single_source_neg_sample(adj, source):
    """Return all nodes i s.t. adj[source, i] = 0"""
    nei = np.asarray(adj[source].todense()).flatten()
    can = np.logical_not(nei)
    can[source] = 0
    can = np.nonzero(can)[0]
    return can


def gt_single_source_random_path(gt_graph, reach_dist, source, can):
    """Return random path between source and each node in can up to
    distance reach_dist in the gt_graph

    Arguments:
        gt_graph {graph_tool.Graph} -- a graph_tool graph
        reach_dist {int} -- the max distance
        source {int} -- source node index
        can {np.array} -- an array of candidate nodes to find shortest
        paths.

    Returns:
        list[np.array] -- shortest paths.
        list[int] -- the length of each shortest path.
    """
    timer = SmartTimer(False)
    timer.record()
    dist_map, pred_map, visited = shortest_distance(
        gt_graph,
        source,
        max_dist=reach_dist,
        pred_map=True,
        return_reached=True,
    )
    timer.cal_and_update("sp")
    all_preds_map = all_predecessors(gt_graph, dist_map, pred_map)
    timer.cal_and_update("allp")
    gd = []
    length = []
    for c in can:
        p = random_shortest_path(
            gt_graph, source, c, dist_map, pred_map, all_preds_map
        )
        gd.append(np.array(p, dtype=int))
        length.append(len(p))
    timer.cal_and_update("gg")
    return gd, length


def get_single_source_shortest_path_from_pred(pred, can, dist):
    """Return the shortest path from the predecessor information

    Returns 2 vectors. The shorstest path between source and can[i]
            path_arr[cumsum(path_len[:i]):cumsum(path_len[:i+1])]

    Arguments:
        pred {np.array} -- pred[i] is the zero-padded predecessors
        of can[i], pred[i, 0] is the source node.
        can {np.array} -- an array of candidate
        dist {np.array} -- dist[i] is the distance between can[i]
        and its source.

    Returns:
        np.array -- compact/sparse representation of paths
        np.array -- length of each shortest path.
    """
    if len(dist) == 0:
        return np.array([]), np.array([])
    ptr = can
    path_len = dist + 1
    path_start_ind = np.r_[0, path_len]
    path_start_ind = np.cumsum(path_start_ind)[:-1]
    path_end_ind = np.cumsum(path_len)
    path_arr = np.zeros(path_end_ind[-1], dtype=int)
    active_paths = path_start_ind < path_end_ind
    while active_paths.sum() > 0:
        cur_ptr = ptr[active_paths]
        cur_start_ind = path_start_ind[active_paths]
        path_arr[cur_start_ind] = cur_ptr

        path_start_ind[active_paths] += 1
        ptr[active_paths] = pred[cur_ptr]
        active_paths = path_start_ind < path_end_ind

    return path_arr, path_len


def get_single_source_sp(g, source, can, max_dist):
    """Return shortest paths between source and each node in can
    up to max_dist

    can_path and full_path_len follows the format in
    get_single_source_shortest_path_from_pred

    Returns:
        np.array -- compact shortest path representation
        np.array -- shortest path length
        np.array -- distance between source and can
    """
    source_dist, source_pred = shortest_distance(
        g, source, pred_map=True, max_dist=max_dist
    )
    source_dist = source_dist.a
    source_pred = source_pred.a
    can_dist = source_dist[can]
    neighbor_can = can_dist <= max_dist
    can_dist[np.logical_not(neighbor_can)] = max_dist + 2

    can_path, can_len = get_single_source_shortest_path_from_pred(
        source_pred, can[neighbor_can], can_dist[neighbor_can]
    )
    full_path_len = np.zeros_like(can)
    full_path_len[neighbor_can] = can_len
    return can_path, full_path_len, can_dist


def get_single_source_ver_gd_far(adj_mat, can, dist):
    """Return the far side vertical GD between source and nodes in can
    source not need with dist computed. Refer to the paper for more
    description of far side vs close side.

    The return format follows the compact/sparse format.

    Arguments:
        adj_mat {scipy.csr_matrix} -- adjacency matrix of the graph
        can {np.array} -- list of candidate to compute vertical geodesics
        from source
        dist {np.array} -- distance between source and all nodes.

    Returns:
        far side vertical GD.
    """
    if len(can) == 0:
        return np.array([]), np.array([])
    can_arr, gd_can_arr = adj_mat[can].nonzero()

    can_dist, gd_can_dist = dist[can[can_arr]], dist[gd_can_arr]

    valid_gd_ind = (can_dist - gd_can_dist) == 1

    can_arr = can_arr[valid_gd_ind]
    gd_arr = gd_can_arr[valid_gd_ind]
    gd_len = np.bincount(can_arr, minlength=len(can))

    # gd_deg = get_gd_deg_flat_batch(gd_arr, gd_len, adj_mat)
    return gd_arr, gd_len


def get_gd_deg_flat_batch(gd, gd_len, adj_mat):
    """Retrun vertical geodesic degrees given VerGD and adj matrix

    Arguments:
        gd {np.array} -- VerGD
        gd_len {np.array} -- VerGD lengths
        adj_mat {scipy.csr_matrix} -- adj matrix of the graph

    Returns:
        np.array -- VerticalGD degrees, same shape as gd
    """
    timer = SmartTimer(False)
    col_split_data = gd[var_size_repeat(len(gd), gd_len, gd_len)]

    repeat_count = np.repeat(gd_len, gd_len)
    row_split_data = np.repeat(gd, repeat_count)
    timer.cal_and_update("prepare")

    neighbor_indicator = np.asarray(
        adj_mat[row_split_data, col_split_data]
    ).flatten()
    timer.cal_and_update("obt")

    reduce_ind = np.insert(repeat_count, 0, 0)
    reduce_ind = np.cumsum(reduce_ind)[:-1]

    gd_deg = np.add.reduceat(neighbor_indicator, reduce_ind).astype(int)
    timer.cal_and_update("compute")
    return gd_deg


def get_single_source_ver_gd_close(
    graph, adj_mat, source, can, dist, max_dist
):
    """Return the close side vertical GD between source and nodes in can.
    Refer to the paper for more description of far side vs close side.

    apply single source BFS on every neighbor of source.

    The return format follows the compact/sparse format.

    Arguments:
        graph {DGLGraph} -- DGLGraph
        adj_mat {scipy.csr_matrix} -- adjacency matrix of the graph
        source {int} -- source node
        can {np.array} -- list of candidate to compute vertical geodesics
        from source
        dist {np.array} -- distance between source and all nodes.
        max_dist {int} -- max dist to search

    Returns:
        close side vertical GD.
    """
    timer = SmartTimer(False)
    if len(can) == 0:
        return np.array([]), np.array([])
    source_nei = np.asarray(adj_mat[source].nonzero()[1]).flatten()
    timer.cal_and_update("nei")
    dist_col = []
    for v in source_nei:
        v_dist = shortest_distance(graph, v, max_dist=max_dist - 1).a
        dist_col.append(v_dist)
    timer.cal_and_update("dist")
    nei_dist = np.stack(dist_col)

    can_dist, gd_can_dist = dist[can], nei_dist[:, can]
    gd_count, gd_arr = np.nonzero(((gd_can_dist - can_dist) == -1).T)
    gd_arr = source_nei[gd_arr]
    gd_len = np.bincount(gd_count, minlength=len(can))
    timer.cal_and_update("gd")

    timer.cal_and_update("bp")
    return gd_arr, gd_len


def get_single_source_ver_gd(g, adj_mat, source, can, max_dist):
    """Compute vertical gd between source and all nodes in can"""
    dist_map = shortest_distance(g, source, max_dist=max_dist)
    dist = dist_map.a
    can_dist = dist[can]
    neighbor_can = can_dist <= max_dist
    can_dist[np.logical_not(neighbor_can)] = max_dist + 2
    far_gd_arr, far_gd_len = get_single_source_ver_gd_far(
        adj_mat, can[neighbor_can], dist
    )
    close_gd_arr, close_gd_len = get_single_source_ver_gd_close(
        g, adj_mat, source, can[neighbor_can], dist, max_dist
    )
    full_far_gd_len = np.zeros_like(can)
    full_far_gd_len[neighbor_can] = far_gd_len
    full_close_gd_len = np.zeros_like(can)
    full_close_gd_len[neighbor_can] = close_gd_len
    return (
        far_gd_arr,
        full_far_gd_len,
        close_gd_arr,
        full_close_gd_len,
        can_dist,
    )


def get_vert_gd(graph, source_dist, node, dist):
    neighbor = graph.get_in_neighbors(node)
    neighbor_dist = source_dist[neighbor]
    gd = neighbor[neighbor_dist == (dist - 1)]
    gd_len = np.array([len(gd)])
    return gd, gd_len


def get_gd_deg(adj, gd):
    t = SmartTimer(False)
    t.record()
    grid = np.meshgrid(gd, gd)
    t.cal_and_update("grid")
    adj = np.asarray(adj[grid[0], grid[1]].todense())
    t.cal_and_update("cal")
    return adj.sum(-1)


def get_gd_deg_dgl(g, gd):
    grid_row, grid_col = np.meshgrid(gd, gd)
    t = g.has_edges_between(grid_row.reshape(-1), grid_col.reshape(-1))
    t = t.view(len(grid_row), len(grid_row))
    deg = t.sum(-1)
    return deg.numpy()


def get_pair_wise_vert_gd(g, adj, head, tail, max_dist):
    """Return the vertical geodesics between head and tail"""
    head_dist_map, head_pred_map = shortest_distance(
        g, head, max_dist=max_dist, pred_map=True
    )
    head_gd = np.array([])
    tail_gd = np.array([])
    head_gd_len = np.array([0])
    tail_gd_len = np.array([0])
    head_gd_deg = np.array([])
    tail_gd_deg = np.array([])
    # Edge case: distance larger than max dist
    if head_dist_map[tail] > max_dist:
        dist = np.array([max_dist + 2])
        ret = (
            dist,
            head_gd,
            tail_gd,
            head_gd_len,
            tail_gd_len,
            head_gd_deg,
            tail_gd_deg,
        )
    # Edge case: head == tail
    elif head_dist_map[tail] == 0:
        dist = np.array([0])
        ret = (
            dist,
            head_gd,
            tail_gd,
            head_gd_len,
            tail_gd_len,
            head_gd_deg,
            tail_gd_deg,
        )
    else:
        dist = head_dist_map[tail]

        tail_gd, tail_gd_len = get_vert_gd(g, head_dist_map.a, tail, dist)
        tail_gd_deg = get_gd_deg(adj, tail_gd)

        tail_dist_map, tail_pred_map = shortest_distance(
            g, tail, max_dist=max_dist, pred_map=True
        )

        head_gd, head_gd_len = get_vert_gd(g, tail_dist_map.a, head, dist)
        head_gd_deg = get_gd_deg(adj, head_gd)

        ret = (
            np.array([dist]),
            head_gd,
            tail_gd,
            head_gd_len,
            tail_gd_len,
            head_gd_deg,
            tail_gd_deg,
        )
    return ret


def get_hor_gd_hop_map(adj_mat, dist, head, tail, remove_edge=False):
    """Return a shortest path between head and tail using hop intersection.
    Instead of doing BFS with depth=dist on head, we do BFS with
    depth=dist/2 on both head tail, so if head and tail have paths, they
    must be intersection at some hop, and we backtrace from the first
    intersected hop to find shortest path.

    Arguments:
        adj_mat {scipy.csr_matrix} -- adj matrix of the graph
        dist {int} -- max distance of the shortest path
        head {int} -- head index
        tail {int} -- tail inded

    Keyword Arguments:
        remove_edge {bool} -- whether remove the link between head and tail
         (default: {False})

    Returns:
        shortest path and path length
    """
    head_k_hop = get_k_hop_neighbors(
        adj_mat,
        head,
        int(dist / 2) + int(dist % 2),
        tail if remove_edge else None,
    )
    tail_k_hop = get_k_hop_neighbors(
        adj_mat, tail, int(dist / 2), head if remove_edge else None
    )
    no_connect_flag = True
    for i in range(1, dist + 1):
        head_hop = int(i / 2) + int(i % 2)
        tail_hop = int(i / 2)
        if head_hop in head_k_hop and tail_hop in tail_k_hop:
            hop_intersect = np.intersect1d(
                head_k_hop[head_hop], tail_k_hop[tail_hop]
            )
            if len(hop_intersect) > 0:
                no_connect_flag = False
                break
        else:
            break
    if no_connect_flag:
        return np.array([]), np.array([-1])

    mid_node = np.random.choice(hop_intersect)

    head_trace = backtrace_hop_mat(adj_mat, head_hop, head_k_hop, mid_node)

    tail_trace = backtrace_hop_mat(adj_mat, tail_hop, tail_k_hop, mid_node)

    return np.r_[np.flip(head_trace), mid_node, tail_trace], np.array([i])


def backtrace_hop_mat(adj_mat, hop, hop_map, last_node):
    """Given a random intersection node, trace the path."""
    trace_head = last_node
    trace = []
    for i in range(hop, 0, -1):
        last_hop_ind = i - 1
        last_hop_nodes = hop_map[last_hop_ind]
        nei = adj_mat[trace_head].nonzero()[1]
        can = np.intersect1d(last_hop_nodes, nei)
        trace_head = np.random.choice(can)
        trace.append(trace_head)

    return trace


def get_ver_gd_hop_map(adj_mat, dist, head, tail, remove_edge=False):
    """Return vertical geodesics between head and tail using hop intersection.
    Instead of doing BFS with depth=dist on head, we do BFS with
    depth=dist/2 on both head tail, so if head and tail have paths, they
    must be intersection at some hop, and we backtrace from the first
    intersected hop to find VerGD.

    Arguments:
        adj_mat {scipy.csr_matrix} -- adj matrix of the graph
        dist {int} -- max distance of the shortest path
        head {int} -- head index
        tail {int} -- tail inded

    Keyword Arguments:
        remove_edge {bool} -- whether remove the link between head and tail
         (default: {False})

    Returns:
        VerGD and distance
    """
    t = SmartTimer(False)
    t.record()
    head_k_hop = get_k_hop_neighbors(
        adj_mat,
        head,
        int(dist / 2) + int(dist % 2),
        tail if remove_edge else None,
    )
    t.cal_and_update("headk")
    tail_k_hop = get_k_hop_neighbors(
        adj_mat, tail, int(dist / 2), head if remove_edge else None
    )
    t.cal_and_update("tailk")
    no_connect_flag = True
    for i in range(1, dist + 1):
        head_hop = int(i / 2) + int(i % 2)
        tail_hop = int(i / 2)
        if head_hop in head_k_hop and tail_hop in tail_k_hop:
            hop_intersect = np.intersect1d(
                head_k_hop[head_hop], tail_k_hop[tail_hop]
            )
            if len(hop_intersect) > 0:
                no_connect_flag = False
                break
        else:
            break
    t.cal_and_update("inter")
    if no_connect_flag:
        return np.array([]), np.array([]), np.array([-1])

    if i == 1:
        return np.array([]), np.array([]), np.array([1])

    head_trace = backtrace_hop_mat_all_nodes(
        adj_mat, head_hop, head_k_hop, hop_intersect
    )

    tail_trace = backtrace_hop_mat_all_nodes(
        adj_mat, tail_hop, tail_k_hop, hop_intersect
    )
    t.cal_and_update("gd")

    return head_trace, tail_trace, np.array([i])


def backtrace_hop_mat_all_nodes(adj_mat, hop, hop_map, last_inter):
    trace_head = last_inter
    for i in range(hop, 1, -1):
        last_hop_ind = i - 1
        last_hop_nodes = hop_map[last_hop_ind]
        nei = adj_mat[trace_head].nonzero()[1]
        can = np.intersect1d(last_hop_nodes, nei)
        trace_head = can

    return trace_head
