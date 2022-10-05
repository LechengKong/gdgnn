import numpy as np

from gnnfree.utils.utils import *

from graph_tool.all import shortest_distance, random_shortest_path, all_predecessors

def single_source_neg_sample(adj, source):
    nei = np.asarray(adj[source].todense()).flatten()
    can = np.logical_not(nei)
    can[source] = 0
    can = np.nonzero(can)[0]
    return can


def gt_single_source_random_path(gt_graph, reach_dist, source, can):
    timer = SmartTimer(False)
    timer.record()
    dist_map, pred_map, visited = shortest_distance(gt_graph, source, max_dist=reach_dist, pred_map=True, return_reached=True)
    timer.cal_and_update('sp')
    all_preds_map = all_predecessors(gt_graph, dist_map, pred_map)
    timer.cal_and_update('allp')
    gd = []
    length = []
    for c in can:
        p = random_shortest_path(gt_graph, source, c, dist_map, pred_map, all_preds_map)
        gd.append(np.array(p,dtype=int))
        length.append(len(p))
    timer.cal_and_update('gg')
    return gd, length

def get_single_source_shortest_path_from_pred(pred, can, dist):
    if len(dist)==0:
        return np.array([]), np.array([])
    ptr = can
    path_len = dist+1
    path_start_ind = np.r_[0, path_len]
    path_start_ind = np.cumsum(path_start_ind)[:-1]
    path_end_ind = np.cumsum(path_len)
    path_arr = np.zeros(path_end_ind[-1],dtype=int)
    active_paths = path_start_ind<path_end_ind
    while active_paths.sum()>0:
        cur_ptr = ptr[active_paths]
        cur_start_ind = path_start_ind[active_paths]
        path_arr[cur_start_ind] = cur_ptr

        path_start_ind[active_paths]+=1
        ptr[active_paths] = pred[cur_ptr]
        active_paths = path_start_ind<path_end_ind

    return path_arr, path_len
        

def get_single_source_sp(g, source, can, max_dist):
    source_dist, source_pred = shortest_distance(g, source, pred_map=True, max_dist=max_dist)
    source_dist = source_dist.a
    source_pred = source_pred.a
    can_dist = source_dist[can]
    neighbor_can = can_dist<=max_dist
    can_dist[np.logical_not(neighbor_can)] = max_dist+2

    can_path, can_len = get_single_source_shortest_path_from_pred(source_pred, can[neighbor_can], can_dist[neighbor_can])
    full_path_len = np.zeros_like(can)
    full_path_len[neighbor_can] = can_len
    return can_path, full_path_len, can_dist

def get_single_source_ver_gd_far(adj_mat, can, dist):
    if len(can)==0:
        return np.array([]), np.array([])
    can_arr, gd_can_arr = adj_mat[can].nonzero()

    can_dist, gd_can_dist = dist[can[can_arr]], dist[gd_can_arr]

    valid_gd_ind = (can_dist-gd_can_dist)==1

    can_arr = can_arr[valid_gd_ind]
    gd_arr = gd_can_arr[valid_gd_ind]
    gd_len = np.bincount(can_arr, minlength=len(can))

    # gd_deg = get_gd_deg_flat_batch(gd_arr, gd_len, adj_mat)
    return gd_arr, gd_len


def get_gd_deg_flat_batch(gd, gd_len, adj_mat):
    timer = SmartTimer(False)
    col_split_data = gd[var_size_repeat(len(gd), gd_len, gd_len)]

    repeat_count = np.repeat(gd_len, gd_len)
    row_split_data = np.repeat(gd, repeat_count)
    timer.cal_and_update('prepare')

    neighbor_indicator = np.asarray(adj_mat[row_split_data, col_split_data]).flatten()
    timer.cal_and_update('obt')

    reduce_ind = np.insert(repeat_count, 0, 0)
    reduce_ind = np.cumsum(reduce_ind)[:-1]

    gd_deg = np.add.reduceat(neighbor_indicator, reduce_ind).astype(int)
    timer.cal_and_update('compute')
    return gd_deg


def get_single_source_ver_gd_close(graph, adj_mat, source, can, dist, max_dist):
    timer = SmartTimer(False)
    if len(can)==0:
        return np.array([]), np.array([])
    source_nei = np.asarray(adj_mat[source].nonzero()[1]).flatten()
    timer.cal_and_update('nei')
    dist_col = []
    for v in source_nei:
        v_dist = shortest_distance(graph, v, max_dist=max_dist-1).a
        dist_col.append(v_dist)
    timer.cal_and_update('dist')
    nei_dist = np.stack(dist_col)

    can_dist, gd_can_dist = dist[can], nei_dist[:,can]
    gd_count, gd_arr = np.nonzero(((gd_can_dist-can_dist)==-1).T)
    gd_arr = source_nei[gd_arr]
    gd_len = np.bincount(gd_count, minlength=len(can))
    timer.cal_and_update('gd')

    # gd_deg = get_gd_deg_flat_batch(gd_arr, gd_len, adj_mat)
    timer.cal_and_update('bp')
    return gd_arr, gd_len

def get_single_source_ver_gd(g, adj_mat, source, can, max_dist):
    dist_map = shortest_distance(g, source, max_dist=max_dist)
    dist = dist_map.a
    can_dist = dist[can]
    neighbor_can = can_dist<=max_dist
    can_dist[np.logical_not(neighbor_can)] = max_dist+2
    far_gd_arr, far_gd_len = get_single_source_ver_gd_far(adj_mat, can[neighbor_can], dist)
    close_gd_arr, close_gd_len = get_single_source_ver_gd_close(g, adj_mat, source, can[neighbor_can], dist, max_dist)
    full_far_gd_len = np.zeros_like(can)
    full_far_gd_len[neighbor_can] = far_gd_len
    full_close_gd_len = np.zeros_like(can)
    full_close_gd_len[neighbor_can] = close_gd_len
    return far_gd_arr, full_far_gd_len, close_gd_arr, full_close_gd_len, can_dist


def get_vert_gd(graph, source_dist, node, dist):
    neighbor = graph.get_in_neighbors(node)
    neighbor_dist = source_dist[neighbor]
    gd = neighbor[neighbor_dist==(dist-1)]
    gd_len = np.array([len(gd)])
    return gd, gd_len

def get_gd_deg(adj, gd):
    t = SmartTimer(True)
    t.record()
    grid = np.meshgrid(gd,gd)
    t.cal_and_update('grid')
    adj = np.asarray(adj[grid[0], grid[1]].todense())
    t.cal_and_update('cal')
    return adj.sum(-1)

def get_gd_deg_dgl(g, gd):
    grid_row, grid_col = np.meshgrid(gd,gd)
    t = g.has_edges_between(grid_row.reshape(-1), grid_col.reshape(-1))
    t = t.view(len(grid_row),len(grid_row))
    deg = t.sum(-1)
    return deg.numpy()

def get_pair_wise_vert_gd(g, adj, head, tail, max_dist):
    head_dist_map, head_pred_map = shortest_distance(g, head, max_dist=max_dist, pred_map=True)
    head_gd = np.array([])
    tail_gd = np.array([])
    head_gd_len = np.array([0])
    tail_gd_len = np.array([0])
    head_gd_deg = np.array([])
    tail_gd_deg = np.array([])
    if head_dist_map[tail] > max_dist:
        dist = np.array([max_dist+2])
        ret = dist, head_gd, tail_gd, head_gd_len, tail_gd_len, head_gd_deg, tail_gd_deg
    elif head_dist_map[tail]==0:
        dist = np.array([0])
        ret = dist, head_gd, tail_gd, head_gd_len, tail_gd_len, head_gd_deg, tail_gd_deg
    else:
        dist = head_dist_map[tail]

        tail_gd, tail_gd_len = get_vert_gd(g, head_dist_map.a, tail, dist)
        tail_gd_deg = get_gd_deg(adj, tail_gd)

        tail_dist_map, tail_pred_map = shortest_distance(g, tail, max_dist=max_dist, pred_map=True)

        head_gd, head_gd_len = get_vert_gd(g, tail_dist_map.a, head, dist)
        head_gd_deg = get_gd_deg(adj, head_gd)

        ret = np.array([dist]), head_gd, tail_gd, head_gd_len, tail_gd_len, head_gd_deg, tail_gd_deg
    return ret