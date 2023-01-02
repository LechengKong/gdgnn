import random
import numpy as np

from torch.utils.data import Dataset
from collate import (
    collate_hor_link_kg,
    collate_ver_link_kg,
    collate_hor_link,
    collate_ver_link,
    collate_graph_hg,
    collate_node_hg,
    collate_tuple_g,
)
from graph_tool.all import (
    shortest_distance,
    adjacency,
    shortest_path,
)

from graph_utils import (
    single_source_neg_sample,
    get_single_source_sp,
    get_single_source_ver_gd,
    get_hor_gd_hop_map,
    get_ver_gd_hop_map,
    get_gd_deg_dgl,
    get_gd_deg_flat_batch,
    get_single_source_ver_gd_far,
    get_pair_wise_vert_gd,
)
from gnnfree.utils.datasets import DatasetWithCollate, SingleGraphDataset
from gnnfree.utils.utils import SmartTimer
from gnnfree.utils.graph import (
    remove_gt_graph_edge,
    add_gt_graph_edge,
    shortest_dist_sparse_mult,
)


class KGNegSampleDataset(SingleGraphDataset):
    """Dataset for Knowledge Graph Completion/Link Prediction"""

    def __init__(
        self,
        graph,
        edges,
        params,
        adj_list,
        reverse_dir_adj=None,
        mode="train",
        neg_link_per_sample=1,
    ):
        """Constructor for KGNegSampleDataset

        Arguments:
            graph {DGLGraph} -- A constructed single knowledge graph for link
            prediction, will be converted to graph_tool graph and adj matrix.

            edges {numpy.array} -- Target edge for prediction/training. Should
            be a N*3 triplet array with {head,relation, tail} format.

            params {Object} -- parameter specification

            adj_list {list[scipy.csr_matrix]} -- A list of adjacency matrix,
            each list corresponds to one relation.

        Keyword Arguments:
            reverse_dir_adj {list[scipy.csr_matrix]} -- used only when a
            a csc_matrix format adj matrix is need. (default: {None})

            mode {str} -- Determine the data generation mode. If 'train' the
            target link will be removed before any operation is used to
            generate link-related information. (default: {'train'})

            neg_link_per_sample {int} -- Number of negative link generated for
            each target link. (default: {1})
        """
        super().__init__(graph)
        self.mode = mode
        self.edges = edges
        self.adj_list = adj_list
        if reverse_dir_adj:
            self.reverse_dir_adj = reverse_dir_adj
        else:
            self.reverse_dir_adj = [adj.tocsc().T for adj in self.adj_list]
        self.num_edges = len(self.edges)
        self.num_rels = len(adj_list)
        self.params = params
        self.num_neg_sample = neg_link_per_sample

        self.timer = SmartTimer(False)

        self.__getitem__(0)

    def __len__(self):
        return self.num_edges

    def sample_links(self, head, rel, tail):
        """Sample negative heads and tails from the relation-specific adj mat
        negative tail also contains positive tail, so the concatenated
        representation is [positive link, neg tail link...,neg_head_link...]

        Returns:
            neg_tail -- [true_tail, neg_tail]
            neg_head -- [neg_head]
        """
        neg_tail_can = single_source_neg_sample(self.adj_list[rel], head)
        neg_head_can = single_source_neg_sample(
            self.reverse_dir_adj[rel], tail
        )

        neg_tail_count = int(
            (len(neg_tail_can) / (len(neg_tail_can) + len(neg_head_can)))
            * self.num_neg_sample
        )
        neg_head_count = self.num_neg_sample - neg_tail_count

        neg_tail = np.random.choice(
            neg_tail_can, neg_tail_count, replace=False
        )
        neg_head = np.random.choice(
            neg_head_can, neg_head_count, replace=False
        )

        neg_tail = np.r_[tail, neg_tail]
        return neg_tail, neg_head


class KGHorGDNegSampleDataset(KGNegSampleDataset):
    """Horizontal Geodesic (HorGD) Dataset for Knowledge graph"""

    def __getitem__(self, index):
        self.timer.record()
        head, rel, tail = self.edges[index]
        neg_tail, neg_head = self.sample_links(head, rel, tail)

        _, _, edges = self.graph.edge_ids(head, tail, return_uv=True)

        # If mode is train and the only edge between head and tail is the
        # target edge, we remove the edge from the HorGD generation.
        if self.mode == "train" and len(edges) == 1:
            remove_gt_graph_edge(self.gt_g, head, tail)

        # Generate single source shortest path for head and tail.

        neg_tail_gd, neg_tail_gd_len, neg_tail_dist = get_single_source_sp(
            self.gt_g, head, neg_tail, self.params.reach_dist
        )
        neg_head_gd, neg_head_gd_len, neg_head_dist = get_single_source_sp(
            self.gt_g, tail, neg_head, self.params.reach_dist
        )

        # Collect data into the final format.
        # [positive link data, neg tail data, neg head data]

        gd_arr = np.concatenate([neg_tail_gd, neg_head_gd])
        gd_len = np.concatenate([neg_tail_gd_len, neg_head_gd_len])
        dist = np.concatenate([neg_tail_dist, neg_head_dist])

        head_arr = np.concatenate(
            [np.array([head]).repeat(len(neg_tail)), neg_head]
        )
        tail_arr = np.concatenate(
            [neg_tail, np.array([tail]).repeat(len(neg_head))]
        )
        rel_arr = np.array([rel]).repeat(len(head_arr))

        self.timer.cal_and_update("gd")
        # Add back the removed link, if in train mode.
        if self.mode == "train" and len(edges) == 1:
            add_gt_graph_edge(self.gt_g, head, tail)
        ret = (
            head_arr,
            tail_arr,
            rel_arr,
            dist,
            gd_arr,
            gd_len,
            np.array([index, index + self.num_edges]),
            np.array([len(head_arr)]),
        )
        # print(ret)
        return ret

    def get_collate_fn(self):
        return collate_hor_link_kg


class KGVerGDNegSampleDataset(KGNegSampleDataset):
    """Vertical Geodesic (VerGD) Dataset for Knowledge graph"""

    def __getitem__(self, index):
        self.timer.record()
        head, rel, tail = self.edges[index]
        neg_tail, neg_head = self.sample_links(head, rel, tail)
        self.timer.cal_and_update("sample")

        _, _, edges = self.graph.edge_ids(head, tail, return_uv=True)

        if self.mode == "train" and len(edges) == 1:
            remove_gt_graph_edge(self.gt_g, head, tail)
        adj = adjacency(self.gt_g)
        self.timer.cal_and_update("update")

        # Get the single sourced vertical geodesics.
        # VerGD are returned in compact/block format:
        # gd_arr=[2, 3, 1, 4 | 1, 2] gd_len=[4, 2]
        # gd_arr is a list node indices representing VerGDs
        # gd_len record the length of each VerGD in order
        # This saves memory and helps with simplistic implementation.

        # The returned values are: head_gd_arr, head_gd_len, tail_gd_arr,
        # tail_gd_len, distances

        head_gd = get_single_source_ver_gd(
            self.gt_g, adj, head, neg_tail, self.params.reach_dist
        )
        tail_gd = get_single_source_ver_gd(
            self.gt_g, adj, tail, neg_head, self.params.reach_dist
        )
        self.timer.cal_and_update("vergd")

        head_close_gd_arr = np.concatenate([head_gd[2], tail_gd[0]])
        tail_close_gd_arr = np.concatenate([head_gd[0], tail_gd[2]])
        head_close_gd_len = np.concatenate([head_gd[3], tail_gd[1]])
        tail_close_gd_len = np.concatenate([head_gd[1], tail_gd[3]])

        dist = np.concatenate([head_gd[4], tail_gd[4]])

        head_arr = np.concatenate(
            [np.array([head]).repeat(len(neg_tail)), neg_head]
        )
        tail_arr = np.concatenate(
            [neg_tail, np.array([tail]).repeat(len(neg_head))]
        )
        rel_arr = np.array([rel]).repeat(len(head_arr))
        self.timer.cal_and_update("prepare")

        self.timer.cal_and_update("gd")
        if self.mode == "train" and len(edges) == 1:
            add_gt_graph_edge(self.gt_g, head, tail)
        ret = (
            head_arr,
            tail_arr,
            rel_arr,
            dist,
            head_close_gd_arr,
            tail_close_gd_arr,
            head_close_gd_len,
            tail_close_gd_len,
            np.array([index, index + self.num_edges]),
            np.array([len(head_arr)]),
        )
        # print(ret)
        return ret

    def get_collate_fn(self):
        return collate_ver_link_kg


class KGFilteredDataset(SingleGraphDataset):
    """Dataset for knowledge graph link prediction in filtered protocol"""

    def __init__(
        self, graph, edges, params, adj_list, mode="train", head_first=True
    ):
        """Very similar to KGNeg datasets, when head_first=True, we sample
        filtred negative tails w.r.t head, and vice versa.
        """
        super().__init__(graph)
        self.mode = mode
        self.edges = edges
        self.adj_list = adj_list
        self.num_edges = len(self.edges)
        self.num_rels = len(adj_list)
        self.params = params
        self.head_first = head_first

        self.timer = SmartTimer(False)

        self.__getitem__(0)

    def __len__(self):
        return self.num_edges

    def sample_links(self, head, rel, tail):
        neg_can = single_source_neg_sample(self.adj_list[rel], head)
        neg_can = np.r_[tail, neg_can]
        return neg_can


class KGHorGDFileredDataset(KGFilteredDataset):
    def __getitem__(self, index):
        if self.head_first:
            head, rel, tail = self.edges[index]
        else:
            tail, rel, head = self.edges[index]

        neg_tail = self.sample_links(head, rel, tail)
        _, _, edges = self.graph.edge_ids(head, tail, return_uv=True)

        if self.mode == "train" and len(edges) == 1:
            remove_gt_graph_edge(self.gt_g, head, tail)

        neg_tail_gd, neg_tail_gd_len, neg_tail_dist = get_single_source_sp(
            self.gt_g, head, neg_tail, self.params.reach_dist
        )

        head_arr = np.array([head]).repeat(len(neg_tail))
        rel_arr = np.array([rel]).repeat(len(neg_tail))

        if self.mode == "train" and len(edges) == 1:
            add_gt_graph_edge(self.gt_g, head, tail)

        # If positive tail, revert the head and tail, since subseqeunt pooling
        # is permutatation invariant, HorGD order does not matter
        if not self.head_first:
            temp = neg_tail
            neg_tail = head_arr
            head_arr = temp
        ret = (
            head_arr,
            neg_tail,
            rel_arr,
            neg_tail_dist,
            neg_tail_gd,
            neg_tail_gd_len,
            np.array([index, index + self.num_edges]),
            np.array([len(head_arr)]),
        )
        return ret

    def get_collate_fn(self):
        return collate_hor_link_kg


class KGVerGDFileredDataset(KGFilteredDataset):
    def __getitem__(self, index):
        self.timer.record()
        if self.head_first:
            head, rel, tail = self.edges[index]
        else:
            tail, rel, head = self.edges[index]

        neg_tail = self.sample_links(head, rel, tail)
        _, _, edges = self.graph.edge_ids(head, tail, return_uv=True)
        self.timer.cal_and_update("sample")

        if self.mode == "train" and len(edges) == 1:
            remove_gt_graph_edge(self.gt_g, head, tail)

        adj = adjacency(self.gt_g)
        self.timer.cal_and_update("graph")
        head_gd = get_single_source_ver_gd(
            self.gt_g, adj, head, neg_tail, self.params.reach_dist
        )
        self.timer.cal_and_update("gd")

        head_arr = np.array([head]).repeat(len(neg_tail))
        rel_arr = np.array([rel]).repeat(len(neg_tail))

        if self.mode == "train" and len(edges) == 1:
            add_gt_graph_edge(self.gt_g, head, tail)
        if not self.head_first:
            temp = neg_tail
            neg_tail = head_arr
            head_arr = temp
            head_close_gd_arr = head_gd[0]
            tail_close_gd_arr = head_gd[2]
            head_close_gd_len = head_gd[1]
            tail_close_gd_len = head_gd[3]
        else:
            head_close_gd_arr = head_gd[2]
            tail_close_gd_arr = head_gd[0]
            head_close_gd_len = head_gd[3]
            tail_close_gd_len = head_gd[1]
        self.timer.cal_and_update("collect")
        ret = (
            head_arr,
            neg_tail,
            rel_arr,
            head_gd[4],
            head_close_gd_arr,
            tail_close_gd_arr,
            head_close_gd_len,
            tail_close_gd_len,
            np.array([index, index + self.num_edges]),
            np.array([len(head_arr)]),
        )
        return ret

    def get_collate_fn(self):
        return collate_ver_link_kg


class HGHorGDDataset(SingleGraphDataset):
    """Dataset for homogeneous link prediction"""

    def __init__(
        self, graph, edges, labels, num_entities, params, mode="train"
    ):
        super().__init__(graph)
        self.mode = mode
        self.params = params
        self.edges = edges
        self.labels = labels
        # num_edges is the number of total edges in the graph, since we
        # use inverted edges, the actual number of edges is pos_edge_count
        # edge i+self.pos_edge_count is the corresponding inverted edge of
        # edge i.
        self.num_edges = len(edges)
        self.pos_edge_count = int(self.num_edges / 2)
        self.timer = SmartTimer(False)
        self.__getitem__(0)

    def __len__(self):
        return self.num_edges

    def __getitem__(self, index):
        self.timer.record()
        edge = self.edges[index]
        head, rel, tail = edge
        label = self.labels[index]
        self.timer.record()
        if self.mode == "train" and label == 1:
            remove_gt_graph_edge(self.gt_g, head, tail)
            masked_edge = [index, index + self.pos_edge_count]
        else:
            masked_edge = []
        self.timer.cal_and_update("pre")
        head_dist, head_pred = shortest_distance(
            self.gt_g, head, pred_map=True, max_dist=self.params.reach_dist
        )
        self.timer.cal_and_update("sp")
        sp_path, _ = shortest_path(self.gt_g, head, tail, pred_map=head_pred)
        sp_path = np.array(sp_path, dtype=int)
        self.timer.cal_and_update("random_p")

        gd_length = np.array([len(sp_path)])
        dist = gd_length - 1
        # Set unreached nodes with a distance of max_distance+2
        dist[dist < 0] = self.params.reach_dist + 2

        gd = sp_path
        self.timer.cal_and_update("gdgg")

        ret = (
            np.array([head]),
            np.array([tail]),
            dist,
            gd,
            gd_length,
            np.array(masked_edge),
            np.array([label]),
        )
        if self.mode == "train" and label == 1:
            add_gt_graph_edge(self.gt_g, head, tail)
        # print(link_arr)
        # print(dist_arr)
        # print(geodesics)
        return ret

    def get_collate_fn(self):
        return collate_hor_link


class HGHorGDInterDataset(SingleGraphDataset):
    """Dataset for homogeneous link prediction. Intersection implementation.
    Increase efficiency in large graphs by only computing ceil(max_dist/2) BFS
    for both nodes in the link."""

    def __init__(
        self, graph, edges, labels, num_entities, params, mode="train"
    ):
        super().__init__(graph)
        self.mode = mode
        self.params = params
        self.edges = edges
        self.labels = labels
        self.num_edges = len(edges)
        self.pos_edge_count = int(self.num_edges / 2)
        self.timer = SmartTimer(False)
        self.__getitem__(0)

    def __len__(self):
        return self.num_edges

    def __getitem__(self, index):
        self.timer.record()
        edge = self.edges[index]
        head, rel, tail = edge
        label = self.labels[index]
        self.timer.record()
        if self.mode == "train" and label == 1:
            remove_edge = True
            masked_edge = [index, index + self.pos_edge_count]
        else:
            remove_edge = False
            masked_edge = []
        self.timer.cal_and_update("pre")
        # Return the shortest path and the distance.
        rand_path, dist = get_hor_gd_hop_map(
            self.adj_mat, self.params.reach_dist, head, tail, remove_edge
        )
        self.timer.cal_and_update("sp")

        gd_length = np.array([len(rand_path)])
        dist[dist < 0] = self.params.reach_dist + 2

        gd = rand_path
        self.timer.cal_and_update("gdgg")

        ret = (
            np.array([head]),
            np.array([tail]),
            dist,
            gd,
            gd_length,
            np.array(masked_edge),
            np.array([label]),
        )
        # print(ret)
        return ret

    def get_collate_fn(self):
        return collate_hor_link


class HGVerGDDataset(SingleGraphDataset):
    """Dataset for vertical geodesics on homogeneous link pred"""

    def __init__(
        self,
        graph,
        edges,
        labels,
        num_entities,
        params,
        mode="train",
        hop_sampling=False,
    ):
        super().__init__(graph)
        self.mode = mode
        self.params = params
        self.edges = edges
        self.labels = labels
        self.num_edges = len(edges)
        self.pos_edge_count = int(self.num_edges / 2)
        self.hop_sampling = hop_sampling
        self.timer = SmartTimer(False)
        self.__getitem__(0)

    def __len__(self):
        return self.num_edges

    def __getitem__(self, index):
        self.timer.record()
        edge = self.edges[index]
        head, rel, tail = edge
        label = self.labels[index]

        self.timer.record()
        if self.mode == "train" and label == 1:
            remove_gt_graph_edge(self.gt_g, head, tail)
            masked_edge = [index, index + self.pos_edge_count]
        else:
            masked_edge = []
        self.timer.cal_and_update("remove")
        # Return the gd, length of GD, and the degree of each GD node in
        # the induced subgraph of VerGD

        (
            dist,
            head_gd,
            tail_gd,
            head_gd_len,
            tail_gd_len,
            head_gd_deg,
            tail_gd_deg,
        ) = get_pair_wise_vert_gd(
            self.gt_g, self.adj_mat, head, tail, self.params.reach_dist
        )
        # print(ret)
        if self.mode == "train" and label == 1:
            add_gt_graph_edge(self.gt_g, head, tail)
        self.timer.cal_and_update("adde")
        # print(link_arr)
        # print(dist_arr)
        # print(geodesics)
        return (
            np.array([head]),
            np.array([tail]),
            dist,
            head_gd.astype(int),
            tail_gd.astype(int),
            head_gd_len,
            tail_gd_len,
            head_gd_deg,
            tail_gd_deg,
            np.array(masked_edge),
            np.array([label]),
        )

    def get_collate_fn(self):
        return collate_ver_link


class HGVerGDInterDataset(SingleGraphDataset):
    """Dataset for homogeneous link prediction. Intersection implementation.
    Increase efficiency in large graphs by only computing ceil(max_dist/2) BFS
    for both nodes in the link."""

    def __init__(
        self,
        graph,
        edges,
        labels,
        num_entities,
        params,
        mode="train",
        hop_sampling=False,
    ):
        super().__init__(graph)
        self.mode = mode
        self.params = params
        self.edges = edges
        self.labels = labels
        self.num_edges = len(edges)
        self.pos_edge_count = int(self.num_edges / 2)
        self.hop_sampling = hop_sampling
        self.timer = SmartTimer(False)
        self.__getitem__(0)

    def __len__(self):
        return self.num_edges

    def __getitem__(self, index):
        edge = self.edges[index]
        head, rel, tail = edge
        label = self.labels[index]

        self.timer.record()
        if self.mode == "train" and label == 1:
            remove_edge = True
            masked_edge = [index, index + self.pos_edge_count]
        else:
            remove_edge = False
            masked_edge = []
        self.timer.cal_and_update("remove")

        head_gd, tail_gd, dist = get_ver_gd_hop_map(
            self.adj_mat, self.params.reach_dist, head, tail, remove_edge
        )
        self.timer.cal_and_update("gd")

        dist[dist < 0] = self.params.reach_dist + 2

        head_gd_len = np.array([len(head_gd)])
        tail_gd_len = np.array([len(tail_gd)])
        self.timer.cal_and_update("prepare")

        # Use dgl to generate vertical gd degree efficiently
        head_gd_deg = get_gd_deg_dgl(self.graph, head_gd)
        tail_gd_deg = get_gd_deg_dgl(self.graph, tail_gd)
        self.timer.cal_and_update("adddeg")
        # print(link_arr)
        # print(dist_arr)
        # print(geodesics)
        ret = (
            np.array([head]),
            np.array([tail]),
            dist,
            head_gd.astype(int),
            tail_gd.astype(int),
            head_gd_len,
            tail_gd_len,
            head_gd_deg,
            tail_gd_deg,
            np.array(masked_edge),
            np.array([label]),
        )
        # print(ret)
        return ret

    def get_collate_fn(self):
        return collate_ver_link


class HGVerGDGraphDataset(DatasetWithCollate):
    """Graph-level vertical geodesics Dataset
    The return values are:
    graph: A DGLGraph

    neighbors_ret, neighbor_count_ret: a sparse representation of
    each node's neighbors. neighbors_ret is a indices vector,
    neighbor_count_ret is a 1*N count vector, where N=number of
    nodes in the graph. Neighbors of node k is
            neighbors_ret[sum(neighbor_count_ret[:k]):
                            sum(neighbor_count_ret[:k+1])]
    This format help with sparse operation in the model.

    dist_ret: same shape as neighbors_ret, dist_ret[i] is the
    distance between neighbors_ret[i] and its center node.

    gd_ret, gd_count_ret: a sparse representation of neighbors'
    vertical geodesics similar to that of neighbors.
    gd_count_ret has the shape of 1*len(neighbors_ret).

    gd_deg: the vertical geodesics degree in the induced
    subgraphs, has the same shape as gd_ret.
    """

    def __init__(self, graphs, labels, params):
        super().__init__()
        self.graphs = graphs
        self.params = params
        self.num_graphs = len(self.graphs)
        self.labels = labels
        self.timer = SmartTimer(False)

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, index):
        self.timer.record()
        graph = self.graphs[index]
        adj = graph.adjacency_matrix(transpose=False, scipy_fmt="csr")
        num_nodes = graph.num_nodes()

        self.timer.cal_and_update("sht")
        # Compute all pair shortest distance up to the max distance
        distance_mat = shortest_dist_sparse_mult(adj, self.params.reach_dist)
        self.timer.cal_and_update("dist")

        # Block self distance
        distance_mat[np.arange(num_nodes), np.arange(num_nodes)] = (
            self.params.reach_dist + 2
        )
        self.timer.cal_and_update("block_dist")

        # Obtain list of reachable nodes in pair wise format
        center_nodes, reached_nodes = (
            distance_mat <= self.params.reach_dist
        ).nonzero()
        self.timer.cal_and_update("reachable nodes")
        # Edge case: no edge in the graph.
        if len(reached_nodes) == 0:
            ret = (
                graph,
                np.array([]),
                np.array([0] * num_nodes),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([self.labels[index]]),
            )
            self.timer.cal_and_update("fulltime")
            return ret
        # Recover self distance
        distance_mat[np.arange(num_nodes), np.arange(num_nodes)] = 0
        self.timer.cal_and_update("unblock dist")
        # Number of reached nodes, if a node is isolated its entry
        # is 0.
        num_reached = np.bincount(center_nodes, minlength=num_nodes)
        self.timer.cal_and_update("reach count")
        # Reached nodes distance
        reached_dist = distance_mat[center_nodes, reached_nodes]
        self.timer.cal_and_update("reached dist")
        # Reached nodes neighbors
        reached_nodes_multi, reached_nodes_neighbor = adj[
            reached_nodes
        ].nonzero()
        self.timer.cal_and_update("reached neighbor")

        # Dist from center nodes to reached nodes repeated
        # Broadcast the reached nodes distance to the shape of reached
        # nodes' neighbors.
        reached_dist_multi = reached_dist[reached_nodes_multi]
        self.timer.cal_and_update("reached dist multi")

        # Center nodes index repeated/broadcasted
        center_nodes_multi = center_nodes[reached_nodes_multi]
        self.timer.cal_and_update("centered node multi")

        # Distance between center nodes and reached nodes neighbors
        center_neighbor_dist = distance_mat[
            center_nodes_multi, reached_nodes_neighbor
        ]
        self.timer.cal_and_update("center neighbor dist")

        # Dist(Reached nodes, Center node)==Dist(Reached nodes neighbor
        # , Center node)+1
        # i.e. compute vertical geodesics.
        gd_nodes = reached_dist_multi - center_neighbor_dist == 1
        self.timer.cal_and_update("goodnei")
        # Compute number of good neighbor nodes that are vertical geodesics
        good_nei_ct = reached_nodes_multi[gd_nodes]
        gd_c = np.bincount(good_nei_ct, minlength=len(reached_nodes))
        # Vertical GD nodes.
        good_nei = reached_nodes_neighbor[gd_nodes]

        self.timer.cal_and_update("gdcount")
        # Count vertical geodesics degrees.
        gd_deg = get_gd_deg_flat_batch(good_nei, gd_c, adj)
        self.timer.cal_and_update("obtain deg")

        neighbors_ret = reached_nodes
        neighbors_count_ret = num_reached
        gd_ret = good_nei
        gd_count_ret = gd_c
        dist_ret = reached_dist
        deg_ret = gd_deg

        ret = (
            graph,
            neighbors_ret,
            neighbors_count_ret,
            dist_ret,
            gd_ret,
            gd_count_ret,
            deg_ret,
            np.array([self.labels[index]]),
        )
        self.timer.cal_and_update("fulltime")
        return ret

    def get_collate_fn(self):
        return collate_graph_hg


class HGVerGDNodeDataset(SingleGraphDataset):
    """The node-level vertical geodesics dataset, similar to graph-level
    version
    The return values are:
    node: the node of interests

    neighbors, neighbor_count: neighbors of node, and the number of
    neighbors.

    neighbor_dist: same shape as neighbors, neighbor_dist[i] is the
    distance between neighbors[i] and node.

    gd_arr, gd_len: a sparse representation of neighbors'
    vertical geodesics.
    gd_len has the shape of 1*neighbor_count.

    gd_deg: the vertical geodesics degree in the induced
    subgraphs, has the same shape as gd_arr."""

    def __init__(self, graph, nodes, labels, params):
        super().__init__(graph)
        self.params = params
        self.nodes = nodes
        self.labels = labels
        self.num_nodes = len(nodes)
        self.timer = SmartTimer(False)
        self.__getitem__(0)

    def __len__(self):
        return self.num_nodes

    def __getitem__(self, index):
        self.timer.record()
        node = self.nodes[index]
        label = self.labels[index]

        self.timer.record()

        node_distance = shortest_distance(
            self.gt_g, source=node, max_dist=self.params.reach_dist
        )

        node_distance = node_distance.a
        node_distance[node] = self.params.reach_dist + 2

        neighbors = (node_distance <= self.params.reach_dist).nonzero()[0]
        neighbor_dist = node_distance[neighbors]

        neighbor_count = len(neighbors)
        if neighbor_count == 0:
            ret = (
                np.array([node]),
                np.array([]),
                np.array([]),
                np.array([0]),
                np.array([]),
                np.array([0]),
                np.array([self.labels[index].numpy()]),
            )
            self.timer.cal_and_update("fulltime")
            return ret

        node_distance[node] = 0

        gd_arr, gd_len = get_single_source_ver_gd_far(
            self.adj_mat, neighbors, node_distance
        )

        gd_deg = get_gd_deg_flat_batch(gd_arr, gd_len, self.adj_mat)

        ret = (
            np.array([node]),
            neighbors,
            neighbor_dist,
            np.array([neighbor_count]),
            gd_arr,
            gd_len,
            gd_deg,
            np.array([label]),
        )

        self.timer.cal_and_update("adde")
        # print(link_arr)
        # print(dist_arr)
        # print(geodesics)
        return ret

    def get_collate_fn(self):
        return collate_node_hg


class TupelDataset(DatasetWithCollate):
    def __init__(self, data) -> None:
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        ret = []
        for t in self.data:
            ret.append(t[index])
        return ret

    def get_collate_fn(self):
        return collate_tuple_g


class SimpleSampleClass(Dataset):
    def __init__(self, num_samples, adj, num_ett):
        self.num_samples = num_samples
        self.adj = adj
        self.num_ett = num_ett

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        head = random.randint(0, self.num_ett - 1)
        nei = self.adj[head]
        while True:
            tail = random.randint(0, self.num_ett - 1)
            if head != tail and nei[0, tail] == 0:
                break
        return np.array([head, tail])
