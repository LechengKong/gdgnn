from torch.utils.data import Dataset
from collate import *
from graph_tool.all import Graph, shortest_distance, adjacency, random_shortest_path, shortest_path, GraphView

from graph_utils import *
from gnnfree.utils.datasets import DatasetWithCollate, SingleGraphDataset
from gnnfree.utils.utils import *
from gnnfree.utils.graph import *

class KGNegSampleDataset(SingleGraphDataset):
    def __init__(self, graph, edges, params, adj_list, reverse_dir_adj=None, mode='train', neg_link_per_sample=1):
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
        neg_tail_can = single_source_neg_sample(self.adj_list[rel], head)
        neg_head_can = single_source_neg_sample(self.reverse_dir_adj[rel], tail)

        neg_tail_count = int((len(neg_tail_can)/(len(neg_tail_can)+len(neg_head_can)))*self.num_neg_sample)
        neg_head_count = self.num_neg_sample-neg_tail_count

        neg_tail = np.random.choice(neg_tail_can, neg_tail_count, replace=False)
        neg_head = np.random.choice(neg_head_can, neg_head_count, replace=False)

        neg_tail = np.r_[tail, neg_tail]
        return neg_tail, neg_head


class KGHorGDNegSampleDataset(KGNegSampleDataset):
    def __getitem__(self, index):
        self.timer.record()
        head, rel, tail = self.edges[index]
        neg_tail, neg_head = self.sample_links(head, rel, tail)

        _,_,edges = self.graph.edge_ids(head,tail, return_uv=True)

        if self.mode=='train' and len(edges)==1:
            remove_gt_graph_edge(self.gt_g, head, tail)

        neg_tail_gd, neg_tail_gd_len, neg_tail_dist = get_single_source_sp(self.gt_g, head, neg_tail, self.params.reach_dist)
        neg_head_gd, neg_head_gd_len, neg_head_dist = get_single_source_sp(self.gt_g, tail, neg_head, self.params.reach_dist)
        
        gd_arr = np.concatenate([neg_tail_gd, neg_head_gd])
        gd_len = np.concatenate([neg_tail_gd_len, neg_head_gd_len])
        dist = np.concatenate([neg_tail_dist, neg_head_dist])

        head_arr = np.concatenate([np.array([head]).repeat(len(neg_tail)), neg_head])
        tail_arr = np.concatenate([neg_tail, np.array([tail]).repeat(len(neg_head))])
        rel_arr = np.array([rel]).repeat(len(head_arr))
        
        self.timer.cal_and_update('gd')
        if self.mode=='train' and len(edges)==1:
            add_gt_graph_edge(self.gt_g, head, tail)
        ret = head_arr, tail_arr, rel_arr, dist, gd_arr, gd_len, np.array([index, index+self.num_edges]), np.array([len(head_arr)])
        # print(ret)
        return ret

    def get_collate_fn(self):
        return collate_hor_link_kg

class KGVerGDNegSampleDataset(KGNegSampleDataset):
    def __getitem__(self, index):
        self.timer.record()
        head, rel, tail = self.edges[index]
        neg_tail, neg_head = self.sample_links(head, rel, tail)

        _,_,edges = self.graph.edge_ids(head,tail, return_uv=True)

        if self.mode=='train' and len(edges)==1:
            remove_gt_graph_edge(self.gt_g, head, tail)
        adj = adjacency(self.gt_g)

        head_gd = get_single_source_ver_gd(self.gt_g, adj, head, neg_tail, self.params.reach_dist)
        tail_gd = get_single_source_ver_gd(self.gt_g, adj, tail, neg_head, self.params.reach_dist)

        head_close_gd_arr = np.concatenate([head_gd[2], tail_gd[0]])
        tail_close_gd_arr = np.concatenate([head_gd[0], tail_gd[2]])
        head_close_gd_len = np.concatenate([head_gd[3], tail_gd[1]])
        tail_close_gd_len = np.concatenate([head_gd[1], tail_gd[3]])
        
        dist = np.concatenate([head_gd[4], tail_gd[4]])

        head_arr = np.concatenate([np.array([head]).repeat(len(neg_tail)), neg_head])
        tail_arr = np.concatenate([neg_tail, np.array([tail]).repeat(len(neg_head))])
        rel_arr = np.array([rel]).repeat(len(head_arr))
        
        self.timer.cal_and_update('gd')
        if self.mode=='train' and len(edges)==1:
            add_gt_graph_edge(self.gt_g, head, tail)
        ret = head_arr, tail_arr, rel_arr, dist, head_close_gd_arr, tail_close_gd_arr, head_close_gd_len, tail_close_gd_len, np.array([index, index+self.num_edges]), np.array([len(head_arr)])
        # print(ret)
        return ret

    def get_collate_fn(self):
        return collate_ver_link_kg

class KGFilteredDataset(SingleGraphDataset):
    def __init__(self, graph, edges, params, adj_list, mode='train', head_first=True):
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
        _,_,edges = self.graph.edge_ids(head,tail, return_uv=True)

        if self.mode=='train' and len(edges)==1:
            remove_gt_graph_edge(self.gt_g, head, tail)

        neg_tail_gd, neg_tail_gd_len, neg_tail_dist = get_single_source_sp(self.gt_g, head, neg_tail, self.params.reach_dist)

        head_arr = np.array([head]).repeat(len(neg_tail))
        rel_arr = np.array([rel]).repeat(len(neg_tail))

        if self.mode=='train' and len(edges)==1:
            add_gt_graph_edge(self.gt_g, head, tail)
        if not self.head_first:
            temp = neg_tail
            neg_tail = head_arr
            head_arr = temp
        ret = head_arr, neg_tail, rel_arr, neg_tail_dist, neg_tail_gd, neg_tail_gd_len, np.array([index, index+self.num_edges]), np.array([len(head_arr)])
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
        _,_,edges = self.graph.edge_ids(head,tail, return_uv=True)
        self.timer.cal_and_update('sample')

        if self.mode=='train' and len(edges)==1:
            remove_gt_graph_edge(self.gt_g, head, tail)

        adj = adjacency(self.gt_g)
        self.timer.cal_and_update('graph')
        head_gd = get_single_source_ver_gd(self.gt_g, adj, head, neg_tail, self.params.reach_dist)
        self.timer.cal_and_update('gd')

        head_arr = np.array([head]).repeat(len(neg_tail))
        rel_arr = np.array([rel]).repeat(len(neg_tail))

        if self.mode=='train' and len(edges)==1:
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
        self.timer.cal_and_update('collect')
        ret = head_arr, neg_tail, rel_arr, head_gd[4], head_close_gd_arr, tail_close_gd_arr, head_close_gd_len, tail_close_gd_len, np.array([index, index+self.num_edges]), np.array([len(head_arr)])
        return ret

    def get_collate_fn(self):
        return collate_ver_link_kg


class HGHorGDDataset(SingleGraphDataset):
    def __init__(self, graph, edges, labels, num_entities, params, mode='train'):
        super().__init__(graph)
        self.mode = mode
        self.params = params
        self.edges = edges
        self.labels = labels
        self.num_edges = len(edges)
        self.pos_edge_count = int(self.num_edges/2)
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
        if self.mode=='train' and label==1:
            remove_gt_graph_edge(self.gt_g, head, tail)
            masked_edge = [index, index+self.pos_edge_count]
        else:
            masked_edge = []
        self.timer.cal_and_update('pre')
        head_dist, head_pred = shortest_distance(self.gt_g, head, pred_map=True, max_dist=self.params.reach_dist)
        self.timer.cal_and_update('sp')
        # all_pred = all_predecessors(self.gt_g, head_dist, head_pred)
        # rand_path = random_shortest_path(self.gt_g, head, tail, head_dist, head_pred,all_pred)
        # rand_path = random_shortest_path(self.gt_g, head, tail)
        rand_path, _ = shortest_path(self.gt_g, head, tail, pred_map=head_pred)
        rand_path = np.array(rand_path,dtype=int)
        self.timer.cal_and_update('random_p')
        
        gd_length = np.array([len(rand_path)])
        dist = gd_length-1
        dist[dist<0] = self.params.reach_dist+2

        gd = rand_path
        self.timer.cal_and_update('gdgg')
        
        ret = np.array([head]), np.array([tail]), dist, gd, gd_length, np.array(masked_edge), np.array([label])
        if self.mode=='train' and label==1:
            add_gt_graph_edge(self.gt_g, head, tail)
        # print(link_arr)
        # print(dist_arr)
        # print(geodesics)
        return ret

    def get_collate_fn(self):
        return collate_hor_link

class HGHorSampleGDDataset(SingleGraphDataset):
    def __init__(self, graph, edges, labels, num_entities, params, mode='train', sample_size=20000, max_nodes_per_hop=10000):
        super().__init__(graph)
        self.mode = mode
        self.params = params
        self.edges = edges
        self.labels = labels
        self.num_edges = len(edges)
        self.pos_edge_count = int(self.num_edges/2)
        self.timer = SmartTimer(False)
        self.sample_size = sample_size
        self.max_nodes_per_hop = max_nodes_per_hop
        self.__getitem__(0)

    def __len__(self):
        return self.num_edges

    def __getitem__(self, index):
        self.timer.record()
        edge = self.edges[index]
        head, rel, tail = edge
        label = self.labels[index]
        self.timer.record()
        
        gt_g, neighbors, maini2sub2 = sample_subgraph_around_link(self.adj_mat, head, tail, self.params.reach_dist, sample_size=self.sample_size, max_nodes_per_hop=self.max_nodes_per_hop)
        subhead, subtail = maini2sub2[head], maini2sub2[tail]

        
        self.timer.cal_and_update('gdgraph')

        if self.mode=='train' and label==1:
            remove_gt_graph_edge(gt_g, subhead, subtail)
            masked_edge = [index, index+self.pos_edge_count]
        else:
            masked_edge = []
        self.timer.cal_and_update('indexing')
        head_dist, head_pred = shortest_distance(gt_g, subhead, pred_map=True, max_dist=self.params.reach_dist)
        self.timer.cal_and_update('sp')
        # all_pred = all_predecessors(self.gt_g, head_dist, head_pred)
        # rand_path = random_shortest_path(self.gt_g, head, tail, head_dist, head_pred,all_pred)
        # rand_path = random_shortest_path(self.gt_g, head, tail)
        rand_path, _ = shortest_path(gt_g, subhead, subtail, pred_map=head_pred)
        rand_path = np.array(rand_path,dtype=int)
        self.timer.cal_and_update('random_p')
        
        gd_length = np.array([len(rand_path)])
        dist = gd_length-1
        dist[dist<0] = self.params.reach_dist+2

        gd = rand_path
        self.timer.cal_and_update('gdgg')
        
        ret = np.array([head]), np.array([tail]), dist, neighbors[gd], gd_length, np.array(masked_edge), np.array([label])
        # print(link_arr)
        # print(dist_arr)
        # print(geodesics)
        return ret

    def get_collate_fn(self):
        return collate_hor_link

class HGVerGDDataset(SingleGraphDataset):
    def __init__(self, graph, edges, labels, num_entities, params, mode='train', hop_sampling=False):
        super().__init__(graph)
        self.mode = mode
        self.params = params
        self.edges = edges
        self.labels = labels
        self.num_edges = len(edges)
        self.pos_edge_count = int(self.num_edges/2)
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
        if self.mode=='train' and label==1:
            remove_gt_graph_edge(self.gt_g, head, tail)
            masked_edge = [index, index+self.pos_edge_count]
        else:
            masked_edge = []
        self.timer.cal_and_update('remove')

        dist, head_gd, tail_gd, head_gd_len, tail_gd_len, head_gd_deg, tail_gd_deg = get_pair_wise_vert_gd(self.gt_g, self.adj_mat, head, tail, self.params.reach_dist)
        # print(ret)
        if self.mode=='train' and label==1:
            add_gt_graph_edge(self.gt_g, head, tail)
        self.timer.cal_and_update('adde')
        # print(link_arr)
        # print(dist_arr)
        # print(geodesics)
        return np.array([head]), np.array([tail]), dist, head_gd.astype(int), tail_gd.astype(int), head_gd_len, tail_gd_len, head_gd_deg, tail_gd_deg, np.array(masked_edge), np.array([label])

    def get_collate_fn(self):
        return collate_ver_link

class HGVerGDSampleDataset(SingleGraphDataset):
    def __init__(self, graph, edges, labels, num_entities, params, mode='train', sample_size=20000, max_nodes_per_hop=10000):
        super().__init__(graph)
        self.mode = mode
        self.params = params
        self.edges = edges
        self.labels = labels
        self.num_edges = len(edges)
        self.max_nodes_per_hop = max_nodes_per_hop
        self.sample_size = sample_size
        self.pos_edge_count = int(self.num_edges/2)
        self.timer = SmartTimer(False)
        self.__getitem__(0)

    def __len__(self):
        return self.num_edges

    def __getitem__(self, index):
        self.timer.record()
        edge = self.edges[index]
        head, rel, tail = edge
        label = self.labels[index]
        
        gt_g, neighbors, maini2sub2 = sample_subgraph_around_link(self.adj_mat, head, tail, self.params.reach_dist, sample_size=self.sample_size, max_nodes_per_hop=self.max_nodes_per_hop)
        self.timer.cal_and_update('subg')

        subhead, subtail = maini2sub2[head], maini2sub2[tail]

        if self.mode=='train' and label==1:
            remove_gt_graph_edge(gt_g, subhead, subtail)
            masked_edge = [index, index+self.pos_edge_count]
        else:
            masked_edge = []
        self.timer.cal_and_update('indexing')
        clean_adj = adjacency(gt_g)
        self.timer.cal_and_update('adj')

        dist, head_gd, tail_gd, head_gd_len, tail_gd_len, head_gd_deg, tail_gd_deg = get_pair_wise_vert_gd(gt_g, clean_adj, subhead, subtail, self.params.reach_dist)
        self.timer.cal_and_update('gd')
        # print(link_arr)
        # print(dist_arr)
        # print(geodesics)
        return np.array([head]), np.array([tail]), dist, neighbors[head_gd.astype(int)], neighbors[tail_gd.astype(int)], head_gd_len, tail_gd_len, head_gd_deg, tail_gd_deg, np.array(masked_edge), np.array([label])

    def get_collate_fn(self):
        return collate_ver_link

class HGVerGDGraphDataset(DatasetWithCollate):
    def __init__(self, graphs, labels, params):
        super().__init__()
        self.graphs = graphs
        self.params = params
        self.num_graphs = len(self.graphs)
        self.labels = labels
        self.timer =SmartTimer(False)

    def __len__(self):
        return self.num_graphs
    
    def __getitem__(self, index):
        self.timer.record()
        graph = self.graphs[index]
        adj = graph.adjacency_matrix(transpose=False, scipy_fmt='csr')
        num_nodes = graph.num_nodes()

        self.timer.cal_and_update('sht')

        distance_mat = shortest_dist_sparse_mult(adj, self.params.reach_dist)
        self.timer.cal_and_update('dist')

        # Block self distance
        distance_mat[np.arange(num_nodes),np.arange(num_nodes)]=self.params.reach_dist+2
        self.timer.cal_and_update('block_dist')

        # Obtain list of reachable nodes in pair wise format
        center_nodes, reached_nodes = (distance_mat<=self.params.reach_dist).nonzero()
        self.timer.cal_and_update('reachable nodes')
        if len(reached_nodes)==0:
            ret = graph, np.array([]), np.array([0]), np.array([]), np.array([]), np.array([0]), np.array([]), np.array([self.labels[index].numpy()])
            self.timer.cal_and_update('fulltime')
            return ret
        # Recover self distance
        distance_mat[np.arange(num_nodes),np.arange(num_nodes)]=0
        self.timer.cal_and_update('unblock dist')
        # Number of reached nodes
        num_reached = np.bincount(center_nodes, minlength=num_nodes)
        self.timer.cal_and_update('reach count')
        # Reached nodes distance
        reached_dist = distance_mat[center_nodes, reached_nodes]
        self.timer.cal_and_update('reached dist')
        # Reached nodes neighbors
        reached_nodes_multi, reached_nodes_neighbor = adj[reached_nodes].nonzero()
        self.timer.cal_and_update('reached neighbor')

        # Dist from center nodes to reached nodes repeated
        reached_dist_multi = reached_dist[reached_nodes_multi]
        self.timer.cal_and_update('reached dist multi')

        # Center nodes index repeated
        center_nodes_multi = center_nodes[reached_nodes_multi]
        self.timer.cal_and_update('centered node multi')

        # Distance between center nodes and reached nodes neighbors
        center_neighbor_dist = distance_mat[center_nodes_multi, reached_nodes_neighbor]
        self.timer.cal_and_update('center neighbor dist')

        # Dist(Reached nodes, Center node)==Dist(Reached nodes neighbor, Center node)+1?
        gd_nodes = (reached_dist_multi-center_neighbor_dist==1)
        self.timer.cal_and_update('goodnei')
        # 
        good_nei_ct = reached_nodes_multi[gd_nodes]
        good_nei = reached_nodes_neighbor[gd_nodes]
        gd_c = np.bincount(good_nei_ct, minlength=len(reached_nodes))

        self.timer.cal_and_update('gdcount')
        gd_deg = get_gd_deg_flat_batch(good_nei, gd_c, adj)
        self.timer.cal_and_update('obtain deg')

        neighbors_ret = reached_nodes
        neighbors_count_ret = num_reached
        gd_ret = good_nei
        gd_count_ret = gd_c
        dist_ret = reached_dist
        deg_ret = gd_deg

        ret = graph, neighbors_ret, neighbors_count_ret, dist_ret, gd_ret, gd_count_ret, deg_ret, np.array([self.labels[index]])
        self.timer.cal_and_update('fulltime')
        return ret

    def get_collate_fn(self):
        return collate_graph_hg

class HGVerGDNodeDataset(SingleGraphDataset):
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

        node_distance = shortest_distance(self.gt_g, source=node, max_dist=self.params.reach_dist)

        node_distance = node_distance.a
        node_distance[node] = self.params.reach_dist+2

        neighbors = (node_distance<=self.params.reach_dist).nonzero()[0]
        neighbor_dist = node_distance[neighbors]

        neighbor_count = len(neighbors)
        if neighbor_count==0:
            ret = np.array([node]), np.array([]), np.array([]), np.array([0]), np.array([]), np.array([0]), np.array([self.labels[index].numpy()])
            self.timer.cal_and_update('fulltime')
            return ret

        node_distance[node] = 0

        gd_arr, gd_len = get_single_source_ver_gd_far(self.adj_mat, neighbors, node_distance)

        gd_deg = get_gd_deg_flat_batch(gd_arr, gd_len, self.adj_mat)

        ret = np.array([node]), neighbors, neighbor_dist, np.array([neighbor_count]), gd_arr, gd_len, gd_deg, np.array([label])
        
        self.timer.cal_and_update('adde')
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

    def __getitem__(self,index):
        head = random.randint(0, self.num_ett-1)
        nei = self.adj[head]
        while True:
            tail = random.randint(0, self.num_ett-1)
            if head != tail and nei[0, tail]==0:
                break
        return np.array([head, tail])