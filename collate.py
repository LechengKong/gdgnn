import sys
import torch
import dgl
import numpy as np
import torch
import random
import time


class HGLinkHorBatch:
    def __init__(self, samples):
        d = zip(*samples)
        self.ls = []
        for d1 in d:
            p = np.concatenate(d1)
            p = torch.tensor(p, dtype=torch.long)
            self.ls.append(p)
        # print(self.ls)

    def pin_memory(self):
        for i in range(len(self.ls)):
            self.ls[i] = self.ls[i].pin_memory()
        return self

    def to_name(self):
        self.head = self.ls[0]
        self.tail = self.ls[1]
        self.dist = self.ls[2]
        self.gd = self.ls[3]
        self.gd_len = self.ls[4]
        self.edge_mask = self.ls[5]
        self.labels = self.ls[6]

    def to(self, device):
        for i in range(len(self.ls)):
            self.ls[i] = self.ls[i].to(device)
        self.device = device
        self.to_name()
        return self


class HGLinkVerBatch:
    def __init__(self, samples):
        d = zip(*samples)
        self.ls = []
        for d1 in d:
            p = np.concatenate(d1)
            p = torch.tensor(p, dtype=torch.long)
            self.ls.append(p)
        # print(self.ls)

    def pin_memory(self):
        for i in range(len(self.ls)):
            self.ls[i] = self.ls[i].pin_memory()
        return self

    def to_name(self):
        self.head = self.ls[0]
        self.tail = self.ls[1]
        self.dist = self.ls[2]
        self.head_gd = self.ls[3]
        self.tail_gd = self.ls[4]
        self.head_gd_len = self.ls[5]
        self.tail_gd_len = self.ls[6]
        self.head_gd_deg = self.ls[7]
        self.tail_gd_deg = self.ls[8]
        self.edge_mask = self.ls[9]
        self.labels = self.ls[10]

    def to(self, device):
        for i in range(len(self.ls)):
            self.ls[i] = self.ls[i].to(device)
        self.device = device
        self.to_name()
        return self


class KGLinkHorBatch:
    def __init__(self, samples):
        d = zip(*samples)
        self.ls = []
        for d1 in d:
            p = np.concatenate(d1)
            p = torch.tensor(p, dtype=torch.long)
            self.ls.append(p)

    def pin_memory(self):
        for i in range(len(self.ls)):
            self.ls[i] = self.ls[i].pin_memory()
        return self

    def to_name(self):
        self.head = self.ls[0]
        self.tail = self.ls[1]
        self.rel = self.ls[2]
        self.dist = self.ls[3]
        self.gd = self.ls[4]
        self.gd_len = self.ls[5]
        self.edge_mask = self.ls[6]
        self.bsize = self.ls[7]

    def to(self, device):
        for i in range(len(self.ls)):
            self.ls[i] = self.ls[i].to(device)
        self.device = device
        self.to_name()
        return self


class KGLinkVerBatch:
    def __init__(self, samples):
        d = zip(*samples)
        self.ls = []
        for i, d1 in enumerate(d):
            p = np.concatenate(d1)
            p = torch.tensor(p, dtype=torch.long)
            self.ls.append(p)

    def pin_memory(self):
        for i in range(len(self.ls)):
            self.ls[i] = self.ls[i].pin_memory()
        return self

    def to_name(self):
        self.head = self.ls[0]
        self.tail = self.ls[1]
        self.rel = self.ls[2]
        self.dist = self.ls[3]
        self.head_gd = self.ls[4]
        self.tail_gd = self.ls[5]
        self.head_gd_len = self.ls[6]
        self.tail_gd_len = self.ls[7]
        self.edge_mask = self.ls[8]
        self.bsize = self.ls[9]

    def to(self, device):
        for i in range(len(self.ls)):
            self.ls[i] = self.ls[i].to(device)
        self.device = device
        self.to_name()
        return self


class HGGraphBatch:
    def __init__(self, samples):
        graph_list = []
        neighbors_list = []
        neighbors_count_list = []
        dist_list = []
        gd_list = []
        gd_count_list = []
        gd_deg_list = []
        labels_list = []
        offset = 0
        for (
            graph,
            neighbors_ind,
            neighbors_count,
            neighbors_dist,
            gd,
            gd_count,
            gd_deg,
            labels,
        ) in samples:
            num_nodes = graph.num_nodes()
            graph_list.append(graph)
            neighbors_ind += offset
            neighbors_list.append(neighbors_ind)
            neighbors_count_list.append(neighbors_count)
            dist_list.append(neighbors_dist)
            gd += offset
            gd_list.append(gd)
            gd_count_list.append(gd_count)
            gd_deg_list.append(gd_deg)
            labels_list.append(labels)
            offset += num_nodes
            # print('didi')
        self.ls = []
        batched_graph = dgl.batch(graph_list)
        self.ls.append(batched_graph)
        d = [
            neighbors_list,
            neighbors_count_list,
            dist_list,
            gd_list,
            gd_count_list,
            gd_deg_list,
        ]
        for i, d1 in enumerate(d):
            p = np.concatenate(d1)
            p = torch.tensor(p, dtype=torch.long)
            self.ls.append(p)
        labels_list = np.concatenate(labels_list)
        self.ls.append(torch.tensor(labels_list, dtype=torch.float))

    def pin_memory(self):
        for i in range(1, len(self.ls)):
            self.ls[i] = self.ls[i].pin_memory()
        return self

    def to_name(self):
        self.g = self.ls[0]
        self.neighbors = self.ls[1]
        self.neighbors_count = self.ls[2]
        self.dist = self.ls[3]
        self.gd = self.ls[4]
        self.gd_count = self.ls[5]
        self.gd_deg = self.ls[6]
        self.labels = self.ls[7]

    def to(self, device):
        for i in range(len(self.ls)):
            self.ls[i] = self.ls[i].to(device)
        self.device = device
        self.to_name()
        return self


class HGNodeBatch:
    def __init__(self, samples):
        d = zip(*samples)
        self.ls = []
        for d1 in d:
            p = np.concatenate(d1)
            p = torch.tensor(p, dtype=torch.long)
            self.ls.append(p)

    def pin_memory(self):
        for i in range(1, len(self.ls)):
            self.ls[i] = self.ls[i].pin_memory()
        return self

    def to_name(self):
        self.node = self.ls[0]
        self.neighbors = self.ls[1]
        self.dist = self.ls[2]
        self.neighbor_count = self.ls[3]
        self.gd = self.ls[4]
        self.gd_count = self.ls[5]
        self.gd_deg = self.ls[6]
        self.labels = self.ls[7]

    def to(self, device):
        for i in range(len(self.ls)):
            self.ls[i] = self.ls[i].to(device)
        self.device = device
        self.to_name()
        return self


class GraphLabelBatch:
    def __init__(self, samples):
        graph, labels = zip(*samples)
        batched_graph = dgl.batch(graph)
        labels = torch.tensor(labels)
        self.ls = [batched_graph, labels]

    def pin_memory(self):
        for i in range(1, len(self.ls)):
            self.ls[i] = self.ls[i].pin_memory()
        return self

    def to(self, device):
        for i in range(len(self.ls)):
            self.ls[i] = self.ls[i].to(device)
        self.device = device
        return self

    def to_name(self):
        self.g = self.ls[0]
        self.labels = self.ls[1]


def collate_hor_link(samples):
    return HGLinkHorBatch(samples)


def collate_ver_link(samples):
    return HGLinkVerBatch(samples)


def collate_hor_link_kg(samples):
    return KGLinkHorBatch(samples)


def collate_ver_link_kg(samples):
    return KGLinkVerBatch(samples)


def collate_graph_hg(samples):
    return HGGraphBatch(samples)


def collate_tuple_g(samples):
    return GraphLabelBatch(samples)


def collate_node_hg(samples):
    return HGNodeBatch(samples)
