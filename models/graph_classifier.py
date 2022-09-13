import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl
from torch.nn import LSTM
from torch_scatter import scatter

from gnnfree.nn.pooling import GDPool
from gnnfree.nn.models.task_predictor import GraphClassifier

class GDGraphClassifier(GraphClassifier):
    def __init__(self, num_classes, emb_dim, gnn, add_self_loop=False, gd_deg=True):
        super().__init__(num_classes, emb_dim, gnn, add_self_loop)

        self.gd_pool = GDPool(emb_dim, gd_deg=gd_deg)

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.mlp_graph.reset_parameters()

    def pool_from_graph(self, g, repr, input):
        node_repr = self.gd_pool(repr, torch.arange(len(repr), device=repr.device), input.neighbors, input.neighbors_count, input.dist, input.gd, input.gd_count, input.gd_deg)
        g.ndata['node_res'] = node_repr
        
        g_sum = dgl.sum_nodes(g, 'node_res')
        return g_sum

class SubGraphClassifier(GraphClassifier):
    def pool_from_graph(self, g, repr, input):
        # print(g.num_nodes())
        g.ndata['node_res'] = repr
        g_sum = dgl.sum_nodes(g, 'node_res')
        # print(g_sum.size())
        g_count = input.g_count

        g_sum = scatter(g_sum, torch.arange(len(g_count), device=repr.device).repeat_interleave(g_count), dim=0, dim_size=len(g_count))
        return g_sum

class GraphAttenClassifier(GraphClassifier):
    def __init__(self, num_classes, emb_dim, gnn, add_self_loop=False):
        super().__init__(num_classes, emb_dim, gnn, add_self_loop)
        self.lstm = LSTM(emb_dim, emb_dim, batch_first=True)

    def pool_from_graph(self, g, repr, input):
        # print(g.num_nodes())
        g.ndata['node_res'] = repr
        g_sum = dgl.sum_nodes(g, 'node_res')
        d_size = g_sum.size()[-1]
        # print(g_sum.size())
        g_count = input.g_count[0]
        g_sum = g_sum.view(-1, g_count, d_size)
        # g_sum = g_sum.sum(1)

        g_sum, _ = self.lstm(g_sum)
        # print(g_out.size())
        return g_sum[:,-1]
        # return g_sum