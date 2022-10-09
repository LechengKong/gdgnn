import torch
import numpy as np
import dgl

from torch.utils.data import DataLoader, RandomSampler

from gnnfree.managers.learner import SingleModelLearner

class LinkPredictionLearner(SingleModelLearner):
    def load(self, batch, device):
        batch = batch.to(device)
        batch.to_name()
        return batch

    def forward_func(self, batch):
        g = self.data.graph.to(batch.device)
        edge_mask = batch.edge_mask
        edge_bool = torch.ones(g.num_edges(), dtype=torch.bool, device=batch.device)
        edge_bool[edge_mask] = 0
        subg = dgl.edge_subgraph(g, edge_bool, relabel_nodes=False)
        batch.g = subg
        res = self.model(subg, batch.head, batch.tail, batch)
        return res

    def data_to_loss_arg(self, res, batch):
        return res, batch.labels

class PrecomputeNELPLearner(SingleModelLearner):
    def preprocess(self, device=None):
        self.processed_graph = self.data.graph.to(device)
        repr = self.model.process_graph(self.processed_graph)
        self.processed_graph.ndata['repr'] = repr
        self.model.embedding_only_mode(True)
    
    def load(self, batch, device):
        batch = batch.to(device)
        batch.to_name()
        return batch

    def forward_func(self, batch):
        batch.g = self.processed_graph
        res = self.model(self.processed_graph, batch.head, batch.tail, batch)
        return res

    def postprocess(self):
        self.processed_graph = None
        self.model.embedding_only_mode(False)

class KGPrecomputeNELPLearner(PrecomputeNELPLearner):
    def data_to_eval_arg(self, res, batch):
        return res, batch.bsize

class HGPrecomputeNELPLearner(PrecomputeNELPLearner):
    def data_to_eval_arg(self, res, batch):
        return res, batch.labels


class LinkFixedSizeRankingLearner(LinkPredictionLearner):

    def data_to_loss_arg(self, res, batch):
        return [res]

    def data_to_eval_arg(self, res, batch):
        return res, batch.bsize


class GraphPredictionLearner(SingleModelLearner):
    def load(self, batch, device):
        batch = batch.to(device)
        batch.to_name()
        return batch

    def forward_func(self, batch):
        res = self.model(batch.g, batch)
        # print(res)
        return res

    def data_to_loss_arg(self, res, batch):
        return res, batch.labels


class NodePredictionLearner(SingleModelLearner):
    def load(self, batch, device):
        batch = batch.to(device)
        batch.to_name()
        return batch

    def forward_func(self, batch):
        g = self.data.graph.to(batch.node.device)
        batch.g = g
        res = self.model(g, batch.node, batch)
        return res

    def data_to_loss_arg(self, res, batch):
        return res, batch.labels


class PrecomputeNENCLearner(SingleModelLearner):
    def preprocess(self, device=None):
        self.processed_graph = self.data.graph.to(device)
        repr = self.model.process_graph(self.processed_graph)
        self.processed_graph.ndata['repr'] = repr
        self.model.embedding_only_mode(True)
    
    def load(self, batch, device):
        batch = batch.to(device)
        batch.to_name()
        return batch

    def forward_func(self, batch):
        batch.g = self.processed_graph
        res = self.model(self.processed_graph, batch.node, batch)
        return res

    def postprocess(self):
        self.processed_graph = None
        self.model.embedding_only_mode(False)

    def data_to_eval_arg(self, res, batch):
        return res, batch.labels