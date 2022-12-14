import torch
import torch.nn as nn
import dgl

from abc import ABCMeta, abstractmethod
from gnnfree.nn.pooling import ReprIndexTransform


class BaseGNNEncoder(nn.Module, metaclass=ABCMeta):
    def __init__(self, gnn):
        super().__init__()

        self.gnn = gnn

    @abstractmethod
    def forward(self, g, input):
        pass

    @abstractmethod
    def get_out_dim(self):
        pass


class BaseGraphEncoder(BaseGNNEncoder, metaclass=ABCMeta):
    def __init__(self, emb_dim, gnn):
        super().__init__(gnn)

        self.emb_dim = emb_dim

    def forward(self, g, input):

        repr = self.gnn(g)

        return self.pool_from_graph(g, repr, input)

    @abstractmethod
    def pool_from_graph(self, g, repr, input):
        pass


class GraphEncoder(BaseGraphEncoder):
    def __init__(self, emb_dim, gnn, pooling="sum"):
        super().__init__(emb_dim, gnn)
        self.pooling = "sum"

    def pool_from_graph(self, g, repr, input):
        repr = self.repr_post_process(g, repr, input)

        g.ndata["node_res"] = repr

        g_repr = dgl.readout_nodes(g, "node_res", op=self.pooling)
        return g_repr

    def repr_post_process(self, g, repr, input):
        return repr

    def get_out_dim(self):
        return self.gnn.out_dim


class BaseLinkEncoder(BaseGNNEncoder, metaclass=ABCMeta):
    def __init__(self, emb_dim, gnn):
        super().__init__(gnn)
        self.emb_dim = emb_dim
        self.use_only_embedding = False

    def embedding_only_mode(self, state=True):
        self.use_only_embedding = state

    def forward(self, g, head, tail, input):
        if self.use_only_embedding:
            repr = g.ndata["repr"]
        else:
            repr = self.gnn(g)
        return self.pool_from_link(repr, head, tail, input)

    @abstractmethod
    def pool_from_link(self, repr, head, tail, input):
        pass


class LinkEncoder(BaseLinkEncoder):
    def __init__(self, emb_dim, gnn):
        super().__init__(emb_dim, gnn)
        self.node_repr_extractor = ReprIndexTransform(gnn.out_dim)

    def pool_from_link(self, repr, head, tail, input):
        repr_list = []
        repr_list.append(self.node_repr_extractor(repr, head))
        repr_list.append(self.node_repr_extractor(repr, tail))
        g_rep = torch.cat(repr_list, dim=1)
        return g_rep

    def get_out_dim(self):
        return self.gnn.out_dim * 2


class BaseNodeEncoder(BaseGNNEncoder):
    def __init__(self, emb_dim, gnn):
        super().__init__(gnn)
        self.emb_dim = emb_dim
        self.use_only_embedding = False

    def embedding_only_mode(self, state=True):
        self.use_only_embedding = state

    def forward(self, g, node, input):
        if self.use_only_embedding:
            repr = g.ndata["repr"]
        else:
            repr = self.gnn(g)
        return self.pool_from_node(g, repr, node, input)

    @abstractmethod
    def pool_from_node(self, g, repr, node, input):
        pass


class NodeEncoder(BaseNodeEncoder):
    def __init__(self, emb_dim, gnn):
        super().__init__(emb_dim, gnn)
        self.node_repr_extractor = ReprIndexTransform(gnn.out_dim)

    def pool_from_node(self, g, repr, node, input):
        return self.node_repr_extractor(repr, node)

    def get_out_dim(self):
        return self.gnn.out_dim
