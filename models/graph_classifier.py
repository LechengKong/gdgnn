import torch
import dgl

from gnnfree.nn.pooling import GDTransform
from gnnfree.nn.models.task_predictor import BaseGraphEncoder


class GDGraphClassifier(BaseGraphEncoder):
    def __init__(self, emb_dim, gnn, gd_deg=True, batch_norm=True):
        super().__init__(emb_dim, gnn)

        self.gd_pool = GDTransform(emb_dim, gd_deg, batch_norm)

    def pool_from_graph(self, g, repr, input):
        node_repr = self.gd_pool(
            repr,
            torch.arange(len(repr), device=repr.device),
            input.neighbors,
            input.neighbors_count,
            input.dist,
            input.gd,
            input.gd_count,
            input.gd_deg,
        )
        g.ndata["node_res"] = node_repr

        g_sum = dgl.sum_nodes(g, "node_res")
        return g_sum

    def get_out_dim(self):
        return self.emb_dim
