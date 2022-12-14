from gnnfree.nn.models.task_predictor import BaseNodeEncoder
from gnnfree.nn.pooling import GDTransform


class GDNodeClassifier(BaseNodeEncoder):
    def __init__(self, emb_dim, gnn, gd_deg=True):
        super().__init__(emb_dim, gnn)
        self.gd_pool = GDTransform(emb_dim, gd_deg)

    def pool_from_node(self, g, repr, node, input):
        node_repr = self.gd_pool(
            repr,
            node,
            input.neighbors,
            input.neighbor_count,
            input.dist,
            input.gd,
            input.gd_count,
            input.gd_deg,
        )
        return node_repr

    def get_out_dim(self):
        return self.emb_dim
