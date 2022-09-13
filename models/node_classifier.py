from gnnfree.nn.models.task_predictor import NodeClassifier
from gnnfree.nn.pooling import GDPool


class GDNodeClassifier(NodeClassifier):
    def __init__(self, num_classes, emb_dim, gnn, add_self_loop=False, gd_deg=True):
        super().__init__(num_classes, emb_dim, gnn, add_self_loop)
        self.gd_pool = GDPool(emb_dim, gd_deg)
    
    def pool_node(self, g, repr, node, input):
        node_repr = self.gd_pool(repr, node, input.neighbors, input.neighbor_count, input.dist, input.gd, input.gd_count, input.gd_deg)
        return node_repr