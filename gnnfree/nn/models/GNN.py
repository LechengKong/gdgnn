import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABCMeta, abstractmethod
from dgl.nn.pytorch import RelGraphConv
from gnnfree.nn.models.basic_models import MLPLayers

from gnnfree.nn.models.gnn_layers import GINELayer, GINLayer
from gnnfree.utils.utils import SmartTimer

from torch_scatter import scatter


class MultiLayerMessagePassing(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        num_layers,
        inp_dim,
        out_dim,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.drop_ratio = 0
        self.JK = JK
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        self.drop_ratio = drop_ratio

        self.conv = torch.nn.ModuleList()

        if batch_norm:
            self.batch_norm = torch.nn.ModuleList()
            for layer in range(num_layers):
                self.batch_norm.append(torch.nn.BatchNorm1d(out_dim))
        else:
            self.batch_norm = None

        self.timer = SmartTimer(False)

    def build_layers(self):
        for layer in range(self.num_layers):
            if layer == 0:
                self.conv.append(
                    self.build_one_layer(self.inp_dim, self.out_dim)
                )
            else:
                self.conv.append(
                    self.build_one_layer(self.out_dim, self.out_dim)
                )

    @abstractmethod
    def build_one_layer(self, inp_dim, out_dim):
        pass

    @abstractmethod
    def layer_forward(self, layer, message):
        pass

    @abstractmethod
    def build_message_from_graph(self, g):
        pass

    @abstractmethod
    def build_message_from_output(self, g, h):
        pass

    def forward(self, g):
        h_list = []

        message = self.build_message_from_graph(g)

        for layer in range(self.num_layers):
            # print(layer, h)
            h = self.layer_forward(layer, message)
            if self.batch_norm:
                h = self.batch_norm[layer](h)
            if layer != self.num_layers - 1:
                h = F.dropout(
                    F.relu(h), p=self.drop_ratio, training=self.training
                )
            else:
                h = F.dropout(h, p=self.drop_ratio, training=self.training)
            message = self.build_message_from_output(g, h)
            h_list.append(h)

        if self.JK == "last":
            repr = h_list[-1]
        elif self.JK == "sum":
            repr = 0
            for layer in range(self.num_layers):
                repr += h_list[layer]
        return repr


class MultiLayerMessagePassingVN(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers,
        inp_dim,
        out_dim,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )

        self.virtualnode_embedding = torch.nn.Embedding(1, self.out_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        self.virtualnode_mlp_list = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            self.virtualnode_mlp_list.append(
                MLPLayers(
                    2, h_units=[self.out_dim, 2 * self.out_dim, self.out_dim]
                )
            )

    def forward(self, g):
        h_list = []

        message = self.build_message_from_graph(g)

        vnode_embed = self.virtualnode_embedding(
            torch.zeros(g.batch_size, dtype=torch.int).to(g.device)
        )

        batch_node_segment = torch.arange(
            g.batch_size, dtype=torch.long, device=g.device
        ).repeat_interleave(g.batch_num_nodes())

        for layer in range(self.num_layers):
            # print(layer, h)
            h = self.layer_forward(layer, message)
            if self.batch_norm:
                h = self.batch_norm[layer](h)
            if layer != self.num_layers - 1:
                h = F.dropout(
                    F.relu(h), p=self.drop_ratio, training=self.training
                )
            else:
                h = F.dropout(h, p=self.drop_ratio, training=self.training)
            message = self.build_message_from_output(g, h)
            h_list.append(h)

            if layer < self.num_layers - 1:
                vnode_emb_temp = (
                    scatter(
                        h, batch_node_segment, dim=0, dim_size=g.batch_size
                    )
                    + vnode_embed
                )

                vnode_embed = F.dropout(
                    self.virtualnode_mlp_list[layer](vnode_emb_temp),
                    self.drop_ratio,
                    training=self.training,
                )

        if self.JK == "last":
            repr = h_list[-1]
        elif self.JK == "sum":
            repr = 0
            for layer in range(self.num_layers):
                repr += h_list[layer]
        return repr


class HomogeneousGNN(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers,
        inp_dim,
        out_dim,
        layer_t=GINLayer,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.layer_t = layer_t
        self.build_layers()

    def build_one_layer(self, inp_dim, out_dim):
        return self.layer_t(inp_dim, out_dim)

    def build_message_from_graph(self, g):
        return {"g": g, "h": g.ndata["feat"]}

    def build_message_from_output(self, g, h):
        return {"g": g, "h": h}

    def layer_forward(self, layer, message):
        return self.conv[layer](message["g"], message["h"])


class HomogeneousEdgeGNN(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers,
        inp_dim,
        out_dim,
        edge_dim,
        layer_t=GINELayer,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, layer_t, drop_ratio, JK, batch_norm
        )
        self.edge_dim = edge_dim
        self.layer_t = layer_t
        self.build_layers()

    def build_one_layer(self, inp_dim, out_dim):
        return self.layer_t(inp_dim, out_dim, self.edge_dim)

    def build_message_from_graph(self, g):
        return {"g": g, "h": g.ndata["feat"], "e": g.edata["feat"]}

    def build_message_from_output(self, g, h):
        return {"g": g, "h": h, "e": g.edata["feat"]}

    def layer_forward(self, layer, message):
        return self.conv[layer](message["g"], message["h"], message["e"])


class RGCN(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers,
        num_rels,
        inp_dim,
        out_dim,
        num_bases=4,
        layer_t=RelGraphConv,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.layer_t = layer_t
        self.build_layers()

    def build_one_layer(self, inp_dim, out_dim):
        return RelGraphConv(
            inp_dim,
            out_dim,
            self.num_rels,
            num_bases=self.num_bases,
            activation=F.relu,
        )

    def build_message_from_graph(self, g):
        return {"g": g, "h": g.ndata["feat"], "e": g.edata["type"]}

    def build_message_from_output(self, g, h):
        return {"g": g, "h": h, "e": g.edata["type"]}

    def layer_forward(self, layer, message):
        return self.conv[layer](message["g"], message["h"], message["e"])
