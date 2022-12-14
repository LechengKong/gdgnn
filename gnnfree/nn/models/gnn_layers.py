import torch
import torch.nn as nn
import torch.nn.functional as F

from gnnfree.nn.models.basic_models import MLPLayers
import dgl.function as fn


def edge_msg_func(edges):

    msg = F.relu(edges.src["h"] + edges.data["e"])

    return {"msg": msg}


def edge_mask_msg_func(edges):

    msg = (F.relu(edges.src["h"] + edges.data["e"])) * (
        torch.logical_and(edges.src["mask"], edges.dst["mask"])
    )

    return {"msg": msg}


class GINELayer(nn.Module):
    def __init__(self, in_feats, out_feats, edge_feats, batch_norm=True):
        super(GINELayer, self).__init__()
        self.mlp = MLPLayers(
            2, [in_feats, 2 * in_feats, out_feats], batch_norm=batch_norm
        )
        self.edge_mlp = MLPLayers(
            2, [edge_feats, 2 * edge_feats, in_feats], batch_norm=batch_norm
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, g, node_feat, edge_feat, mask=None):
        with g.local_scope():
            g.ndata["h"] = node_feat
            g.edata["e"] = self.edge_mlp(edge_feat)
            if mask is not None:
                g.ndata["mask"] = mask.view(-1, 1)
                g.update_all(
                    edge_mask_msg_func, fn.sum(msg="msg", out="out_h")
                )
            else:
                g.update_all(edge_msg_func, fn.sum(msg="msg", out="out_h"))
            out = self.mlp((1 + self.eps) * node_feat + g.ndata["out_h"])
            return out


def mask_msg_func(edges):
    msg = edges.src["h"] * (
        torch.logical_and(edges.src["mask"], edges.dst["mask"])
    )
    return {"msg": msg}


class GINLayer(nn.Module):
    def __init__(self, in_feats, out_feats, batch_norm=True):
        super(GINLayer, self).__init__()
        self.mlp = MLPLayers(
            2, [in_feats, 2 * in_feats, out_feats], batch_norm=batch_norm
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, g, feature, mask=None):
        with g.local_scope():
            g.ndata["h"] = feature
            if mask is not None:
                g.ndata["mask"] = mask.view(-1, 1)
                g.update_all(mask_msg_func, fn.sum(msg="msg", out="out_h"))
            else:
                g.update_all(
                    fn.copy_u("h", "msg"), fn.sum(msg="msg", out="out_h")
                )
            out = self.mlp((1 + self.eps) * feature + g.ndata["out_h"])
            # if mask is not None:
            #     out = out*mask.view(-1,1)
            return out
