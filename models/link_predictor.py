import torch
import torch.nn as nn

from gnnfree.nn.models.task_predictor import BaseLinkEncoder
from gnnfree.nn.pooling import *


class GDLinkPredictor(BaseLinkEncoder):
    def __init__(self, emb_dim, gnn, feature_list, num_rels=None):

        super().__init__(emb_dim, gnn)
        self.feature_list = feature_list
        self.num_rels = num_rels
        self.link_dim = 0
        self.build_predictor()

    def build_predictor(self):
        self.feature_module = nn.ModuleDict()
        for ft in self.feature_list:
            if ft == "":
                continue
            if ft == "dist":
                self.feature_module[ft] = IdentityTransform(1)
            elif ft == "head":
                self.feature_module[ft] = ReprIndexTransform(self.emb_dim)
            elif ft == "tail":
                self.feature_module[ft] = ReprIndexTransform(self.emb_dim)
            elif ft == "HeadVerGD":
                self.feature_module[ft] = VerGDTransform(
                    self.emb_dim, gd_deg=False
                )
            elif ft == "HeadVerGDDeg":
                self.feature_module[ft] = VerGDTransform(
                    self.emb_dim, gd_deg=True
                )
            elif ft == "TailVerGD":
                self.feature_module[ft] = VerGDTransform(
                    self.emb_dim, gd_deg=False
                )
            elif ft == "TailVerGDDeg":
                self.feature_module[ft] = VerGDTransform(
                    self.emb_dim, gd_deg=True
                )
            elif ft == "HorGD":
                self.feature_module[ft] = ScatterReprTransform(self.emb_dim)
            elif ft == "Rel" and self.num_rels is not None:
                self.feature_module[ft] = EmbTransform(
                    self.emb_dim, self.num_rels
                )
            else:
                raise NotImplementedError
            self.link_dim += self.feature_module[ft].get_out_dim()

    def get_out_dim(self):
        return self.link_dim

    def pool_from_link(self, repr, head, tail, input):
        repr_list = []
        for ft in self.feature_list:
            if ft == "":
                continue
            if ft == "dist":
                repr_list.append(self.feature_module[ft](input.dist))
            elif ft == "head":
                repr_list.append(self.feature_module[ft](repr, head))
            elif ft == "tail":
                repr_list.append(self.feature_module[ft](repr, tail))
            elif ft == "HeadVerGD":
                repr_list.append(
                    self.feature_module[ft](
                        repr, input.head_gd, input.head_gd_len, None
                    )
                )
            elif ft == "HeadVerGDDeg":
                repr_list.append(
                    self.feature_module[ft](
                        repr,
                        input.head_gd,
                        input.head_gd_len,
                        input.head_gd_deg,
                    )
                )
            elif ft == "TailVerGD":
                repr_list.append(
                    self.feature_module[ft](
                        repr, input.tail_gd, input.tail_gd_len, None
                    )
                )
            elif ft == "TailVerGDDeg":
                repr_list.append(
                    self.feature_module[ft](
                        repr,
                        input.tail_gd,
                        input.tail_gd_len,
                        input.tail_gd_deg,
                    )
                )
            elif ft == "HorGD":
                repr_list.append(
                    self.feature_module[ft](repr, input.gd, input.gd_len)
                )
            elif ft == "Rel" and self.num_rels is not None:
                repr_list.append(self.feature_module[ft](input.rel))
        g_rep = torch.cat(repr_list, dim=1)
        return g_rep
