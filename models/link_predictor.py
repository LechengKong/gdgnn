import torch
import torch.nn.functional as F
import torch.nn as nn

from gnnfree.nn.models.basic_models import MLPLayers
from gnnfree.nn.pooling import feature_module_dict
from gnnfree.nn.models.task_predictor import LinkPredictor

def feature_selector(ft, repr, input):
    if ft=='node_pair':
        return (repr, input.head, input.tail)
    elif ft=='rel':
        return [input.rel]
    elif ft=='HorGD':
        return (repr, input.gd, input.gd_len)
    elif ft=='VerGD':
        return (repr, input.head_gd, input.tail_gd, input.head_gd_len, input.tail_gd_len)
    elif ft=='dist':
        return [input.dist]
    elif ft=='':
        return ()
    else:
        raise NotImplementedError

class GDLinkPredictor(LinkPredictor):
    def __init__(self, emb_dim, gnn, feature_list, add_self_loop=False):

        self.feature_list = feature_list

        super().__init__(emb_dim, gnn, add_self_loop)

    def build_predictor(self):
        self.feature_module = nn.ModuleDict()
        self.feature_norm = nn.ModuleDict()
        for ft in self.feature_list:
            if ft != '':
                if ft == 'dist':
                    self.feature_module[ft] = feature_module_dict['dist']()
                else:
                    self.feature_module[ft] = feature_module_dict[ft](self.emb_dim)
                self.link_dim+=self.feature_module[ft].get_out_dim()
                self.feature_norm[ft] = nn.BatchNorm1d(self.feature_module[ft].get_out_dim())

    def predict_link(self, repr, head, tail, input):
        repr_list = []
        repr_list.append(self.node_pair_extract(repr, head, tail))
        for ft in self.feature_list:
            if ft!='':
                repr_list.append(self.feature_module[ft](*feature_selector(ft, repr, input)))
        g_rep = torch.cat(repr_list, dim=1)
        output = self.link_mlp(g_rep)
        return output

class RelLinkPredictor(GDLinkPredictor):
    def __init__(self, emb_dim, num_rels, gnn, feature_list, add_self_loop=False):
        self.num_rels = num_rels
        super().__init__(emb_dim, gnn, feature_list, add_self_loop)
    
    def build_predictor(self):
        super().build_predictor()
        self.feature_list.append('rel')
        self.feature_module['rel'] = feature_module_dict['rel'](self.emb_dim, self.num_rels)
        self.link_dim += self.feature_module['rel'].get_out_dim()
        self.feature_norm['rel'] = nn.BatchNorm1d(self.feature_module['rel'].get_out_dim())
        
        