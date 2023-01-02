import torch
from gnnfree.nn.models.GNN import (
    MultiLayerMessagePassing,
    MultiLayerMessagePassingVN,
)
from gnnfree.nn.models.gnn_layers import GINELayer
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


def MOLGNN_generator(parent_GNN):
    class MOLGNN(parent_GNN):
        def __init__(
            self,
            num_layers,
            inp_dim,
            out_dim,
            mol_emb_dim,
            edge_dim,
            drop_ratio=0,
            JK="last",
            batch_norm=True,
        ):
            super().__init__(
                num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
            )
            self.mol_emb_dim = mol_emb_dim
            self.edge_dim = edge_dim

            self.atom_encoder = AtomEncoder(mol_emb_dim)
            self.bond_encoder = torch.nn.ModuleList()
            self.encoded_inp_dim = inp_dim + mol_emb_dim
            self.encoded_edge_dim = edge_dim + mol_emb_dim

            self.build_layers()

        def build_layers(self):
            for layer in range(self.num_layers):
                self.bond_encoder.append(BondEncoder(self.mol_emb_dim))
                if layer == 0:
                    self.conv.append(
                        GINELayer(
                            self.encoded_inp_dim,
                            self.out_dim,
                            self.encoded_edge_dim,
                        )
                    )
                else:
                    self.conv.append(
                        GINELayer(
                            self.out_dim, self.out_dim, self.encoded_edge_dim
                        )
                    )

        def build_one_layer(self, inp_dim, out_dim):
            pass

        def build_message_from_graph(self, g):
            return {
                "g": g,
                "h": torch.cat(
                    [g.ndata["feat"], self.atom_encoder(g.ndata["atom_feat"])],
                    dim=-1,
                ),
                "e": g.edata["feat"],
                "bond": g.edata["bond_feat"],
            }

        def build_message_from_output(self, g, h):
            return {
                "g": g,
                "h": h,
                "e": g.edata["feat"],
                "bond": g.edata["bond_feat"],
            }

        def layer_forward(self, layer, message):
            e_feat = torch.cat(
                [message["e"], self.bond_encoder[layer](message["bond"])],
                dim=-1,
            )

            return self.conv[layer](message["g"], message["h"], e_feat)

    return MOLGNN


MOLGNN = MOLGNN_generator(MultiLayerMessagePassing)
MOLGNNVN = MOLGNN_generator(MultiLayerMessagePassingVN)
