import torch.nn as nn


class MLPLayers(nn.Module):
    def __init__(
        self, layers, h_units, dropout=0, batch_norm=True, activation=nn.ReLU
    ):
        super().__init__()
        # self.mlp = nn.Sequential()
        # self.activation = activation
        self.layers = layers
        self.fc_list = nn.ModuleList()
        self.batch_norm_list = nn.ModuleList()
        self.batch_norm = batch_norm
        self.dropout_ratio = dropout
        self.activation = activation()
        self.dropout = nn.Dropout(self.dropout_ratio)
        for i in range(layers):
            if i > 0:
                if batch_norm:
                    self.batch_norm_list.append(nn.BatchNorm1d(h_units[i]))
            self.fc_list.append(nn.Linear(h_units[i], h_units[i + 1]))

    def reset_parameters(self):
        for layer in self.fc_list:
            layer.reset_parameters()
        for layer in self.batch_norm_list:
            layer.reset_parameters()

    def forward(self, x):
        for i in range(self.layers):
            if i > 0:
                if self.batch_norm:
                    x = self.batch_norm_list[i - 1](x)
                x = self.activation(x)
                if self.dropout_ratio > 0:
                    x = self.dropout(x)
            x = self.fc_list[i](x)

        return x


class Predictor(nn.Module):
    def __init__(self, encoder, mlp):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp

    def forward(self, *args):
        emb = self.encoder(*args)
        return self.mlp(emb)
