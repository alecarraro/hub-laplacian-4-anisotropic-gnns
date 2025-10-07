import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dims: list, activation=None, bias=True):
        """
        dims: list of layer dimensions [input, hidden1, ..., output]
        activation: single activation function (applied after each hidden layer)
        bias: whether to include biases in linear layers
        """
        super().__init__()
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            if i < len(dims) - 2 and activation is not None:
                layers.append(activation)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, filter_list,
                 activation=None, use_bn=False, dropout_p=0.0):
        super().__init__()
        self.K = len(filter_list) - 1
        self.filters = filter_list
        self.activation = activation

        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_channels, out_channels))
            for _ in range(self.K + 1)
        ])
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.zeros_(self.bias)
        for w in self.weights:
            nn.init.xavier_uniform_(w)

        self.use_bn = use_bn
        if use_bn:
            # BatchNorm over the feature dimension
            self.bn = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else None

    def forward(self, X):
        # X: [N, in_channels]
        out = torch.zeros(X.size(0), self.bias.size(0), device=X.device)
        for k, Lk in enumerate(self.filters):
            out += Lk @ X @ self.weights[k]
        out += self.bias

        if self.activation is not None:
            out = self.activation(out)
        if self.use_bn:
            out = self.bn(out)         # normalize each feature channel
        if self.dropout is not None:
            out = self.dropout(out)    # randomly zero some features

        return out