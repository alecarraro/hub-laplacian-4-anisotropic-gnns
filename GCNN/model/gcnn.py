from .filters import create_filter_list
from .layers import MLP, ConvLayer
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import to_dense_adj
import torch
import torch.nn as nn


class GCNNalpha(nn.Module):
    def __init__(
        self,
        dims: list,
        output_dim: int,
        hops: int,
        activation,
        gso_generator: callable,
        alpha: float = 0.5,
        learn_alpha: bool = True,
        pooling: str = 'max',
        readout_hidden_dims: list = None,   # renamed
        apply_readout: bool = True,
        use_bn=False,
        dropout_p=0.0
    ):
        super().__init__()

        self.reduction = pooling
        self.apply_readout = apply_readout
        self.hops = hops 

        self.learn_alpha = learn_alpha
        if learn_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))  # Non-trainable

        self.gso_generator = gso_generator

        self.layers = nn.ModuleList()
        self.layer_degrees = []

        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            conv_layer = ConvLayer(in_dim,
                out_dim,
                [torch.eye(1)] * (hops + 1),
                activation=activation,
                use_bn= use_bn,
                dropout_p = dropout_p)
            
            self.layers.append(conv_layer)

        if readout_hidden_dims is not None:
            self.readout = MLP([dims[-1]] + readout_hidden_dims,
                               activation=activation)
            last_readout_size = readout_hidden_dims[-1]
        else:
            self.readout = None
            last_readout_size = dims[-1]

        self.output_lin = nn.Linear(last_readout_size, output_dim, bias=True)

    def forward(self, X, batch, edge_index):
        adj = to_dense_adj(edge_index, batch)
        gsos = []
        for i in range(adj.size(0)):
            A_i = adj[i]
            num_nodes_i = (A_i.sum(dim=1) != 0).sum().item()
            A_i = A_i[:num_nodes_i, :num_nodes_i]
            S_i = self.gso_generator(A_i, self.alpha)
            gsos.append(S_i)

        S = torch.block_diag(*gsos)
        x = X

        filters = create_filter_list(S, self.hops)

        for i, conv_layer in enumerate(self.layers):
            conv_layer.filters = filters
            x = conv_layer(x)

        if self.reduction == 'sum':
            x = global_add_pool(x, batch)
        elif self.reduction == 'mean':
            x = global_mean_pool(x, batch)
        elif self.reduction == 'max':
            x = global_max_pool(x, batch)

        # apply readout or default linear mapping to output_dim
        if self.apply_readout and self.readout is not None:
            x = self.readout(x)

        # always apply the final linear to get the correct output_dim
        if self.apply_readout:
            x = self.output_lin(x)
            # if it ends up with last‐dim 1, squeeze
            if x.dim() > 1 and x.size(-1) == 1:
                x = x.squeeze(-1)
        return x