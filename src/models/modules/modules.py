import torch
from torch import nn
from torch.nn import functional as F
from src.models.modules.layers import (
    GumbelSoftmax, 
    Gaussian,
    GraphTransformerBlock,
    GraphConvolutionBlock,
)


class MFNet(nn.Module):
    """MFNet which performs Matrix Factorization based on ag and pe graph, could be
    considered as the user-item relation matrix.
    
    N (int): adjacency matrix node num.
    K (int): latent feature to be predicted.
    """
    def __init__(self, N, K=50):
        super().__init__()
        self.U = nn.Parameter(torch.randn(N, K))
        self.ag_V = nn.Parameter(torch.randn(N, K))
        self.pe_V = nn.Parameter(torch.randn(N, K))

    def forward(self, user_id, value_id):
        """calculate the dot product of ag pred and pe pred based on the 
        user_id and value_id.

        Args:
            user_id (int): user id from sampling.
            value_id (int): value id from sampling.
        """
        ag_pred = (self.U[user_id] * self.ag_V[value_id]).sum(dim=-1)
        pe_pred = (self.U[user_id] * self.pe_V[value_id]).sum(dim=-1)
        return ag_pred, pe_pred


class Encoder(nn.Module):
    """Encoder model which encodes the logits y and latent vector z.
    Currently support GCN and Graph Transformer two kinds block during
    message passing step.

    Args:
        y_dim (int): dimension of the gaussian mixture.
        z_dim (int): dimension of feature latent vector.
        in_channels (list): encoder input channels list.
        out_channels (list): encoder output channels list.
        block_type (string): encoder block type, support 'GCN' and 'Transformer'
        use_bias (boolean): whether use bias in models.
        num_heads (int): number of heads in graph transformer block.
    """
    def __init__(
        self,
        y_dim,
        z_dim,
        in_channels=[],
        out_channels=[],
        block_type="Transformer",
        use_bias=True,
        num_heads=16,
    ):
        super(Encoder, self).__init__()
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_type = block_type
        self.use_bias = use_bias
        self.num_heads = num_heads
        self.encoder_y = torch.nn.ModuleList()
        self.encoder_z = torch.nn.ModuleList()

        assert self.z_dim == self.out_channels[-1]
        assert len(self.in_channels) == len(self.out_channels)
        
        # encode logits.
        if self.block_type == "Transformer":
            for in_dim, out_dim in zip(self.in_channels, self.out_channels):
                self.encoder_y.append(
                    GraphTransformerBlock(
                        input_dim=in_dim,
                        output_dim=out_dim,
                        num_heads=self.num_heads,
                        use_bias=self.use_bias,
                    )
                )
        elif self.block_type == "GCN":
            for in_dim, out_dim in zip(self.in_channels, self.out_channels):
                self.encoder_y.append(
                    GraphConvolutionBlock(
                        in_features=in_dim,
                        out_features=out_dim,
                        use_bias=self.use_bias,
                    )
                )
        else:
            raise NotImplementedError("Not implemented method, \
                only support Transformer and GCN currently.")

        # add softmax function at last layer.
        self.encoder_y.append(
            GumbelSoftmax(self.out_channels[-1], self.y_dim)
        )

        # encode latent vectors.
        if self.block_type == "Transformer":
            for index, item in enumerate(zip(self.in_channels, self.out_channels)):
                if index == 0:
                    self.encoder_z.append(
                        GraphTransformerBlock(
                            input_dim=item[0]+self.y_dim,
                            output_dim=item[1],
                            num_heads=self.num_heads,
                            use_bias=self.use_bias,
                        )
                    )
                else:
                    self.encoder_z.append(
                        GraphTransformerBlock(
                            input_dim=item[0],
                            output_dim=item[1],
                            num_heads=self.num_heads,
                            use_bias=self.use_bias,
                        )
                    )
        elif self.block_type == "GCN":
            for index, item in enumerate(zip(self.in_channels, self.out_channels)):
                if index == 0:
                    self.encoder_z.append(
                        GraphConvolutionBlock(
                            in_features=item[0]+self.y_dim,
                            out_features=item[1],
                            use_bias=self.use_bias,
                        )
                    )
                else:
                    self.encoder_z.append(
                        GraphConvolutionBlock(
                            in_features=item[0],
                            out_features=item[1],
                            use_bias=self.use_bias,
                        )
                    )
        else:
            raise NotImplementedError("Not implemented method, \
                only support Transformer and GCN currently.")
        
        # add reparameterize.
        self.encoder_z.append(
            Gaussian(self.out_channels[-1], self.z_dim)
        )

    def encode_y(self, x, temperature, hard, adj_matrix, mask_matrix=None):
        num_layers = len(self.encoder_y)
        for i, layer in enumerate(self.encoder_y):
            if i == num_layers - 1:
                logits, prob, y = layer(x, temperature, hard)
            else:
                if self.block_type == "Transformer":
                    x = layer(x, adj_matrix, mask_matrix)
                elif self.block_type == "GCN":
                    x = layer(x, adj_matrix)
                else:
                    raise NotImplementedError("Not implemented method, \
                        only support Transformer and GCN currently.")
        return logits, prob, y

    def encode_z(self, x, y, adj_matrix, mask_matrix):
        y = torch.unsqueeze(y, 0)
        concat = torch.cat((x, y), dim=2)
        num_layers = len(self.encoder_z)
        for i, layer in enumerate(self.encoder_z):
            if i == num_layers - 1:
                mu, sigma, z = layer(concat)
            else:
                if self.block_type == "Transformer":
                    concat = layer(concat, adj_matrix, mask_matrix)
                elif self.block_type == "GCN":
                    concat = layer(concat, adj_matrix)
                else:
                    raise NotImplementedError("Not implemented method, \
                        only support Transformer and GCN currently.")
        return mu, sigma, z
    
    def forward(self, x, adj_matrix, mask_matrix, temperature=1.0, hard=0):
        logits, prob, y = self.encode_y(
            x=x,
            adj_matrix=adj_matrix,
            mask_matrix=mask_matrix,
            temperature=temperature,
            hard=hard,
        )
        mu, var, z = self.encode_z(
            x=x,
            y=y,
            adj_matrix=adj_matrix,
            mask_matrix=mask_matrix,
        )
        return y, mu, var, z, logits, prob


class GraphDecoder(nn.Module):
    """Graph decoder which performs dot product to reconstruct the adj matrix based 
    on latent vector z and estimate the mean and variance of y.

    Args:
        y_dim (int): gaussain mixture number.
        z_dim (int): latent vector dimension.
    """
    def __init__(self, y_dim, z_dim):
        super(GraphDecoder, self).__init__()
        self.y_dim = y_dim
        self.z_dim = z_dim

        # p(z|y)
        self.y_mu = nn.Linear(self.y_dim, self.z_dim)
        self.y_var = nn.Linear(self.y_dim, self.z_dim)

    def forward(self, y, z):
        # tensor.t() only works on 2 dimensional tensor, need to squeeze firstly.
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        z = torch.squeeze(z)
        dot_product = torch.mm(z, z.t())
        reconstruct_graph = torch.sigmoid(dot_product)
        return y_mu, y_var, reconstruct_graph
