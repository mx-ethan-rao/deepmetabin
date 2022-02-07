from re import M
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np


class GraphAttentionBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        use_bias=True,
    ):
        """ Attention block performed on graph.
        
        Args:
            input_dim (int): input dimension of embedding.
            output_dim (int): output dimension of embedding.
            num_heads (int): number of heads.
            use_bias (boolean): whether to use bias in attention block.

        Attrs:
            Q (nn.Linear): Query matrix.
            K (nn.Linear): Keys matrix.
            V (nn.Linear): Values matrix.
        """
        super().__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.Q = nn.Linear(input_dim, output_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(input_dim, output_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(input_dim, output_dim * num_heads, bias=use_bias)

    def propagate_graph_attention(self, g):
        """ This function performs for the aggregation of graph network, use dgl 
        implementation for now, could be substituded.
        Args:
            g (dgl graph object): graph g to compute attention.
        """
        g.apply_edges('K_h', 'Q_h', 'score')
        g.apply_edges('score', np.sqrt(self.output_dim))

        edges = g.edges()
        g.send_and_recv(edges, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))


    def forward(self, g, h):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.output_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.output_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.output_dim)

        self.propagate_graph_attention(g)
        output = g.ndata['wV']

        return output


class GraphTransformerBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        use_bias=True,
    ):
        """ GraphTransformer block integrated attention.
        TODO: could further add resdual strategy in transformer block.

        Args:
            input_dim (int): input dimension of embedding.
            output_dim (int): output dimension of embedding.
            num_heads (int): number of heads.
            use_bias (boolean): whether to use bias in attention block.

        Attrs:
            attention (GraphAttentionBlock): compute the graph embedding using
                attention mechanism.
            batch_norm (nn.BatchNorm1d): normalization.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.attention = GraphAttentionBlock(input_dim, output_dim//num_heads, num_heads, use_bias)

        # optional
        self.projection_layer = nn.Linear(output_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, g, h):
        attention_output = self.attention(g, h).view(-1, self.output_dim)
        attention_output = F.relu(attention_output)
        attention_output = self.batch_norm(attention_output)
        return attention_output


class VAE(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        latent_dim,
        dropout=0.1,
        num_blocks=2,
        use_bias=True,
    ):
        """ VAE module, main architecture of GMGAT.

        Args:
            in_channels (list): list of input dimension for each block.
            out_channels (list): list of output dimension for each block.
            num_heads (int): number of heads.
            latent_dim (int): dimension of latent embedding (mu, sigma).
            dropout (int): dropout ratio.
            num_blocks (int): number of transformer blocks.
            use_bias (boolean): whether to use bias in attention block.

        Attrs:
            attention (GraphAttentionBlock): compute the graph embedding using
                attention mechanism.
            batch_norm (nn.BatchNorm1d): normalization.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.use_bias = use_bias

        assert len(self.in_channels) == len(self.out_channels) == self.num_blocks
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.mu_mlp = nn.Linear(self.out_channels[-1], self.latent_dim)
        self.sigma_mlp = nn.Linear(self.out_channels[-1], self.latent_dim)
        self.softplus = nn.Softplus()

        for in_dim, out_dim in zip(self.in_channels, self.out_channels):
            self.encoder.append(GraphTransformerBlock(
                input_dim=in_dim,
                output_dim=out_dim,
                num_heads=self.num_heads,
                use_bias=self.use_bias,
            ))

        for in_dim, out_dim in zip(self.out_channels[::-1], self.in_channels[::-1]):
            self.decoder.append(GraphAttentionBlock(
                input_dim=in_dim,
                output_dim=out_dim,
                num_heads=self.num_heads,
                use_bias=self.use_bias,
            ))

        def reparameterize(self, mu, sigma):
            """reparameterize function.

            Args:
                mu (tensor): stands for the mean from mu_mlp.
                sigma (tensor): stands for the log of sigma^2.
            """
            epsilon = torch.randn(mu.size(0), mu.size(1)).type_as(mu)
            latent = mu + epsilon * torch.exp(sigma/2)

        def encode(self, gragh, input):
            latent = self.encoder(gragh, input)
            mu = self.mu_mlp(latent)
            sigma = self.sigma_mlp(latent)
            return mu, sigma

        def decode(self, latent):
            return self.decoder(latent)

        def forward(self, graph, input):
            mu_estimate, sigma_estimate = self.encode(graph, input)
            resample = self.reparameterize(mu_estimate, sigma_estimate)
            reconstruct = self.decode(resample)
            return reconstruct
