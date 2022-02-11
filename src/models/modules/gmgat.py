import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.dim_per_head = output_dim // num_heads

        self.Q = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.K = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.V = nn.Linear(input_dim, output_dim, bias=use_bias)

    def propagate_graph_attention(self, Q_h, K_h, V_h, adj_matrix, mask_matrix):
        """ This function performs for the aggregation of graph network based on the
        output of single head Q_h, K_h, V_h, adjacent matrix, mask matrix.

        Args:
            Q_h (tensor): Querys, dimention is (B, N, D);
            K_h (tensor): Keys, dimention is (B, N, D);
            V_h (tensor): Values, dimention is (B, N, D);
            adj_matrix (tensor): normalized adjacent matirx, dimension is (B, N, N);
            mask_matrix (tensor): masked matrix, dimension is (B, N, N);

        Returns:
            h (tensor): calculation results of self-attention mechanisms.
        """
        K_h_transpose = torch.transpose(K_h, 1, 2) # dim B, D, N
        score = (adj_matrix * torch.matmul(Q_h, K_h_transpose) + mask_matrix) / np.sqrt(self.dim_per_head)
        attention_score = nn.Softmax(dim=2)(score) # dim B, N, N
        attention_output = torch.matmul(attention_score, V_h) # dim B, N, D
        return attention_output

    def forward(self, h, adj_matrix, mask_matrix):
        """forward function of Attention block.

        Args:
            h (tensor): input of graph tensor, dimension is (B, N, D);
            adj_matrix (tensor): normalized adjacent matrix;
            mask_matrix (tensor): masked matrix;

        Returns:
            output (tensor): result from the multi-head attention block.
        """
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        Q_multi_head_h = Q_h.view(-1, self.num_heads, self.dim_per_head) # dim B, N, H, D
        K_multi_head_h = K_h.view(-1, self.num_heads, self.dim_per_head) # dim B, N, H, D
        V_multi_head_h = V_h.view(-1, self.num_heads, self.dim_per_head) # dim B, N, H, D
        
        heads_output_list = []
        for i in range(self.num_heads):
            single_head_output = self.propagate_graph_attention(
                Q_h=Q_multi_head_h[:, :, i],
                K_h=K_multi_head_h[:, :, i],
                V_h=V_multi_head_h[:, :, i],
                adj_matrix=adj_matrix,
                mask_matrix=mask_matrix,
            ) # dim B, N, D
            heads_output_list.append(single_head_output)
        output = torch.cat(heads_output_list, dim=2) # dim B, N, D*H
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

        self.attention = GraphAttentionBlock(input_dim, output_dim, num_heads, use_bias)

        # optional
        self.projection_layer = nn.Linear(output_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, h, adj_matrix, mask_matrix):
        """forward function of transformer block, support one block of attention for now.
        
        Args:
            h (tensor): input of graph tensor, dimension is (B, N, D);
            adj_matrix (tensor): normalized adjacent matrix;
            mask_matrix (tensor): masked matrix;

        Returns:
            attention_out (tensor): result from transformer block.
        """
        attention_output = self.attention(h, adj_matrix, mask_matrix)
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
            self.decoder.append(GraphTransformerBlock(
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
            return latent

        def encode(self, h, adj_matrix, mask_matrix):
            latent = h
            for i in range(self.num_blocks):
                latent = self.encoder[i](latent, adj_matrix, mask_matrix)
            mu = self.mu_mlp(latent)
            sigma = self.sigma_mlp(latent)
            return mu, sigma

        def decode(self, resample, adj_matrix, mask_matrix):
            latent = resample
            for i in range(self.num_blocks):
                latent = self.decoder[i](latent, adj_matrix, mask_matrix)
            return latent

        def forward(self, batch):
            """Forward function of VAE model, which takes the batch item,
            a dictionary object as input, including graph attributes and 
            adjacency matrix and mask matrix.

            Args:
                batch (dictionary): includes three attributes:
                    "graph_attrs": graph vector (B, N, D);
                    "adj_matrix": adjacency matrix dim (B, N, N);
                    "mask_matrix": masked matrix (B, N, N);

            Returns:
                reconstruct (tensor): output from VAE model.
            """
            h = batch["graph_attrs"]
            adj_matrix = batch["adj_matrix"]
            mask_matrix = batch["mask_matrix"]
            mu_estimate, sigma_estimate = self.encode(
                h, adj_matrix, mask_matrix 
            )
            resample = self.reparameterize(mu_estimate, sigma_estimate)
            reconstruct = self.decode(resample)
            return reconstruct
