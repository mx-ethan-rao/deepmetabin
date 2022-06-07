import torch
import torch.nn as nn
import torch.nn.init as init
from src.models.modules.modules import (
    Encoder,
    GraphDecoder,
)


class VAE(nn.Module):
    """VAE module, main architecture of GMGAT.
        
    Args:
        encoder_in_channels (list): list of input dimension for encoder each block.
        encoder_out_channels (list): list of output dimension for encoder each block.
        num_heads (int): number of heads.
        y_dim (int): number of gaussain mixture distributions.
        latent_dim (int): dimension of latent embedding (mu, sigma).
        dropout (int): dropout ratio.
        num_blocks (int): number of transformer blocks.
        use_bias (boolean): whether to use bias in attention block.
        block_type (string): block type in encoder, support GCN and Transformer.

    Attrs:
        attention (GraphAttentionBlock): compute the graph embedding using
            attention mechanism.
        batch_norm (nn.BatchNorm1d): normalization.
    """
    def __init__(
        self,
        encoder_in_channels=[],
        encoder_out_channels=[],
        num_heads=16,
        y_dim=30,
        latent_dim=512,
        dropout=0.1,
        num_blocks=3,
        use_bias=True,
        block_type="Transformer",
    ):
        super(VAE, self).__init__()
        self.encoder_in_channels = encoder_in_channels
        self.encoder_out_channels = encoder_out_channels
        self.num_heads = num_heads
        self.y_dim = y_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.use_bias = use_bias
        self.block_type = block_type

        assert self.encoder_out_channels[-1] == self.latent_dim
        assert len(self.encoder_in_channels) == len(self.encoder_out_channels) == self.num_blocks
        self.encoder = Encoder(
            y_dim=self.y_dim,
            z_dim=self.latent_dim,
            in_channels=self.encoder_in_channels,
            out_channels=self.encoder_out_channels,
            block_type=self.block_type,
            use_bias=self.use_bias,
            num_heads=self.num_heads,
        )
        self.decoder = GraphDecoder(y_dim=self.y_dim, z_dim=self.latent_dim)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

    def encode(self, h, adj_matrix, mask_matrix, temperature=1, hard=0):
        logits, prob, y = self.encoder.encode_y(
            x=h,
            temperature=temperature,
            hard=hard,
            adj_matrix=adj_matrix,
            mask_matrix=mask_matrix,
        )
        mu, var, z = self.encoder.encode_z(
            x=h,
            y=y,
            adj_matrix=adj_matrix,
            mask_matrix=mask_matrix,
        )
        return mu, var, y, z, logits, prob

    def decode(self, y, z):
        y_mu, y_var, reconstruct_graph = self.decoder(y, z)
        return y_mu, y_var, reconstruct_graph

    def forward(self, h, adj_matrix, mask_matrix):
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
            mu_estimate (tensor): estimate mean vector.
            sigma_estimate (tensor): estimiate sigma vector.
        """
        mu, var, y, z, logits, prob = self.encode(
            h=h,
            adj_matrix=adj_matrix,
            mask_matrix=mask_matrix, 
        )
        y_mu, y_var, reconstruct_graph = self.decode(y, z)
        output_dict = {
            "mean": mu,
            "var": var,
            "gaussian": z,
            "logits": logits,
            "prob_cat": prob,
            "categorical": y,
            "y_mean": y_mu,
            "y_var": y_var,
            "reconstruct_graph": reconstruct_graph
        }
        return output_dict
