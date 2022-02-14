import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
from src.models.modules.gmgat import VAE
from torch.optim import Adam


class GMGATModel(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        latent_dim,
        dropout=0.1,
        num_blocks=2,
        use_bias=True,
        lr=0.0001,
        *args,
        **kwargs,
    ):
        """
        GMGATModel, inheriting the LightningModule, need to implement 
        the training_step, validation_step, test_step and optimizers.
        
        Args:
            in_channels (list): list of input dimension for each block.
            out_channels (list): list of output dimension for each block.
            num_heads (int): number of heads.
            latent_dim (int): dimension of latent embedding (mu, sigma).
            dropout (int): dropout ratio.
            num_blocks (int): number of transformer blocks.
            use_bias (boolean): whether to use bias in attention block.

        Attrs:
            GMGAT (VAE): trainable VAE model.
        """
        super().__init__()
        self.GMGAT = VAE(
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            latent_dim=latent_dim,
            dropout=dropout,
            num_blocks=num_blocks,
            use_bias=use_bias,
        )
        self.lr = lr

        # loss
        self.losses = LossFunction()
        
    def forward(self, graph, input):
        return self.GMGAT(graph, input)
        
    def training_step(self, batch, batch_idx):        
        attributes = batch["graph_attrs"]
        reconstruct, mu_estimate, sigma_estimate = self.GMGAT(batch)
        
        # TODO: add loss and logger.
        reconstruction_loss = self.losses.reconstruction_loss(
            ground_truth=attributes,
            reconstruct=reconstruct,
        )
        KL_loss = self.losses.KL_loss(
            mu_estimate=mu_estimate,
            sigma_estimate=sigma_estimate
        )
        loss = reconstruction_loss + KL_loss
        return {"loss": loss}

    def test_step(self):
        pass

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def on_epoch_end(self) -> None:
        # remove the cached accuracy here.
        return super().on_epoch_end()


class LossFunction:
    def __init__(self):
        self.l2_distance = nn.MSELoss()

    def reconstruction_loss(self, ground_truth, reconstruct):
        """Calculated the reconstruction loss between the ground truth
        graph attributes and reconstruction results from decoder.
        
        Args:
            ground_truth (tensor): ground truth of graph embedding.
            reconstruct (tensor): reconstruction of graph embedding.

        Returns:
            reconstuction_loss (tensor): reconstruction loss.
        """
        reconstruction_loss = self.l2_distance(ground_truth, reconstruct)
        return reconstruction_loss

    def KL_loss(self, mu_estimate, sigma_estimate):
        """ Calculated the KL divergence loss between normal distribution
        and estimated distribution from encoder.

        Args:
            mu_estimate (tensor): estimate mu from encoder.
            sigma_esimate (tenor): estimate log(sigma) from encoder.

        Returns:
            KL_loss (tensor): KL_loss.
        """
        
        KL_loss = -0.5 * (
            1 + sigma_estimate - mu_estimate.pow(2) - sigma_estimate.exp()
        ).sum(dim=1).mean()
        return KL_loss
