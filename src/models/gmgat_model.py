import torch
import torchmetrics
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
            VAE (VAE): trainable VAE model.
        """
        self.VAE = VAE(
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            latent_dim=latent_dim,
            dropout=dropout,
            num_blocks=num_blocks,
            use_bias=use_bias,
        )

    def forward(self, graph, input):
        return self.VAE(graph, input)
        
    def training_step(self, batch, batch_idx):
        graph, input = batch
        predicts = self.VAE(graph, input)
        # TODO: loss and logger.

    def test_step(self):
        pass

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)

    def on_epoch_end(self) -> None:
        # remove the cached accuracy here.
        return super().on_epoch_end()