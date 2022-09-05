import wandb
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from torch.optim import Adam
from src.models.modules.vae import VAENet, LossFunctions
from src.utils.util import(
    log_tsne_figure,        
)


class VAEModel(pl.LightningModule):
    def __init__(
        self,
        encoder_input_dim=104,
        encoder_hidden_dim=512,
        encoder_latent_dim=32,
        decoder_latent_dim=32,
        decoder_hidden_dim=512,
        decoder_output_dim=104,
        log_path="/home/eddie/log",
        w_rec=1,
        w_kl=0.000156,
        lr=1e-4,
        *args,
        **kwargs,
    ):
        super(VAEModel, self).__init__()
        self.network = VAENet(
            encoder_input_dim=encoder_input_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_latent_dim=encoder_latent_dim,
            decoder_latent_dim=decoder_latent_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_output_dim=decoder_output_dim,
        )
        self.lr = lr
        self.w_rec = w_rec
        self.w_kl = w_kl
        self.losses = LossFunctions(
            w_rec=self.w_rec,
            w_kl=self.w_kl,
        )
        self.log_path = log_path
        
    def forward(self):
        pass
        
    def training_step(self, batch, batch_idx):
        x = batch["feature"]
        latent, x_hat, mean, log_var = self.network(x)
        rec_loss, kl_loss, loss = self.losses.cal_loss(
            x=x,
            x_hat=x_hat,
            mean=mean,
            log_var=log_var,
        )
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/reconstruction_loss", rec_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/kl_loss", kl_loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss}
    
    def test_step(self):
        pass
    
    def validation_step(self, batch, batch_idx):
        attributes = batch["feature"]
        latent, x_hat, mean, log_var = self.network(attributes)
        self.log("val/acc", attributes.shape[0], on_step=False, on_epoch=True, prog_bar=False)
        
        # Visualize latent space:
        result_tsne_figure_path = log_tsne_figure(
            batch=batch,
            latent=latent,
            log_path=self.log_path,
        )
        wandb.log({"val/tsne_figure": wandb.Image(result_tsne_figure_path)})

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
 
    def on_epoch_end(self) -> None:
        # remove the cached accuracy here.
        return super().on_epoch_end()
