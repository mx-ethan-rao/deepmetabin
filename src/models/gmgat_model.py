import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import torch
import wandb
from torch.optim import Adam
from sklearn.mixture import GaussianMixture
from src.models.modules.gmgat import VAE
from src.models.losses import LossFunctions
from src.utils.util import (
    summary_bin_list_from_batch,
    get_long_contig_logits_from_batch,
    log_ag_graph,
    log_knn_graph,
    evaluate,
    log_tsne_figure,
    refine_gmm,
)


class GMGATModel(pl.LightningModule):
    """GMGATModel, inheriting the LightningModule, needs to implement the 
    training_step, validation_step, test_step and optimizers. To better 
    balance the ag-pe and knn graph, adopt GAN training strategy, alternat-
    -ing train ag graph encoder and knn graph encoder with different cross-
    entropy weights.
    
    Args:
        ag_encoder_in_channels (list): ag graph encoder input channels.
        ag_encoder_out_channels (list): ag graph encoder output channels.
        knn_encoder_in_channels (list): knn graph encoder input channels.
        knn_encoder_out_channels (list): knn graph encoder output channels.
        gaussian_size (int): size of latent space vector.
        num_classes (int): size of the gaussian mixture.
        lr (float): learning rate of the models.
        num_heads (int): number of heads in graph transformer block.
        num_blocks (int): number of encoder blocks.
        ag_w_cat (float): ag graph encoder categorical loss weight.
        ag_w_gauss (float): ag graph encoder gaussian loss weight.
        ag_w_rec (float): ag graph encoder reconstrcuction loss weight.
        ag_w_ce (float): ag graph encoder cross entropy loss weight.
        knn_w_cat (float): knn graph encoder categorical loss weight.
        knn_w_gauss (float): knn graph encoder gaussian loss weight.
        knn_w_rec (float): knn graph encoder reconstruction loss weight.
        knn_w_ce (float): knn graph encoder cross entropy loss weight.
        k (int): k neighbors used when plotting KNN graph.
        processed_zarr_dataset_path (string): path of processed zarr dataset,
            for logging the ag graph.
        block_type (string): graph encoder type, support 'GCN' and 'Transformer'.
        plot_graph_size (int): size of logging graph in wandb.
        log_path (string): path to save the logging result.
        use_gmm (boolean): whether use gmm fit the latent vector z.
        use_bias (True): whether use bias in models.
        dropout (float): dropout ratio in models.

    Attrs:
        AGGMGAT (VAE): ag graph encoder GMGAT model.
        KNNGMGAT (VAE): knn graph encoder GMGAT model.
    """
    def __init__(
        self,
        ag_encoder_in_channels=[50, 128, 256],
        ag_encoder_out_channels=[128, 256, 512],
        knn_encoder_in_channels=[104, 128, 256],
        knn_encoder_out_channels=[128, 256, 512],
        gaussian_size=512,
        num_classes=10,
        lr=0.0001,
        num_heads=16,
        num_blocks=3,
        ag_w_cat=None,
        ag_w_gauss=None,
        ag_w_rec=None,
        ag_w_ce=None,
        knn_w_cat=None,
        knn_w_gauss=None,
        knn_w_rec=None,
        knn_w_ce=None,
        processed_zarr_dataset_path=None,
        block_type="GCN",
        plot_graph_size=200,
        log_path="",
        k=5,
        use_gmm=False,
        use_bias=True,
        dropout=0.1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.AGGMGAT = VAE(
            encoder_in_channels=ag_encoder_in_channels,
            encoder_out_channels=ag_encoder_out_channels,
            num_heads=num_heads,
            y_dim=num_classes,
            latent_dim=gaussian_size,
            dropout=dropout,
            num_blocks=num_blocks,
            use_bias=use_bias,
            block_type=block_type,
        )
        self.KNNGMGAT = VAE(
            encoder_in_channels=knn_encoder_in_channels,
            encoder_out_channels=knn_encoder_out_channels,
            num_heads=num_heads,
            y_dim=num_classes,
            latent_dim=gaussian_size,
            dropout=dropout,
            num_blocks=num_blocks,
            use_bias=use_bias,
            block_type=block_type,
        )
        self.lr = lr
        self.ag_w_cat = ag_w_cat
        self.ag_w_gauss = ag_w_gauss
        self.ag_w_rec = ag_w_rec
        self.ag_w_ce = ag_w_ce
        self.knn_w_cat = knn_w_cat
        self.knn_w_gauss = knn_w_gauss
        self.knn_w_rec = knn_w_rec
        self.knn_w_ce = knn_w_ce
        self.k = k
        self.log_path = log_path
        self.losses = LossFunctions()
        self.processed_zarr_dataset_path = processed_zarr_dataset_path
        self.plot_graph_size = plot_graph_size
        self.use_gmm = use_gmm
        self.gmm = GaussianMixture(n_components=num_classes) if self.use_gmm else None
        self.automatic_optimization = False

    def forward(self, graph, input):
        pass
    
    def gmvae_loss(self, data, out_net, w_cat, w_gauss, w_rec):
        z, data_recon = out_net["gaussian"], out_net["reconstruct_graph"]
        logits, prob_cat = out_net["logits"], out_net["prob_cat"]
        y_mu, y_var = out_net["y_mean"], out_net["y_var"]
        mu, var = out_net["mean"], out_net["var"]

        loss_rec = self.losses.reconstruction_graph_loss(data, data_recon)
        loss_gauss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)
        loss_cat = -self.losses.entropy(logits, prob_cat) - np.log(0.1)
        loss_total = w_rec * loss_rec + w_gauss * loss_gauss + w_cat * loss_cat

        predicted_clusters = prob_cat.argmax(-1)
        highest_probs = prob_cat.max(-1).values
        loss_dict = {
            "total": loss_total,
            "predicted_clusters": predicted_clusters,
            "reconstruction": loss_rec * w_rec,
            "gaussian": loss_gauss,
            "categorical": loss_cat,
            "highest_prob": highest_probs,
            "logits": logits,
        }
        return loss_dict

    def training_step(self, batch, batch_idx):
        ag_opt, knn_opt = self.optimizers()        
        knn_attributes = batch["knn_feature"]
        ag_pe_attributes = batch["ag_pe_feature"]
        ag_adj_matrix = batch["ag_adj_matrix"]
        knn_adj_matrix = batch["knn_adj_matrix"]
        mask_matrix = batch["ag_mask_matrix"]
        ag_output_dict = self.AGGMGAT(
            h=ag_pe_attributes,
            adj_matrix=ag_adj_matrix,
            mask_matrix=mask_matrix,
        )
        knn_output_dict = self.KNNGMGAT(
            h=knn_attributes,
            adj_matrix=knn_adj_matrix,
            mask_matrix=mask_matrix,
        )

        ##########################
        # Optimize AGGMGAT #
        ##########################
        knn_logits = knn_output_dict["logits"].detach()
        long_contig_knn_logits = get_long_contig_logits_from_batch(batch, knn_logits)
        long_contig_ag_logits = get_long_contig_logits_from_batch(batch, ag_output_dict["logits"])
        ag_loss_dict = self.gmvae_loss(ag_adj_matrix, ag_output_dict, self.ag_w_cat, self.ag_w_gauss, self.ag_w_rec)
        ag_reconstruction_loss = ag_loss_dict["reconstruction"]
        ag_gaussian_loss = ag_loss_dict["gaussian"]
        ag_categorical_loss = ag_loss_dict["categorical"]
        ag_cross_entropy_loss = nn.CrossEntropyLoss()(
            long_contig_ag_logits,
            long_contig_knn_logits,
        )
        ag_loss = ag_loss_dict["total"] + self.ag_w_ce * ag_cross_entropy_loss
        ag_opt.zero_grad()
        self.manual_backward(ag_loss)
        ag_opt.step()

        ##########################
        # Optimize KNNGMGAT #
        ##########################
        ag_output_dict = self.AGGMGAT(
            h=ag_pe_attributes,
            adj_matrix=ag_adj_matrix,
            mask_matrix=mask_matrix,
        ) # use updated AGGMGAT to recompute the logits.
        ag_logits = ag_output_dict["logits"].detach()
        long_contig_ag_logits = get_long_contig_logits_from_batch(batch, ag_logits)
        knn_loss_dict = self.gmvae_loss(knn_adj_matrix, knn_output_dict, self.knn_w_cat, self.knn_w_gauss, self.knn_w_rec)
        knn_reconstruction_loss = knn_loss_dict["reconstruction"]
        knn_gaussian_loss = knn_loss_dict["gaussian"]
        knn_categorical_loss = knn_loss_dict["categorical"]
        knn_cross_entropy_loss = nn.CrossEntropyLoss()(
            long_contig_ag_logits,
            long_contig_knn_logits,
        )
        knn_loss = knn_loss_dict["total"] + self.knn_w_ce * knn_cross_entropy_loss
        knn_opt.zero_grad()
        self.manual_backward(knn_loss)
        knn_opt.step()

        # logging loss curve
        loss = ag_loss + knn_loss
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/ag_reconstruction_loss", ag_reconstruction_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/ag_gaussian_loss", ag_gaussian_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/ag_categorical_loss", ag_categorical_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/ag_crossentropy_loss", ag_cross_entropy_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/knn_reconstruction_loss", knn_reconstruction_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/knn_gaussian_loss", knn_gaussian_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/knn_categorical_loss", knn_categorical_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/knn_crossentorpy_loss", knn_cross_entropy_loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss}

    def test_step(self):
        # TODO: update test step function here.
        pass

    def validation_step(self, batch, batch_idx):
        #TODO: add the visualization of ag and knn two lines output.
        attributes = batch["knn_feature"]
        mask_matrix = batch["ag_mask_matrix"]
        knn_adj_matrix = batch["knn_adj_matrix"]
        output_dict = self.KNNGMGAT(
            h=attributes,
            adj_matrix=knn_adj_matrix,
            mask_matrix=mask_matrix,
        )
        prob_cat = output_dict["prob_cat"]
        latent = output_dict["gaussian"]
        bin_tensor = prob_cat.argmax(-1)
        gd_bin_list, result_bin_list, non_labeled_id_list = summary_bin_list_from_batch(batch, bin_tensor)

        # Compute metrics.
        precision, recall, ARI, F1 = evaluate(
            gd_bin_list=gd_bin_list,
            result_bin_list=result_bin_list,
            non_labeled_id_list=non_labeled_id_list,
            unclassified=0,
        )

        # logging graph for visualization.
        contig_id_list = [int(id) for index, id in enumerate(torch.squeeze(batch["id"]))]
        plotting_contig_list = contig_id_list[:self.plot_graph_size]
        gd_ag_graph_path, result_ag_graph_path = log_ag_graph(
            plotting_graph_size=self.plot_graph_size,
            processed_zarr_dataset_path=self.processed_zarr_dataset_path,
            plotting_contig_list=plotting_contig_list,
            log_path=self.log_path,
            gd_bin_list=gd_bin_list,
            result_bin_list=result_bin_list,
        )
        gd_knn_graph_path, result_knn_graph_path = log_knn_graph(
            plotting_graph_size=self.plot_graph_size,
            plotting_contig_list=plotting_contig_list,
            k=self.k,
            batch=batch,
            log_path=self.log_path,
            gd_bin_list=gd_bin_list,
            result_bin_list=result_bin_list,
        )

        # Visualize latent space.
        result_tsne_figure_path = log_tsne_figure(
            batch=batch,
            latent=torch.squeeze(latent),
            log_path=self.log_path,
        ) 
        self.log("val/acc", attributes.shape[0], on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/precision", precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/recall", recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/F1", F1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/ARI", ARI, on_step=False, on_epoch=True, prog_bar=False)
        wandb.log({"val/ground_truth_ag_subgraph": wandb.Image(gd_ag_graph_path)})
        wandb.log({"val/result_ag_subgraph": wandb.Image(result_ag_graph_path)})
        wandb.log({"val/ground_truth_knn_subgraph": wandb.Image(gd_knn_graph_path)})
        wandb.log({"val/result_knn_subgraph": wandb.Image(result_knn_graph_path)})
        wandb.log({"val/tsne_figure": wandb.Image(result_tsne_figure_path)})

    def configure_optimizers(self):
        ag_opt = Adam(self.AGGMGAT.parameters(), lr=self.lr)
        knn_opt = Adam(self.KNNGMGAT.parameters(), lr=self.lr)
        return ag_opt, knn_opt

    def on_epoch_end(self) -> None:
        # remove the cached accuracy here.
        return super().on_epoch_end()

    def log_gmm(self, batch, latent):
        """Use EM algorithm to further train the latent vector and predict the target cluster.

        Args:
            batch (dictionary): batch from datamodule to get the node labels.
            latent (tensor): z tensor output from the encoder.

        Returns:
            None
        """
        gmm_precision, gmm_recall, gmm_ARI, gmm_F1 = refine_gmm(
            gmm=self.gmm,
            batch=batch,
            latent=latent,
        )
        self.log("val/gmm_precision", gmm_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/gmm_recall", gmm_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/gmm_F1", gmm_F1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/gmm_ARI", gmm_ARI, on_step=False, on_epoch=True, prog_bar=False)
