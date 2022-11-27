from operator import ne
import wandb
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from torch.optim import Adam
from sklearn.mixture import GaussianMixture
from src.models.modules.gmvae import GMVAENet
from src.models.losses import LossFunctions
from src.utils.util import (
    summary_bin_list_from_batch, 
    log_ag_graph,
    log_knn_graph,
    log_tsne_figure,
    evaluate,
    refine_gmm,
)


class DeepBinModel(pl.LightningModule):
    def __init__(
        self,
        input_size=104,
        gaussian_size=10,
        num_classes=10,
        lr=0.0001,
        w_cat=None,
        w_gauss=None,
        w_rec=None,
        rec_type="mse",
        processed_zarr_dataset_path=None,
        plot_graph_size=200,
        log_path="",
        k=5,
        latent_save_path="",
        use_gmm=True,
        *args,
        **kwargs,
    ):
        """DeepModel, inheriting the LightningModule, need to implement the training_step,
        validation_step, test_step and optimizers, different from the GMVAEModel, including
        neighbors feature reconstruction loss.

        Args:
            input_size (int): size of input vector, 104 in current settings.
            gaussian_size (int): size of latent space vector.
            num_classes (int): size of the gaussian mixture.
            lr (float): learning rate of the models.
            k (int): k neighbors used when plotting KNN graph.
            w_cat (float): categorical loss weights.
            w_gaussian (float): gaussian loss weights.
            w_rec (float): reconstruction loss weights.
            rec_type (string): reconstruction loss type, supporting 'mse', 'bce' currently.
            processed_zarr_dataset_path (string): path of processed zarr dataset,
                for logging the ag graph.
            plot_graph_size (int): size of logging graph in wandb.
            log_path (string): path to save the logging result.
            use_gmm (boolean): whether use gmm fit the latent vector z.

        Attrs:
        """
        super().__init__()
        self.network = GMVAENet(
            x_dim = input_size,
            z_dim = gaussian_size,
            y_dim = num_classes,
        )
        self.lr = lr
        self.w_cat = w_cat
        self.w_gauss = w_gauss
        self.w_rec = w_rec
        self.rec_type = rec_type
        self.k = k
        self.losses = LossFunctions()
        self.processed_zarr_dataset_path = processed_zarr_dataset_path
        self.plot_graph_size = plot_graph_size
        self.log_path = log_path
        self.latent_save_path = latent_save_path
        self.use_gmm = use_gmm
        self.gmm = GaussianMixture(n_components=num_classes, random_state=2021) if self.use_gmm else None
        self.count = 0
        self.epoch_list = []

    def unlabeled_loss(self, data, out_net):
        z, data_recon = out_net["gaussian"], out_net["x_rec"]
        logits, prob_cat = out_net["logits"], out_net["prob_cat"]
        y_mu, y_var = out_net["y_mean"], out_net["y_var"]
        mu, var = out_net["mean"], out_net["var"]

        loss_rec = self.losses.reconstruction_loss(data, data_recon)
        loss_gauss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)
        loss_cat = -self.losses.entropy(logits, prob_cat) - np.log(0.1)
        loss_total = self.w_rec * loss_rec + self.w_gauss * loss_gauss + self.w_cat * loss_cat

        predicted_clusters = prob_cat.argmax(-1)
        highest_probs = prob_cat.max(-1).values
        loss_dict = {
            "total": loss_total,
            "predicted_clusters": predicted_clusters,
            "reconstruction": loss_rec * self.w_rec,
            "gaussian": loss_gauss,
            "categorical": loss_cat,
            "highest_prob": highest_probs,
        }
        return loss_dict

    def forward(self):
        pass
        
    def training_step(self, batch, batch_idx):
        attributes = batch["feature"]
        neighbor_attributes = batch["neighbors_feature"].squeeze()
        neighbors_mask = batch["neighbors_feature_mask"].squeeze()
        neighbors_weight = batch["neighbors_weight"].squeeze()
        out_net = self.network(attributes)
        loss_dict = self.unlabeled_loss(attributes, out_net)

        loss = loss_dict["total"]
        reconstruction_loss = loss_dict["reconstruction"]
        gaussian_loss = loss_dict["gaussian"]
        categorical_loss = loss_dict["categorical"]
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/reconstruction_loss", reconstruction_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/gaussian_loss", gaussian_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/categorical_loss", categorical_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        loss_rec_neigh = 0
        for i in range(self.k):
            nei_feat = neighbor_attributes[:, i]
            nei_mask = neighbors_mask[:, i]
            nei_weight = neighbors_weight[:, i]
            # origin feature based neighbor reconstruction.
            rec_nei = self.network(nei_feat)["x_rec"]
            rec_loss = self.losses.reconstruction_loss_by_dim(attributes, rec_nei, nei_mask, nei_weight)
            loss_rec_neigh += rec_loss

        loss_rec_neigh = loss_rec_neigh / self.k
        loss += loss_rec_neigh
        self.log("train/rec_neigh_loss", loss_rec_neigh, on_step=False, on_epoch=True, prog_bar=False)
        self.count += 1
        return {"loss": loss, "loss_rec_neighor": loss_rec_neigh}
    
    def test_step(self):
        pass
    
    def validation_step(self, batch, batch_idx):
        attributes = batch["feature"]
        out_net = self.network(attributes)
        prob_cat = out_net["prob_cat"]
        latent = out_net["gaussian"]
        bin_tensor = prob_cat.argmax(-1)
        gd_bin_list, result_bin_list, non_labeled_id_list = summary_bin_list_from_batch(batch, bin_tensor)

        # Computing metrics here.
        # precision, recall, ARI, F1 = evaluate(
        #     gd_bin_list=gd_bin_list,
        #     result_bin_list=result_bin_list,
        #     non_labeled_id_list=non_labeled_id_list,
        #     unclassified=0,
        # )

        # # plotting graph for visualization here.
        # contig_id_list = [int(id) for index, id in enumerate(batch["id"])]
        # plotting_contig_list = contig_id_list[:self.plot_graph_size]
        """
        gd_ag_graph_path, result_ag_graph_path = log_ag_graph(
            plotting_graph_size=self.plot_graph_size,
            processed_zarr_dataset_path=self.processed_zarr_dataset_path,
            plotting_contig_list=plotting_contig_list,
            log_path=self.log_path,
            graph_type="origin",
            gd_bin_list=gd_bin_list,
            result_bin_list=result_bin_list,
        )
        gd_knn_graph_path, result_knn_graph_path = log_knn_graph(
            plotting_graph_size=self.plot_graph_size,
            plotting_contig_list=plotting_contig_list,
            log_path=self.log_path,
            k=self.k,
            graph_type="origin",
            batch=batch,
            gd_bin_list=gd_bin_list,
            result_bin_list=result_bin_list,
        )
        """
        if self.current_epoch < 100:
            self.use_gmm = False
            self.log("val/gmm_F1", -1, on_step=False, on_epoch=True, prog_bar=False)
        else:
            self.use_gmm = True
        # add gmm to the latent vector, wrap a function here.
        if self.use_gmm:
            self.log_gmm(
                batch=batch,
                latent=latent
            )

        # Visualize latent space.
        # result_tsne_figure_path = log_tsne_figure(
        #     batch=batch,
        #     latent=latent,
        #     log_path=self.log_path,
        # )
        
        # Save latent.
        latent_save_path = "{}/latent_{}_{}".format(self.latent_save_path, self.current_epoch, self.global_step)
        latent_feature = latent.numpy()
        np.save(latent_save_path, latent_feature)

        # self.log("val/acc", attributes.shape[0], on_step=False, on_epoch=True, prog_bar=False)
        # self.log("val/precision", precision, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("val/recall", recall, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("val/F1", F1, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("val/ARI", ARI, on_step=False, on_epoch=True, prog_bar=False)
  
        # wandb.log({"val/tsne_figure": wandb.Image(result_tsne_figure_path)})
        

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
 
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
        predicts, gmm_precision, gmm_recall, gmm_ARI, gmm_F1 = refine_gmm(
                gmm=self.gmm,
                batch=batch,
                latent=latent,
        )
        """
        _, gmm_result_ag_graph_path = log_ag_graph(
            plotting_graph_size=self.plot_graph_size,
            processed_zarr_dataset_path=self.processed_zarr_dataset_path,
            plotting_contig_list=plotting_contig_list,
            log_path=self.log_path,
            graph_type="gmm",
            gd_bin_list=gmm_gd_bin_list,
            result_bin_list=gmm_result_bin_list,
        )
        _, gmm_result_knn_graph_path = log_knn_graph(
            plotting_graph_size=self.plot_graph_size,
            plotting_contig_list=plotting_contig_list,
            log_path=self.log_path,
            k=self.k,
            batch=batch,
            graph_type="gmm",
            gd_bin_list=gmm_gd_bin_list,
            result_bin_list=gmm_result_bin_list,
        )
        gmm_precision, gmm_recall, gmm_ARI, gmm_F1 = evaluate(
            gd_bin_list=gmm_gd_bin_list,
            result_bin_list=gmm_result_bin_list,
            non_labeled_id_list=gmm_non_labeled_id_list,
            unclassified=0
        )
        """
        self.log("val/gmm_precision", gmm_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/gmm_recall", gmm_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/gmm_F1", gmm_F1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/gmm_ARI", gmm_ARI, on_step=False, on_epoch=True, prog_bar=False)
        self.if_stop(gmm_F1, self.current_epoch, self.global_step)
        #wandb.log({"val/result_gmm_ag_subgraph": wandb.Image(gmm_result_ag_graph_path)})
        #wandb.log({"val/result_gmm_knn_subgraph": wandb.Image(gmm_result_knn_graph_path)})

    def if_stop(self, f1, current_epoch, global_step):
        patience = 33
        self.epoch_list.append((f1, current_epoch, global_step))
        current_best = max(self.epoch_list, key=lambda elem: elem[0])
        best_idx = self.epoch_list.index(current_best)
        if len(self.epoch_list) - best_idx - 1 >= patience:
            import os
            os.rename("{}/latent_{}_{}.npy".format(self.latent_save_path, current_best[1], current_best[2]), \
                "{}/latent_{}_{}_best.npy".format(self.latent_save_path, current_best[1], current_best[2]))
            import sys
            sys.exit(0)

