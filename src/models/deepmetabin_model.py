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
    generate_csv_from_bin_tensor,
)


class DeepMetaBinModel(pl.LightningModule):
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
        csv_path="",
        k=5,
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
            csv_path (string): path to save the binning result csv.
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
        self.csv_path = csv_path
        self.use_gmm = use_gmm
        self.gmm = GaussianMixture(n_components=num_classes) if self.use_gmm else None
        self.sota_precision = 0
        self.sota_recall = 0
        self.sota_F1 = 0
        self.sota_ARI = 0


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
        weights = batch["weights"].squeeze()
        
        ag_based_neighbor_attributes = batch["ag_based_neighbors_feature"].squeeze()
        ag_based_weights = batch["ag_based_knn_weights"].squeeze()
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
        loss_ag_rec_neigh = 0
        
        # Reconstruct knn based neighbor features.
        for i in range(self.k):
            nei_feat = neighbor_attributes[:, i]
            nei_weight = weights[:, i]
            rec_nei = self.network(nei_feat)["x_rec"]
            rec_loss = self.losses.reconstruction_loss(attributes, rec_nei)
            loss_rec_neigh += (rec_loss * nei_weight).mean()
        loss_rec_neigh /= self.k
        
        for i in range(self.k):
            ag_nei_feat = ag_based_neighbor_attributes[:, i]
            ag_nei_weight = ag_based_weights[:, i]
            ag_rec_nei = self.network(ag_nei_feat)["x_rec"]
            ag_rec_loss = self.losses.reconstruction_loss(attributes, ag_rec_nei)
            loss_ag_rec_neigh += (ag_rec_loss * ag_nei_weight).mean()
        loss_ag_rec_neigh /= self.k
        
        self.log("train/rec_neigh_loss", loss_rec_neigh, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/rec_ag_neigh_loss", loss_ag_rec_neigh, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "loss_rec_neighor": loss_rec_neigh, "loss_ag_rec_neighbor": loss_ag_rec_neigh}
    
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
        precision, recall, ARI, F1 = evaluate(
            gd_bin_list=gd_bin_list,
            result_bin_list=result_bin_list,
            non_labeled_id_list=non_labeled_id_list,
            unclassified=0
        )

        # plotting graph for visualization here.
        contig_id_list = [int(id) for index, id in enumerate(batch["id"])]
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
            log_path=self.log_path,
            k=self.k,
            batch=batch,
            gd_bin_list=gd_bin_list,
            result_bin_list=result_bin_list,
        )

        # add gmm to the latent vector, wrap a function here.
        if self.use_gmm:
            self.log_gmm(batch, latent)

        # Visualize latent space.
        result_tsne_figure_path = log_tsne_figure(
            batch=batch,
            latent=latent,
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
        if gmm_ARI > self.sota_ARI or gmm_F1 > self.sota_F1:
            self.sota_precision = gmm_precision
            self.sota_recall = gmm_recall
            self.sota_F1 = gmm_F1
            self.sota_ARI = gmm_ARI
            suffix = "precision-{}-recall-{}-F1-{}-ARI-{}.csv".format(
                gmm_precision,
                gmm_recall,
                gmm_F1,
                gmm_ARI,
            )
            output_csv_path = self.csv_path + suffix
            generate_csv_from_bin_tensor(
                output_csv_path=output_csv_path,
                id_tensor=batch["id"],
                results=predicts,)
        self.log("val/gmm_precision", gmm_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/gmm_recall", gmm_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/gmm_F1", gmm_F1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/gmm_ARI", gmm_ARI, on_step=False, on_epoch=True, prog_bar=False)
