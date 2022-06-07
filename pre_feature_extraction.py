import torch
from tqdm import tqdm, trange
from absl import app, flags
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.models.modules.modules import MFNet
from src.datamodules.datasets.graph_dataset import MatrixFactorizationDataset


class MatrixFactorizationManager:
    """MatrixFactorization Manager, which performs the matrix factorization ahead.
    Could treat ag graph and pe graph as user-item scores matrix.

    Args:
        zarr_dataset_path (string): path of processed zarr dataset.
        batch_size (int): batch size of training dataloader.
        model_save_path (string): path to save the shared matrix U.
        train_epochs (int): total training epoch num.
        feature_length (int): length of the output feature.
        learing_rate (float): learning rate.
        weight_decay (float): weight decay in optimizer.
        weight_ag_graph (float): weight in ag graph.
        weight_pe_graph (float): weight in pe graph.
    """
    def __init__(
        self,
        zarr_dataset_path: str,
        batch_size: int,
        model_save_path: str,
        train_epochs: int,
        feature_lengrh: int = 50,
        learning_rate: float = 0.00001,
        weight_decay: float = 0.0005,
        weight_ag_graph: float = 0.8,
        weight_pe_graph: float = 0.2,
    ):
        self.zarr_dataset_path = zarr_dataset_path
        self.batch_size = batch_size
        self.model_save_path = model_save_path
        self.train_epochs = train_epochs
        self.feature_length = feature_lengrh
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.weight_ag_graph = weight_ag_graph
        self.weight_pe_graph = weight_pe_graph
        self.node_num = 0
        assert int(self.weight_ag_graph + self.weight_pe_graph) == 1

    def get_dataloader(self):
        train_dataset = MatrixFactorizationDataset(zarr_dataset_path=self.zarr_dataset_path)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.node_num = train_dataset.node_num
        return train_dataloader

    def get_model(self):
        model = MFNet(N=self.node_num, K=self.feature_length)
        return model

    def train(self):
        model = self.get_model()
        dataloader = self.get_dataloader()
        opt = torch.optim.Adam(
            model.parameters(),
            self.learning_rate,
            weight_decay=self.weight_decay
        )

        print("##### Start the training! #####")
        for epoch in trange(self.train_epochs):
            model.train()
            total_loss, total_samples = 0, 0
            for batch in tqdm(dataloader, desc="training per batch......."):
                user_id = batch["user_id"]
                item_id = batch["item_id"]
                ag_value = batch["ag_value"].to(torch.float32)
                pe_value = batch["pe_value"].to(torch.float32)
                ag_pred, pe_pred = model(user_id, item_id)
                #TODO: further add the regulization term pi^2 and qj^2 to the loss function.
                ag_loss = F.mse_loss(ag_pred, ag_value)
                pe_loss = F.mse_loss(pe_pred, pe_value)
                total_samples += user_id.shape[0]
                loss = ag_loss * self.weight_ag_graph + pe_loss * self.weight_pe_graph
                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()

            model.eval()
            train_loss = total_loss / total_samples
            print("#### Epoch {}; train mse loss {};".format(epoch, train_loss))
        torch.save(model.U, self.model_path)


def main(argv=None):
    MFManager = MatrixFactorizationManager(**FLAGS.flag_values_dict())
    MFManager.train()


if __name__ == '__main__':
    FLAGS = flags.FLAGS

    # Details: Pls check MatrixFactorizationManager arguments.
    flags.DEFINE_string("zarr_dataset_path", "", "")
    flags.DEFINE_integer("batch_size", "", "")
    flags.DEFINE_string("model_save_path", "", "")
    flags.DEFINE_integer("train_epochs", "", "")
    flags.DEFINE_integer("feature_length", "", "")
    flags.DEFINE_float("learning_rate", "", "")
    flags.DEFINE_float("weight_decay", "", "")
    flags.DEFINE_float("weight_ag_graph", "", "")
    flags.DEFINE_float("weight_pe_graph", "", "")
    app.run(main)
