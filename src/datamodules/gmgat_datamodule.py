import torch
import pytorch_lightning as pl
from src.datamodules.datasets.graph_dataset import GMGATSingleGraphDataset
from torch.utils.data import DataLoader, random_split


class GMGATDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_val_test_split,
        zarr_dataset_path: str = "",
        U_feature_path: str = "",
        batch_size: int = 1,
        k: int = 5,
        sigma: int = 1,
        num_workers=0,
        pin_memory=False,
        *args,
        **kwargs,
    ):
        """DataModule of GMGATModel, specify the dataloaders of
        metagenomic data.

        Args:
            train_val_test_split (list): train, test, val splitting
            zarr_dataset_path (string): processed zarr dataset path.
            U_feature_path (string): pre-extracted ag pe feature path.
            batch_size (int): batch size of data module.
            k (int): k parameter, stands for the batch size of data-module.
            sigma (float): sigma parameter, Gaussian variance when computing
                neighbors coefficient.
        """
        super().__init__()
        self.zarr_dataset_path = zarr_dataset_path
        self.U_feature_path = U_feature_path
        self.k = k
        self.sigma = sigma
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.k = k
        self.sigma = sigma
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        dataset = GMGATSingleGraphDataset(
            zarr_dataset_path=self.zarr_dataset_path,
            U_feature_path=self.U_feature_path,
            k=self.k,
            sigma=self.sigma,
        )
        self.data_train = dataset
        self.data_test = dataset
        self.data_val = dataset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=len(self.data_val),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
