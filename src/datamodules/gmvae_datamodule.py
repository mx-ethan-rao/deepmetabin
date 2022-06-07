import torch
import pytorch_lightning as pl
from src.datamodules.datasets.graph_dataset import GraphDataset
from torch.utils.data import DataLoader, random_split


class GMVAEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        zarr_dataset_path,
        train_val_test_split,
        batch_size=1,
        sigma=1,
        num_workers=0,
        pin_memory=False,
        use_neighbor_feature=False,
        k=15,
        *args,
        **kwargs,
    ):
        """DataModule of GMGATModel, specify the dataloaders of
        metagenomic data.

        Args:
            graph_dataset_roots (list): list of the graph datasets path.
            graph_attrs_dataset_roots (list): list of the graph attrs 
                datasets path (bam file).
        """
        super().__init__()
        self.zarr_dataset_path = zarr_dataset_path
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.k = k
        self.use_neighbor_feature = use_neighbor_feature
        self.sigma = sigma

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        dataset = GraphDataset(
            zarr_dataset_path=self.zarr_dataset_path,
            k=self.k,
            sigma=self.sigma,
            use_neighbor_feature=self.use_neighbor_feature,
        )
        # self.data_train, self.data_val, self.data_test = random_split(
        #     dataset=dataset,
        #     lengths=self.train_val_test_split,
        #     generator=torch.Generator().manual_seed(42),
        # )
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
        # Modify here, use whole dataset for validating.
        # Use whole dataset to validating the performence.
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
