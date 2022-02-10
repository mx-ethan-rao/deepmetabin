import torch
import pytorch_lightning as pl
from datasets.graph_dataset import GraphDataset
from torch.utils.data import DataLoader, random_split


class GMGATDataModule(pl.LightningDataModule):
    def __init__(
        self,
        graph_dataset_roots,
        graph_attrs_dataset_roots,
        train_val_test_split,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
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
        self.graph_dataset_roots = graph_dataset_roots
        self.graph_attrs_dataset_roots = graph_attrs_dataset_roots
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def prepare_data(self):
        pass

    def setup(self):
        dataset = GraphDataset(
            graph_dataset_roots=self.graph_dataset_roots,
            graph_attrs_dataset_roots=self.graph_attrs_dataset_roots,
        )
        self.data_train, self.data_val, self.data_test = random_split(
            dataset=dataset,
            lengths=self.train_val_test_split,
            generator=torch.Generator().manual_seed(42),

        )

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
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
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
