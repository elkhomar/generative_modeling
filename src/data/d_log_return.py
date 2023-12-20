
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

# The dataset 

class log_returns_Dataset(Dataset):

    def __init__(self, root_path):
        df = pd.read_csv(Path(root_path), sep=',', header=None)
        tensor = torch.from_numpy(df.values).float()

        self.mean = tensor[:, 1:].mean(dim=0)
        self.std = tensor[:, 1:].std(dim=0)
        self.data = tensor[:, 1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index].reshape(-1)
    
class DataModule(LightningDataModule):
    """Example preprocessing and batching poses

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_split: Tuple[int, int] = (0.8, 0.2),
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        normalize=True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data = log_returns_Dataset(data_dir + "data_train_log_return.csv")
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

        self.train_val_split = train_val_split

    @property
    def n_features(self):
        return 4

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        if not self.data_train and not self.data_val:
            # Training and Val set
            self.data_train, self.data_val = random_split(
                dataset= self.data,
                lengths= [int(self.train_val_split[0]*len(self.data)) + 1, int(self.train_val_split[1]*len(self.data))],
                generator=torch.Generator().manual_seed(42),
            )
        
    def train_dataloader(self, shuffle=True):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
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
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    def get_pose_from_model(self, index):
        return self.data_train.model_output_to_pose(index)


if __name__ == "__main__":
    _ = DataModule()
    dataset = log_returns_Dataset(
        "data/financial_data/data_train_log_return.csv"
    )
    print(dataset.get_pose(0))
