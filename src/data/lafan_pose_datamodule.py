
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split


import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

# The dataset 
class PoseSequenceDataset(Dataset):
    def __init__(self, root_path, seq_len):
        files_names = [anim.name for anim in sorted((Path(root_path) / "poses").iterdir())]
        poses = [np.load(Path(root_path) / "poses" / anim)["poses"] for anim in files_names]

        end_indices = np.cumsum(np.array([len(i) for i in poses]))
        bad_indices = np.sort(np.concatenate([end_indices - i for i in range(1, seq_len)]))
        all_indices = np.arange(end_indices[-1])
        good_masks = np.isin(all_indices, bad_indices, assume_unique=True, invert=True)
        self.selectable_indices = torch.from_numpy(all_indices[good_masks])

        self.data = torch.from_numpy(np.concatenate(poses)).float()
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.selectable_indices)

    def __getitem__(self, index):
        selectable_index = self.selectable_indices[index] 
        return self.data[selectable_index:selectable_index+self.seq_len].reshape(-1)
    
    # Returns data as a pose dictionnary (first pose of the sequence fixed at pelvis position 0)
    @staticmethod
    def model_output_to_pose(y):
        '''
        converts a list of feature vectors [n_frames, n_features] into an animation dictionnary {trans_pelvis : tensor[n_frames, 3], poses : tensor[n_frames, 24, 3]}
        '''
        n_frames = y.shape[0]
        pose = {"trans_pelvis" : torch.tensor([0, 0, 0]).repeat(n_frames, 1), "poses" : y[:, :72].reshape(n_frames, 24, 3)}
        return pose
    

class LAFANDataModule(LightningDataModule):
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
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        seq_len: int = 2,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_check: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def n_features(self):
        return self.hparams.seq_len * 72

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).

        Here, we don't need the data to be downloaded, so we can leave it empty.
        """

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # frame_dim = 72 (24 joints * 3 coordinates) => frame_dim * seq_len = 144

        if not self.data_train and not self.data_val and not self.data_test and not self.data_check:
            # Training set
            self.data_train = PoseSequenceDataset(self.hparams.data_dir + 'LAFAN/train/', seq_len=self.hparams.seq_len)
            self.data_check = PoseSequenceDataset(self.hparams.data_dir + 'LAFAN/val/', seq_len=self.hparams.seq_len)

            # Validation and Test set
            self.data_val, self.data_test = random_split(
                dataset= self.data_check, 
                lengths= [int((1 + len(self.data_check))/2), int(len(self.data_check)/2)],
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
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def check_dataloader(self):
        return DataLoader(
            dataset=self.data_check,
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
    
    _ = LAFANDataModule()

    dataset = PoseSequenceDataset("lightning-hydra-template/data/LAFAN/train/", 2)
    print(dataset.get_pose(0))
