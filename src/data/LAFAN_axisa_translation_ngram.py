import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class PoseSequenceDataset(Dataset):
    def __init__(self, root_path, seq_len):
        files_names = [anim.name for anim in (Path(root_path) / "poses").iterdir()]
        poses = [np.load(Path(root_path) / "poses" / anim)["poses"] for anim in files_names]
        trans = [np.load(Path(root_path) / "trans_pelvis" / anim)["trans_pelvis"] for anim in files_names]

        end_indices = np.cumsum(np.array([len(i) for i in poses]))
        bad_indices = np.sort(np.concatenate([end_indices - i for i in range(1, seq_len)]))
        all_indices = np.arange(end_indices[-1])
        good_masks = np.isin(all_indices, bad_indices, assume_unique=True, invert=True)
        self.selectable_indices = torch.from_numpy(all_indices[good_masks])

        self.data = torch.from_numpy(np.concatenate([np.concatenate(poses), np.concatenate(trans)], axis=1)).float()
        self.data = torch.from_numpy(np.concatenate(poses)).float()
        
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.selectable_indices)

    def __getitem__(self, index):
        selectable_index = self.selectable_indices[index] 
        return self.data[selectable_index:selectable_index+self.seq_len].reshape(-1)