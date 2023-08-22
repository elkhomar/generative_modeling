from typing import Any, Dict, Optional, Tuple
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import kinetix_scenegraph.utils.rotation_conversions as kin
from torch.nn.functional import normalize


import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import kinetix_scenegraph as ks
from kinetix_scenegraph.skeletons.skeleton import Skeleton
import os

# The dataset 
class PoseSequenceDataset(Dataset):

# Static variables, only initialized on the training set and reused for the other datasets
    train_mean = None
    train_std = None

    train_max = None
    train_min = None

    def __init__(self, root_path, seq_len, training=False):
        files_names = [anim.name for anim in sorted((Path(root_path) / "poses").iterdir())]
        # Retrieve the poses and translation from the files excluding the T-poses
        poses = [np.load(Path(root_path) / "poses" / anim)["poses"][150:] for anim in files_names]
        trans = [np.load(Path(root_path) / "trans_pelvis" / anim)["trans_pelvis"][150:] for anim in files_names]
        ground_contacts = [np.load(Path(root_path) / "ground_contacts" / anim)["ground_contacts"][150:] for anim in files_names]
        self.fps = 60
        self.normalize_eps = 1e-8
        self.training  = training
        self.end_indices = np.cumsum(np.array([len(i) for i in poses]))
        bad_indices = np.sort(np.concatenate([self.end_indices - i for i in range(0, seq_len)]))
        bad_indices = np.concatenate((np.array([0.0]), bad_indices))
        all_indices = np.arange(self.end_indices[-1])
        good_masks = np.isin(all_indices, bad_indices, assume_unique=True, invert=True)
        self.selectable_indices = torch.from_numpy(all_indices[good_masks])

        ### Store the ground contacts
        self.ground_contacts = torch.from_numpy(np.concatenate(ground_contacts)).float()

        self.axisa = torch.from_numpy(np.concatenate(poses)).float()
        self.seq_len = seq_len

        ### Computing the projected velocity
        # First compute the projected translations
        self.trans = torch.from_numpy(np.concatenate(trans)).float()        
        self.projected_trans = self.trans[:, 0:3:2]
        # Compute the velocity
        dt = 1/self.fps
        self.velocity = (self.projected_trans - self.projected_trans.roll(1, 0))/dt
        # Set the first velocity of each animation to the next one
        self.wrong_velocity_idx = np.concatenate((np.array([0.0]), self.end_indices[:-1]))
        self.velocity[self.wrong_velocity_idx] = self.velocity[self.wrong_velocity_idx + np.ones_like(self.wrong_velocity_idx)]
        
        ### Compute joint specific data and angular velocity
        self.initialize_keypoints()

        # Assemble and normalize
        self.data = torch.cat([self.velocity, self.angular_velocity, self.keypoints[:, 0].unsqueeze(1), self.ground_contacts, self.keypoints[:, 1:], self.keypoint_velocities, self.rot6d], dim=1)
        self.n_features = self.data.shape[1]
        #self.data = self.normalize(self.data)
        

    def normalize(self, data):
        if(self.training == True):
            type(self).train_max = torch.max(data, dim=0).values
            type(self).train_min = torch.min(data, dim=0).values

        maxx = type(self).train_max
        minn = type(self).train_min

        normalized_data =  2*(data - minn)/pow((pow((maxx - minn), 2) + self.normalize_eps), 0.5) - 1
        return normalized_data

    def denormalize(self, data):
        maxx = type(self).train_max
        minn = type(self).train_min

        denormalized_data = (data + 1)*pow((pow((maxx - minn), 2) + self.normalize_eps), 0.5)/2 + minn
        return denormalized_data
    

    def standardize(self, data):
        # If training set the compute the mean and std that will be used for normalization of all datasets
        # Training dataset should then be called before the validation and test dataset
        if(self.training == True):
            type(self).train_mean = torch.mean(data, dim=0)
            type(self).train_std = torch.std(data, dim=0)

        mean = type(self).train_mean
        std = type(self).train_std
        standardized_data =  (data - mean)/pow((pow(std, 2) + self.normalize_eps), 0.5)
        return standardized_data
    
    def destandardize(self, data):
        mean = type(self).train_mean
        std = type(self).train_std
        destandardized_data = data*pow((pow(std, 2) + self.normalize_eps), 0.5) + mean
        return destandardized_data
    

    def __len__(self):
        return len(self.selectable_indices)

    def __getitem__(self, index):
        selectable_index = self.selectable_indices[index]
        if(type(index) == int):
            return self.data[selectable_index:selectable_index+self.seq_len].reshape(-1)
        else:
            sliced = torch.cat([self.data[idx:idx+self.seq_len].reshape(-1).unsqueeze(0) for idx in selectable_index])
            return sliced
        
    def initialize_keypoints(self):

        def unroll_euler_1d(angles):  
            """  
            angles : shape (batch_size, time)  
            """  
            positive_changes = torch.where(angles[..., 1:] - angles[..., :-1] < -np.pi, torch.ones_like(angles[..., :-1]), torch.zeros_like(angles[..., :-1]))  
            negative_changes = torch.where(angles[..., 1:] - angles[..., :-1] > np.pi, torch.ones_like(angles[..., :-1]), torch.zeros_like(angles[..., :-1]))  
            all_changes = positive_changes - negative_changes  
            cumsum = torch.zeros_like(angles)  
            cumsum[..., 1:] = torch.cumsum(all_changes, dim=-1)  
            return angles + cumsum * 2 * np.pi
    
        # Uses the kinetix scene graph to retrieve the 3d keypoints
        self.Kinetix_without_hands_positions = torch.Tensor(
        [[-1.43760722e-03, 9.54190561e-01, 2.13284530e-02],
        [5.45075648e-02, 8.74072471e-01, -5.23403287e-06],
        [1.00507133e-01, 4.85931197e-01, 8.22260603e-03],
        [8.78663585e-02, 7.30262800e-02, -2.690302765e-02],
        [1.27022058e-01, 1.53466300e-02, 8.94109830e-02],
        [-5.92250526e-02, 8.66445789e-01, 3.26549634e-03],
        [-1.03564493e-01, 4.84292308e-01, -1.36650261e-03],
        [-8.57003257e-02, 7.84770000e-02, -3.48879360e-02],
        [-1.18686341e-01, 1.92904300e-02, 8.95627439e-02],
        [2.59872526e-03, 1.07332193e+00, -1.18642114e-02],
        [6.49475120e-03, 1.20880211e+00, 1.62070766e-02],
        [4.68270201e-03, 1.26314347e+00, 1.89546309e-02],
        [-7.39822537e-03, 1.45910470e+00, -8.57547671e-03],
        [3.41666304e-03, 1.54568098e+00, 4.00794521e-02],
        [7.24876300e-02, 1.36938194e+00, 3.19502503e-03],
        [1.87243566e-01, 1.41426762e+00, -1.26703978e-02],
        [4.28342104e-01, 1.39951754e+00, -3.28501277e-02],
        [6.77591443e-01, 1.41056996e+00, -4.11638841e-02],
        [7.82230735e-01, 1.40361353e+00, -4.22633551e-02],
        [-7.44956434e-02, 1.36787126e+00, -2.20314227e-03],
        [-1.81450114e-01, 1.41413637e+00, -8.90111178e-03],
        [-4.27909374e-01, 1.39844642e+00, -3.70209515e-02],
        [-6.83784366e-01, 1.40501566e+00, -4.30157445e-02],
        [-7.88080752e-01, 1.40036152e+00, -4.39063571e-02]]
    )

        self.Kinetix_without_hands_names = [
            #  'Kinetix_Reference',
            "m_avg_Pelvis",          # joint nb 0
            "m_avg_L_Hip",           # joint nb 1
            "m_avg_L_Knee",          # joint nb 2
            "m_avg_L_Ankle",         # joint nb 3
            "m_avg_L_Foot",          # joint nb 4
            "m_avg_R_Hip",           # joint nb 5
            "m_avg_R_Knee",          # joint nb 6
            "m_avg_R_Ankle",         # joint nb 7
            "m_avg_R_Foot",          # joint nb 8
            "m_avg_Spine1",          # joint nb 9
            "m_avg_Spine2",          # joint nb 10
            "m_avg_Spine3",          # joint nb 11
            "m_avg_Neck",            # joint nb 12
            "m_avg_Head",            # joint nb 13
            "m_avg_L_Collar",        # joint nb 14
            "m_avg_L_Shoulder",      # joint nb 15
            "m_avg_L_Elbow",         # joint nb 16
            "m_avg_L_Wrist",         # joint nb 17
            "m_avg_L_Hand",          # joint nb 18
            "m_avg_R_Collar",        # joint nb 19
            "m_avg_R_Shoulder",      # joint nb 20
            "m_avg_R_Elbow",         # joint nb 21
            "m_avg_R_Wrist",         # joint nb 22
            "m_avg_R_Hand",          # joint nb 23p.connect(p.GUI, options="--background_color_red=0.08 --background_color_blue=0.08 --background_color_green=0.08 --width=500 --height=350")

        ]

        self.Kinetix_without_hands_hierarchy = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [0, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [0, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [12, 13],
            [11, 14],
            [14, 15],
            [15, 16],
            [16, 17],
            [17, 18],
            [11, 19],
            [19, 20],
            [20, 21],
            [21, 22],
            [22, 23],
        ]

        ### 1. Create the skeleton
        skeleton = Skeleton("Skeleton", None, [])
        skeleton.construct_from_zero_pose(self.Kinetix_without_hands_names, self.Kinetix_without_hands_positions, self.Kinetix_without_hands_hierarchy)
        skeleton.batch_size = self.axisa.shape[0]
        skeleton.set_poses_axis_angle(self.axisa.reshape(-1, 24, 3))
        self.initial_rotation_matrices = skeleton.bone_list[0].global_matrices[:, :3, :3] # Save the initial global matrices, used for reconstructing
        skeleton.bone_list[0].set_local_positions(self.trans) # Lafan translations are given for the pelvis, so we initialize in the following way


        ### 2. Twist and swing decomposition
        pelvis_quat = skeleton.bone_list[0].get_local_rotations_quaternion() # Original quaternion

        # Extract the swing of the pelvis quaternion (without world y-axis)
        y_axis = torch.Tensor([[0, 1, 0]]).repeat((skeleton.batch_size, 1))
        quat_y_axis = ks.utils.rotation_conversions.quaternion_apply(pelvis_quat, y_axis)
        quat_swing = ks.utils.geometry_utils.quaternion_from_2_vectors(y_axis, quat_y_axis)
        quat_swing /= quat_swing.norm(dim=-1, keepdim=True)
        # Compute the twist using the original quaternion and the swing
        quat_twist = kin.quaternion_multiply(kin.quaternion_invert(quat_swing), pelvis_quat)


        ### 3. Use the twist to compute the angular velocity around the y axis
        mat = kin.quaternion_to_matrix(quat_twist)
        euler = kin.matrix_to_euler_angles(mat, "YZX")
        euler[:, 0] = unroll_euler_1d(euler[:, 0])

        dt = 1/self.fps
        self.angular_velocity = (euler[:, 0].unsqueeze(1) - euler[:, 0].unsqueeze(1).roll(1, 0))/dt # The up/y coordinate of cross product of two consecutive projected orientations is used to compute angular velocity
        self.angular_velocity[self.wrong_velocity_idx] = self.angular_velocity[self.wrong_velocity_idx + np.ones_like(self.wrong_velocity_idx)] # The wrong velocity is replaced by the previous one


        ### 4. Projecting the skeleton in character space
        # Only keep the swing of the pelvis 
        skeleton.bone_list[0].set_local_rotations_quaternion(quat_swing)
        #Center the pelvis trans on 0 in the x and z axis
        pelvis_pos = skeleton.bone_list[0].get_global_positions()
        pelvis_pos[:, [0, 2]] *= 0
        skeleton.bone_list[0].set_global_positions(pelvis_pos)

        bone_root_centered_kp = [skeleton.bone_list[i].get_global_positions().unsqueeze(dim=0) for i in range(24)] # KP3D from the translation vector of the affine matrix
        bone_root_centered_rot6d = [kin.matrix_to_rotation_6d(kin.quaternion_to_matrix(skeleton.bone_list[i].get_global_rotations_quaternion())).unsqueeze(dim=0) for i in range(24)] # Rot6d from right, up vectors of the matrix concatenated one after the other
        

        ### 5. Remove Rot6d of the feet and hands (constant in our dataset)    
        self.indices_to_remove_rot6d = [4, 8, 18, 23]  # Indices of lfoot, rfoot, lhand, rhand.
        mask = torch.ones(len(bone_root_centered_rot6d), dtype=bool)
        mask[self.indices_to_remove_rot6d] = False
        bone_root_centered_rot6d = torch.cat(bone_root_centered_rot6d)[mask] # Collapse bones into tensors

        indices_to_remove_kp3d = [0]  # Index of the pelvis
        mask = torch.ones(len(bone_root_centered_kp), dtype=bool)
        mask[indices_to_remove_kp3d] = False
        self.height = bone_root_centered_kp[0][0, :, 1].unsqueeze(1) # Save the height of the pelvis
        bone_root_centered_kp = torch.cat(bone_root_centered_kp)[mask] # Collapse bones into tensors


        ### 6. Concatenate the joint features [n_joints, n_frames, n_features] to [n_frames, n_features*joints]
        self.keypoints = torch.flatten(torch.transpose(bone_root_centered_kp, 0, 1), start_dim=1)
        self.keypoints = torch.cat([self.height , self.keypoints], dim=1) # Add back the height of the pelvis
        self.rot6d = torch.flatten(torch.transpose(bone_root_centered_rot6d, 0, 1), start_dim=1)
        
        ### 7. Compute the velocity of the root and joints
        self.keypoint_velocities = (self.keypoints - self.keypoints.roll(1, 0))/dt
        self.keypoint_velocities[self.wrong_velocity_idx] = self.keypoint_velocities[self.wrong_velocity_idx + np.ones_like(self.wrong_velocity_idx)]
    
    # Returns data as a pose dictionnary (first pose of the sequence fixed at pelvis position 0)
    def model_output_to_pose(self, y, initial_matrix):
        '''
        converts :
          - list of feature pose transitions [n_frames, n_features]
          - an initial transform of the pelvis [4, 4]
        into an animation dictionnary :
        {
        trans_pelvis : tensor[n_frames, 3], 
        poses : tensor[n_frames, 24, 3]
        }
        '''

        #y = self.denormalize(y)

        ### 1. Get the available rot6d from the model output and complete it with the removed rotations
        rot6d = y[:, -self.rot6d.shape[1]:].reshape(-1, self.rot6d.shape[1]//6, 6)

        # Complete the rot6d with the removed rotations, For lleg, rleg, larm, rarm, repete the rot6d of the previous joints, this loop only works if end index of array should be repeated
        total_rot6d = []
        indices_to_duplicate = [0] + [index - 1 - i for i, index in enumerate(self.indices_to_remove_rot6d)] # 0 is added at the begining for iteration purposes
        for i, index in enumerate(indices_to_duplicate[:-1]):
            total_rot6d += [rot6d[:, index:indices_to_duplicate[i+1]+1]]
        total_rot6d += [rot6d[:, indices_to_duplicate[-1]].unsqueeze(dim=1)]
        rot6d = torch.cat(total_rot6d, dim=1)


        ### 2. Create a skeleton from the rot6d (will be in character space)
        skeleton = Skeleton("Skeleton", None, [], device="cuda")
        skeleton.construct_from_zero_pose(self.Kinetix_without_hands_names, self.Kinetix_without_hands_positions, self.Kinetix_without_hands_hierarchy)
        skeleton.batch_size = y.shape[0]
        global_quaternions = kin.matrix_to_quaternion(kin.rotation_6d_to_matrix(rot6d))

        # Place the skeleton in character space with the global rot6 given in the input
        for i, bone in enumerate(skeleton.bone_list):
            bone.set_global_rotations_quaternion(global_quaternions[:, i]) # Pelvis gets back it's swing


        ### 3. Add back the twist to the root rotation
        # Integrate the root space 2D rotation cumulated angle differences
        angular_change_per_frame = y[:, 2]/self.fps
        total_orientation_per_frame = torch.cumsum(angular_change_per_frame, dim=0) 
        # Add twist to the pelvis from the integrated angular change
        character_pelvis_rotation = skeleton.bone_list[0].get_global_rotations_quaternion()
        y_pelvis_rotation = torch.zeros(y.shape[0], 3).to("cuda")
        y_pelvis_rotation[:, 1] = total_orientation_per_frame
        twist = kin.axis_angle_to_quaternion(y_pelvis_rotation)
        world_pelvis_rotation = kin.quaternion_multiply(character_pelvis_rotation, twist)
        skeleton.bone_list[0].set_global_rotations_quaternion(world_pelvis_rotation) # Pelvis gets back it's twist


        ### 4. Integrate the translations
        delta_trans_2d = y[:, :2]/self.fps
        total_trans_2d = torch.cumsum(delta_trans_2d, dim=0)
        trans = torch.cat([total_trans_2d[:, 0].unsqueeze(1), y[:, 3].unsqueeze(1),total_trans_2d[:, 1].unsqueeze(1)], dim=1)
        skeleton.bone_list[0].set_local_positions(trans) # Set the pelvis position

        ### 5. Correct the root position with the initial matrix
        skewed_pelvis_pose = kin.quaternion_to_matrix(world_pelvis_rotation)
        initial_skewed_orientation = skewed_pelvis_pose[0]
        correction = torch.linalg.inv(initial_skewed_orientation)@initial_matrix.unsqueeze(0)
        pelvis_pose = skewed_pelvis_pose@correction

        skeleton.bone_list[0].set_global_rotations_quaternion(kin.matrix_to_quaternion(pelvis_pose))
        first_frame_pelvis = skeleton.bone_list[0].get_global_positions()

        ### 6. Extract the rotations and translations from the skeleton
        axisa = torch.cat([bone.get_local_rotations_axisangle().unsqueeze(dim=1) for bone in skeleton.bone_list], dim=1)
        pose = {"trans_pelvis" : trans, "poses" : axisa}
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
        self.seq_len = self.hparams.seq_len

    @property
    def n_features(self):
        return self.hparams.seq_len * 263

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
            self.data_train = PoseSequenceDataset(self.hparams.data_dir + 'LAFAN/train/', seq_len=self.hparams.seq_len, training=True)
            self.data_check = PoseSequenceDataset(self.hparams.data_dir + 'LAFAN/val/', seq_len=self.hparams.seq_len, training=False)

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


    dir_path = os.path.dirname(os.path.realpath(__file__))
    _ = LAFANDataModule(dir_path + "/../../data/", 8)
    _.setup()
    first_idx = 110000
    last_idx = 111000
    data_to_reconstruct = _.data_train[first_idx:last_idx]
    data_to_reconstruct = _.data_train[first_idx:last_idx].to("cuda")[:, :_.n_features//_.seq_len]
    initial_pose = _.data_train.initial_rotation_matrices[_.data_train.selectable_indices[first_idx]].to("cuda")
    #initial_pose = _.data_train.initial_rotation_matrices[first_idx].to("cuda")
    torch.save(_.data_train.model_output_to_pose(data_to_reconstruct, initial_pose), "Misc/check_reconstruction_ground_contacts.pt")