import torch
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader

import pybullet as p
import time
import pybullet_data
from kinetix_scenegraph.core.scene import Scene
from kinetix_scenegraph.primitives import Axes, Cube, Sphere
from kinetix_scenegraph.skeletons.skeleton import Skeleton
import kinetix_scenegraph.utils.rotation_conversions as kin
import pybullet as p



fps = 60

# Load path from current dir
"""with open("configs/paths/default.yaml") as f:
    cfg = yaml.load(f, Loader=SafeLoader)
    path = cfg["current_dir"]
    path = path.replace("${paths.root_dir}/", "")
    path += "/"
"""

path = "Misc/"
anim = torch.load(path + 'check_reconstruction.pt')
anim_ori = torch.load(path + 'locomotion_1.pt')
#anim = torch.load('Saved/Results/No_translation/Vae_beta_0.002/interpolation_anim/interpolate_frame_0_to_4000')
pose = anim["poses"][0:].to("cpu")
trans_pelvis = anim["trans_pelvis"][0:].to("cpu")

pose_ori = anim_ori["poses"][0:]
trans_pelvis_ori = anim_ori["trans_pelvis"][0:]

NBR_FRAMES = pose.shape[0]

Kinetix_without_hands_positions = torch.Tensor(
    [[-1.43760722e-03, 9.54190561e-01, 2.13284530e-02],
     [5.45075648e-02, 8.74072471e-01, -5.23403287e-06],
     [1.00507133e-01, 4.85931197e-01, 8.22260603e-03],
     [8.78663585e-02, 7.30262800e-02, -2.69002765e-02],
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

Kinetix_without_hands_names = [
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

Kinetix_without_hands_hierarchy = [
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

scene = Scene()
n = 0

# Reconst anim
skeleton = Skeleton("Skeleton", scene, [])
skeleton.construct_from_zero_pose(Kinetix_without_hands_names, Kinetix_without_hands_positions, Kinetix_without_hands_hierarchy)
skeleton.batch_size = pose.shape[0]

"""skewed_pelvis_pose = kin.axis_angle_to_matrix(pose)[:, 0]
initial_orientation = kin.axis_angle_to_matrix(pose)[0, 0]
initial_original_orientation = kin.axis_angle_to_matrix(pose_ori)[0, 0]
correction = torch.linalg.inv(initial_orientation)@initial_original_orientation.unsqueeze(0)
pelvis_pose = skewed_pelvis_pose@correction
pose[:, 0] = kin.matrix_to_axis_angle(pelvis_pose)"""

skeleton.set_poses_axis_angle(pose)
skeleton.bone_list[0].set_local_positions(trans_pelvis)

# Original anim
skeleton_ori = Skeleton("Skeleton", scene, [])
skeleton_ori.construct_from_zero_pose(Kinetix_without_hands_names, Kinetix_without_hands_positions, Kinetix_without_hands_hierarchy)
skeleton_ori.batch_size = pose.shape[0]

skeleton_ori.set_poses_axis_angle(pose_ori)
trans_pelvis_ori[:, [0, 2]] -= trans_pelvis_ori[0,[0, 2]]
skeleton_ori.bone_list[0].set_local_positions(trans_pelvis_ori + torch.tensor([0, 0, 0]))

Axes("Pelvis", skeleton.bone_list[0])
Axes("Pelvis", skeleton_ori.bone_list[0])
axes = Axes("Axes", scene)
#axe_reconst_pelvis = Axes("Axes", skeleton.bone_list[0])
#axe_reconst_root = Axes("Axes", skeleton)
scene.render_batch_animation(animation_fps=fps)
