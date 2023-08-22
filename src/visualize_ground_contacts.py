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
import pandas as pd



fps = 60

# Load path from current dir
"""with open("configs/paths/default.yaml") as f:
    cfg = yaml.load(f, Loader=SafeLoader)
    path = cfg["current_dir"]
    path = path.replace("${paths.root_dir}/", "")
    path += "/"
"""

path = "data/LAFAN/train/"
n_anim = 0

pose_dir = path + 'poses/'
trans_pelvis_dir = path + 'trans_pelvis/'
ground_contacts_dir = path + 'ground_contacts/'

# Extract anim from dir
pose = torch.from_numpy(np.load(pose_dir + os.listdir(pose_dir)[n_anim])["poses"]).reshape(-1, 24, 3)
trans_pelvis = torch.from_numpy(np.load(trans_pelvis_dir + os.listdir(trans_pelvis_dir)[n_anim])["trans_pelvis"])
ground_contacts = torch.from_numpy(np.load(ground_contacts_dir + os.listdir(ground_contacts_dir)[n_anim])["ground_contacts"])



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

skeleton = Skeleton("Skeleton", scene, [])
skeleton.construct_from_zero_pose(Kinetix_without_hands_names, Kinetix_without_hands_positions, Kinetix_without_hands_hierarchy)
skeleton.batch_size = pose.shape[0]

skeleton.set_poses_axis_angle(pose)

trans_pelvis[:, [0, 2]] -= trans_pelvis[0, [0, 2]]
skeleton.bone_list[0].set_local_positions(trans_pelvis)

indices = [3, 4, 7, 8]
for i, bone_index in enumerate(indices):
    contact = Sphere("contact", skeleton.bone_list[bone_index], color=(1, 0, 0))
    scale = torch.ones((pose.shape[0], 3)) * 0.025 * ground_contacts[:, i].unsqueeze(1)
    contact.set_local_scales(scale)

Axes("Pelvis", skeleton.bone_list[0])
axes = Axes("Axes", scene)
#axe_reconst_pelvis = Axes("Axes", skeleton.bone_list[0])
#axe_reconst_root = Axes("Axes", skeleton)
scene.render_batch_animation(animation_fps=fps)
