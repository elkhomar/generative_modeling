# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
import os
from yaml.loader import SafeLoader

path = os.getcwd()
path += "/notebooks/Data_exploration/"

data = torch.load(path + 'feature_space_ground_contacts.pt')
df = pd.DataFrame(data.numpy())

# Rename columns
features_list = ['deltax', 'deltaz', 'delta_orientation', 'pelvis height']

for i in range(0, 4):
    features_list.append(f'ground_contact_{i}')

for i in range(1, 24):  # 19 joints including the pelvis, excluding the pelvis
    for ax in 'xyz':
        features_list.append(f'3d_keypoints_{i}_{ax}')

features_list.append(f'pelvis_height_velocity')
for i in range(1, 24):  # 19 joints velocity including the pelvis, excluding the pelvis
    for ax in 'xyz':
        features_list.append(f'3d_velocities_{i}_{ax}')

for i in range(0, 20):  # orientation vectors for 18 joints
    for ax in 'xyz':
        features_list.append(f'orientation_1_{i}_{ax}')
    for ax in 'xyz':
        features_list.append(f'orientation_2_{i}_{ax}')

df.columns = features_list
df['frame'] = df.index
df['any_gc'] = df['ground_contact_0'] + df['ground_contact_1'] + df['ground_contact_2'] + df['ground_contact_3']


import plotly.express as px
joint_idx = 1
fig = px.scatter_3d(df, x='3d_keypoints_4_x', y='3d_keypoints_4_z', z='3d_keypoints_4_y', color='any_gc', range_x=[-2, 2], range_y=[-2, 2], range_z=[-2, 2], hover_data=['frame'])
#fig = px.scatter_3d(df, x=f'orientation_1_{joint_idx}_x', y=f'orientation_1_{joint_idx}_z', z=f'orientation_1_{joint_idx}_y', color='frame', range_x=[-2, 2], range_y=[-2, 2], range_z=[-2, 2])
#fig = px.scatter_3d(df, x=f'deltax', y=f'deltaz', z=f'pelvis height', color='frame', range_x=[-2, 2], range_y=[-2, 2], range_z=[-2, 2])
fig.show()