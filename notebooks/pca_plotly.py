# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
import os
from yaml.loader import SafeLoader
from sklearn.decomposition import PCA


path = os.getcwd()
path += "/notebooks/Data_exploration/"

data = torch.load(path + 'training_features_all_lafan_root_space')

# Compute PCA space
pca = PCA(n_components=3).fit(data)
projected_dataset = pca.transform(data)

### Dataframe formating with pandas

df = pd.DataFrame(data.numpy())

# Rename columns
features_list = ['deltax', 'deltaz', 'delta_orientation', 'pelvis height']

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

### Adding the PCA components to the dataframe
df['PCA_1'] = projected_dataset[:, 0]
df['PCA_2'] = projected_dataset[:, 1]
df['PCA_3'] = projected_dataset[:, 2]


### Plotting with plotly

import plotly.express as px
joint_idx = 1
fig = px.scatter_3d(df[:1000000], x='PCA_1', y='PCA_2', z='PCA_3', color='frame')
#fig = px.scatter_3d(df, x=f'orientation_1_{joint_idx}_x', y=f'orientation_1_{joint_idx}_z', z=f'orientation_1_{joint_idx}_y', color='frame', range_x=[-2, 2], range_y=[-2, 2], range_z=[-2, 2])
#fig = px.scatter_3d(df, x=f'deltax', y=f'deltaz', z=f'pelvis height', color='frame', range_x=[-2, 2], range_y=[-2, 2], range_z=[-2, 2])
fig.show()