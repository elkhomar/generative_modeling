from torch import nn

latent_dim = 8
lin1_dim = 64
output_dim = 256
depth = 3

r = pow(output_dim/lin1_dim, 1/depth)
        
modules = []

# First layer
modules.append(nn.Linear(latent_dim, lin1_dim))
modules.append(nn.LeakyReLU())

# Middle layers
for i in range(depth-1):
    modules.append(nn.Linear(round(lin1_dim*pow(r, i)), round(lin1_dim*pow(r, i+1))))
    modules.append(nn.LeakyReLU())

# Last layer
modules.append(nn.Linear(round(lin1_dim*pow(r, depth-1)), round(lin1_dim*pow(r, depth))))

model = nn.Sequential(*modules)