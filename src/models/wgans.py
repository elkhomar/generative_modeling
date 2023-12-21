import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import metrics

dataset = pd.read_csv('/home/sebastien/projets/WGAN/PyTorch-GAN/implementations/wgan/data_train_log_return.csv',index_col=0)

log_data = np.log(dataset)
train_data = log_data.iloc[:336]
test_data = log_data.iloc[336:746]
train_tensor = torch.tensor(train_data.values.astype(np.float32))
test_tensor = torch.tensor(test_data.values.astype(np.float32))

# Create data loaders for the training set
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=16, shuffle=True)


class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Assuming 4 columns in dataset
        )

    def forward(self, z):
        return self.model(z)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
noise_dim = 100
lr = 0.0002
epochs = 100
# Model initialization
generator = Generator(noise_dim)
critic = Critic()
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_c = optim.Adam(critic.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    for real_data, in train_loader:
        # Update critic
        optimizer_c.zero_grad()
        z = torch.randn(real_data.size(0), noise_dim)
        fake_data = generator(z).detach()
        critic_loss = -torch.mean(critic(real_data)) + torch.mean(critic(fake_data))
        critic_loss.backward()
        optimizer_c.step()

        # Update generator
        optimizer_g.zero_grad()
        z = torch.randn(real_data.size(0), noise_dim)
        fake_data = generator(z)
        generator_loss = -torch.mean(critic(fake_data))
        generator_loss.backward()
        optimizer_g.step()
    print(f"Epoch {epoch+1}/{epochs} completed.")   
    
def generate_data(generator, noise_dim, num_examples_to_generate):
    with torch.no_grad():
        z = torch.randn(num_examples_to_generate, noise_dim)
        generated_data = generator(z)
    return generated_data.numpy()

new_data = generate_data(generator, noise_dim, 410)