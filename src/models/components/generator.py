from torch import nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, latent_dim=4, g_hidden_dim=16, g_output_dim=4):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, g_hidden_dim)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

        self.fc5 = nn.Linear(2*g_output_dim, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return self.fc4(x)
