from torch import nn
import torch.nn.functional as F
import torch


class Discriminator(nn.Module):
    def __init__(self, d_input_dim=4, d_hidden_dim=16):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, d_hidden_dim)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.sigmoid(self.fc4(x))
