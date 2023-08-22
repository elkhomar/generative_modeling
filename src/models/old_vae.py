import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning import LightningModule


class VAE(LightningModule):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.beta = 0.002
        self.mse = nn.MSELoss(reduction="mean")

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            #nn.Flatten(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
        )

        # Latent vectors
        self.mu = nn.Linear(64, latent_dim)
        self.log_var = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        mu = self.mu(self.encoder(x))
        log_var = self.log_var(self.encoder(x))
        return mu, log_var

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var, x

    def loss_function(self, x_hat, x, mu, log_var):
        # Reconstruction loss
        mse = self.mse(x_hat, x)

        # KL divergence loss
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), axis=1))
        return (mse + self.beta * kld, mse, kld)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat, mu, log_var, x = self.forward(x)
        loss, mse, kld = self.loss_function(x_hat, x, mu, log_var)
        self.log("train/mse", mse)
        self.log("train/kl", kld)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat, mu, log_var, x = self.forward(x)
        loss, mse, kld = self.loss_function(x_hat, x, mu, log_var)
        self.log("val/mse", mse)
        self.log("val/kl", kld)
        self.log("val/loss", loss)

    def test_step(self, batch, batch_idx):
        x = batch
        x_hat, mu, log_var, x = self.forward(x)
        loss, mse, kld = self.loss_function(x_hat, x, mu, log_var)
        self.log("val/mse", mse)
        self.log("val/kl", kld)
        self.log("val/loss", loss)

if __name__ == "__main__":
    _ = VAE(None, None)