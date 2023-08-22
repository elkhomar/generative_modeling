import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
import kinetix_scenegraph.utils.rotation_conversions as kx

 
class beta_VAEModule(LightningModule):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 input_dim,
                 latent_dim, 
                 beta=1.0
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.beta = self.hparams.beta
        self.mse = nn.MSELoss(reduction="mean")

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
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

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.train_mse = MeanMetric()
        self.train_kl = MeanMetric()

        self.val_loss = MeanMetric()
        self.val_mse = MeanMetric()
        self.val_kl = MeanMetric()

        self.test_loss = MeanMetric()
        self.test_mse = MeanMetric()
        self.test_kl = MeanMetric()

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
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_mse.reset()
        self.val_kl.reset()

    def loss_function(self, x_hat, x, mu, log_var):
        # Reconstruction loss
        # mse = self.mse(x_hat, x)
        # Rot 6D mse is computed using the matrix conversion
        x_mat = kx.rotation_6d_to_matrix(x.reshape(-1,24,6)).reshape(-1,24*9)
        x_hat_mat = kx.rotation_6d_to_matrix(x_hat.reshape(-1,24,6)).reshape(-1,24*9)
        mse = self.mse(x_mat, x_hat_mat)

        # KL divergence loss
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), axis=1))
        return (mse + self.beta * kld, mse, kld)

    
    def training_step(self, batch, batch_idx):
        x = batch
        x_hat, mu, log_var, x = self.forward(x)
        loss, mse, kld = self.loss_function(x_hat, x, mu, log_var)

        # update and log metrics
        self.train_mse(mse)
        self.log("train/mse", self.train_mse, on_step=True, on_epoch=True, prog_bar=True)

        self.train_kl(kld)
        self.log("train/kl", self.train_kl, on_step=True, on_epoch=True, prog_bar=True)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
            pass
    
    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat, mu, log_var, x = self.forward(x)
        loss, mse, kld = self.loss_function(x_hat, x, mu, log_var)

        # update and log metrics
        self.val_mse(mse)
        self.log("val/mse", self.val_mse, on_step=True, on_epoch=True, prog_bar=True)

        self.val_kl(kld)
        self.log("val/kl", self.val_kl, on_step=True, on_epoch=True, prog_bar=True)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x = batch
        x_hat, mu, log_var, x = self.forward(x)
        loss, mse, kld = self.loss_function(x_hat, x, mu, log_var)

        self.test_mse(mse)
        self.log("test/mse", self.test_mse, on_step=True, on_epoch=True, prog_bar=True)

        self.test_kl(kld)
        self.log("test/kl", self.test_kl, on_step=True, on_epoch=True, prog_bar=True)

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
            """Choose what optimizers and learning-rate schedulers to use in your optimization.
            Normally you'd need one. But in the case of GANs or similar you might have multiple.

            Examples:
                https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
            """
            optimizer = self.hparams.optimizer(params=self.parameters())
            if self.hparams.scheduler is not None:
                scheduler = self.hparams.scheduler(optimizer=optimizer)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val/loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            return {"optimizer": optimizer}

if __name__ == "__main__":
    _ = beta_VAEModule(None, None, None, None, None)