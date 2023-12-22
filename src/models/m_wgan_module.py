import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from src.custom_metrics import absolute_kendall_error_torch, anderson_darling_distance
from src.visualisations import log_pairplots
import scipy.stats as stats
import numpy as np


class WGAN(LightningModule):
    def __init__(self,
                 optimizer1: torch.optim.Optimizer,
                 optimizer2: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 generator: torch.nn.Module,
                 discriminator: torch.nn.Module,
                 input_dim,
                 latent_dim: int = 100,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 batch_size: int = 64, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size

        self.automatic_optimization = False

        self.G = generator
        self.G.training = True
        self.D = discriminator
        self.D.training = True
        self.criterion = nn.BCELoss()

        # Non standard passes (should be changed)
        self.val_data = None
        self.log_dir = None
        self.input_dim = input_dim

        # Metrics
        self.train_g_loss = MeanMetric()
        self.train_d_loss = MeanMetric()

        self.val_g_loss = MeanMetric()
        self.val_d_loss = MeanMetric()

        self.val_AD = MeanMetric()
        self.val_AK = MeanMetric()
        self.val_AD_min = MinMetric()
        self.val_AK_min = MinMetric()

        self.test_g_loss = MeanMetric()
        self.test_d_loss = MeanMetric()

    def sample_z(self, n):
#       z = torch.randn(n, self.latent_dim, device=self.device)
        z= torch.tensor(np.random.weibull(1, size=(n, self.latent_dim))).to(self.device)

        return z

    def sample_G(self, n):
        z = self.sample_z(n)
        return self.G(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var, x

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_d_loss.reset()
        self.val_g_loss.reset()
        self.val_AD.reset()
        self.val_AK.reset()
        self.val_AD_min.reset()
        self.val_AK_min.reset()

    def training_step(self, batch, batch_idx):
        # Implementation follows the PyTorch tutorial:
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        g_opt, d_opt = self.optimizers()

        X = batch
        batch_size = X.shape[0]

        real_label = torch.ones((batch_size, 1), device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)

        g_X = self.sample_G(batch_size)

        ##########################
        # Optimize Discriminator #
        ##########################
        d_x = self.D(X)
        errD_real = self.criterion(d_x, real_label)

        d_z = self.D(g_X.detach())
        errD_fake = self.criterion(d_z, fake_label)

        errD = errD_real + errD_fake

        d_opt.zero_grad()
        self.manual_backward(errD)
        d_opt.step()

        ######################
        # Optimize Generator #
        ######################
        d_z = self.D(g_X)
        errG = self.criterion(d_z, real_label)

        g_opt.zero_grad()
        self.manual_backward(errG)
        g_opt.step()

        # update and log metrics
        self.train_g_loss(errG)
        self.log("train/g_loss", self.train_g_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.train_d_loss(errD)
        self.log("train/d_loss", self.train_d_loss, on_step=True, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        pass

    # Validation section
    def validation_step(self, batch, batch_idx):
        # Implementation follows the PyTorch tutorial:
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        g_opt, d_opt = self.optimizers()

        X = batch
        batch_size = X.shape[0]

        real_label = torch.ones((batch_size, 1), device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)

        g_X = self.sample_G(batch_size)

        ##########################
        # Optimize Discriminator #
        ##########################
        d_x = self.D(X)
        errD_real = self.criterion(d_x, real_label)

        d_z = self.D(g_X.detach())
        errD_fake = self.criterion(d_z, fake_label)

        errD = errD_real + errD_fake

        ######################
        # Optimize Generator #
        ######################
        d_z = self.D(g_X)
        errG = self.criterion(d_z, real_label)

        # update and log metrics
        self.val_g_loss(errG)
        self.log("val/g_loss", self.val_g_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.val_d_loss(errD)
        self.log("val/d_loss", self.val_d_loss, on_step=True, on_epoch=True, prog_bar=True)

        # should be changed
        self.log("val/loss", self.val_g_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # Generate samples and compute Anderson-Darling distance and Absolute Kendall error
        x_hat = self.sample_G(len(self.val_data))
        x = self.val_data
        if (self.current_epoch % 10 == 0):
            
            self.val_AD_min(anderson_darling_distance(x, x_hat))
            self.log("val/AD_min", self.val_AD_min.compute(), on_step=False, on_epoch=True, prog_bar=True)
            self.val_AD(anderson_darling_distance(x, x_hat))
            self.log("val/AD", self.val_AD, on_step=False, on_epoch=True, prog_bar=True)        
            
            self.val_AK(absolute_kendall_error_torch(x, x_hat))
            self.log("val/AK", self.val_AK, on_step=False, on_epoch=True, prog_bar=True)
            self.val_AK_min(absolute_kendall_error_torch(x, x_hat))
            self.log("val/AK_min", self.val_AK_min.compute(), on_step=False, on_epoch=True, prog_bar=True)
            # Create histograms of generated samples with seaborn
            log_pairplots(x_hat, x, self.current_epoch, self.log_dir + "/visualisations/")

    def test_step(self, batch, batch_idx):

        # Implementation follows the PyTorch tutorial:
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        g_opt, d_opt = self.optimizers()

        X = batch
        batch_size = X.shape[0]

        real_label = torch.ones((batch_size, 1), device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)

        g_X = self.sample_G(batch_size)

        ##########################
        # Optimize Discriminator #
        ##########################
        d_x = self.D(X)
        errD_real = self.criterion(d_x, real_label)

        d_z = self.D(g_X.detach())
        errD_fake = self.criterion(d_z, fake_label)

        errD = errD_real + errD_fake

        ######################
        # Optimize Generator #
        ######################
        d_z = self.D(g_X)
        errG = self.criterion(d_z, real_label)

        self.test_g_loss(errG)
        self.log("test/g_loss", self.test_g_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.test_d_loss(errD)
        self.log("test/d_loss", self.test_d_loss, on_step=True, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.G.parameters(), lr=self.lr)
        d_opt = torch.optim.Adam(self.D.parameters(), lr=self.lr)
        return g_opt, d_opt

    def generate_final_samples(self):
        z = self.sample_z(410)
        x_hat = self.decode(z)
        torch.save(x_hat, self.log_dir + "/final_samples.pt")
        torch.save(z, self.log_dir + "/final_latent.pt")
        return x_hat, z


if __name__ == "__main__":
    _ = WGAN(None, None, None, None, None)
