import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric
from src.custom_metrics import absolute_kendall_error_torch, anderson_darling_distance
from src.visualisations import log_pairplots


class beta_VAEModule(LightningModule):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 input_dim,
                 latent_dim,
                 encoder_dims,
                 decoder_dims,
                 beta=1.0,
                 activation="LeakyReLU",
                 predict_log=True,
                 ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.beta = self.hparams.beta
        self.activation = eval("nn." + activation)

        self.predict_log = predict_log

        self.mse = nn.MSELoss(reduction="mean")

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_dims[0]),
            self.activation(),
            nn.Linear(encoder_dims[0], encoder_dims[1]),
            self.activation(),
            nn.Linear(encoder_dims[1], encoder_dims[2]),
            self.activation(),
        )

        # Latent vectors
        self.mu = nn.Linear(encoder_dims[2], latent_dim)
        self.log_var = nn.Linear(encoder_dims[2], latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_dims[0]),
            self.activation(),
            nn.Linear(decoder_dims[0], decoder_dims[1]),
            self.activation(),
            nn.Linear(decoder_dims[1], decoder_dims[2]),
            self.activation(),
            nn.Linear(decoder_dims[2], input_dim)
        )

        # Non standard passes (should be changed)
        self.val_data = None
        self.log_dir = None

        # Metrics
        self.train_loss = MeanMetric()
        self.train_mse = MeanMetric()
        self.train_kl = MeanMetric()

        self.val_loss = MeanMetric()
        self.val_mse = MeanMetric()
        self.val_kl = MeanMetric()

        self.val_data = None
        self.val_AD = MeanMetric()
        self.val_AK = MeanMetric()

        self.test_loss = MeanMetric()
        self.test_mse = MeanMetric()
        self.test_kl = MeanMetric()

    # Training section
    def encode(self, x):
        mu = self.mu(self.encoder(x))
        log_var = self.log_var(self.encoder(x))
        return mu, log_var

    def decode(self, z):
        return self.decoder(z)  # use .exp() to make a lognormal prior and F.normalize(z) for spherical latent space

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
        mse = self.mse(x_hat, x) if self.predict_log else self.mse(x_hat, x.log())

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

    # Validation section
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

    # Compute Anderson-Darling distance and Absolute Kendall error

    def on_validation_epoch_end(self):
        # Generate samples and compute Anderson-Darling distance and Absolute Kendall error
        z = torch.randn(len(self.val_data), self.latent_dim).to("cuda")  # Generate latents N(0, I)
        x_hat = self.decode(z)

        x = self.val_data if self.predict_log else self.val_data.log()

        self.val_AD(anderson_darling_distance(x, x_hat))
        self.log("val/AD", self.val_AD, on_step=False, on_epoch=True, prog_bar=True)
        
        self.val_AK(absolute_kendall_error_torch(x, x_hat))
        self.log("val/AK", self.val_AK, on_step=False, on_epoch=True, prog_bar=True)

        if (self.current_epoch % 10 == 0):
            # Create histograms of generated samples with seaborn
            log_pairplots(x_hat, x, self.current_epoch, self.log_dir + "/visualisations/")

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
