import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric


 
class MVAEModule(LightningModule):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 input_dim,
                 latent_dim, 
                 beta=1.0,
                 n_experts=6,
                 hidden_size=256,
                 prediction_length=8
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_features = input_dim//2

        self.n_experts = self.hparams.n_experts
        self.hidden_size = self.hparams.hidden_size
        self.prediction_length = self.hparams.prediction_length

        self.p = 1.0


        self.beta = self.hparams.beta
        self.mse = nn.MSELoss(reduction="mean")

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )

        # Latent vectors
        self.mu = nn.Linear(256, latent_dim)
        self.log_var = nn.Linear(256, latent_dim)

        self.gate = nn.Sequential(
            nn.Linear(self.n_features + self.latent_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, n_experts),
            nn.Softmax(dim=1))

        self.decoder_layers = [
            (
                nn.Parameter(torch.empty(self.n_experts, self.n_features + self.latent_dim, self.hidden_size, device="cuda")),
                nn.Parameter(torch.empty(self.n_experts, self.hidden_size, device="cuda")),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(n_experts, self.hidden_size + latent_dim, self.hidden_size, device="cuda")),
                nn.Parameter(torch.empty(n_experts, self.hidden_size, device="cuda")),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(n_experts, self.hidden_size + latent_dim, self.n_features, device="cuda")),
                nn.Parameter(torch.empty(n_experts, self.n_features,device="cuda")),
                None,
            ),
        ]

        self.initialize_decoders()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.train_mse = MeanMetric()
        self.train_kl = MeanMetric()

        self.train_loss_resampled = MeanMetric()
        self.train_mse_resampled = MeanMetric()
        self.train_kl_resampled = MeanMetric()

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


    def initialize_decoders(self):
        for index, (weight, bias, _) in enumerate(self.decoder_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w_expert" + index, weight)
            self.register_parameter("b_expert" + index, bias)

    def decode(self, z, c):

        coefficients = self.gate(torch.cat((c, z), dim=1))
        z = z.unsqueeze(1)
        layer_out = c.unsqueeze(1)

        for (weight, bias, activation) in self.decoder_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
            coefficients.shape[0], *weight.shape[1:3]
            )
            input_ = torch.cat((z, layer_out), dim=2)
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            out = torch.baddbmm(input = mixed_bias, batch1 = input_, batch2 = mixed_weight)
            layer_out = activation(out) if activation is not None else out

        predpose = layer_out.reshape(layer_out.shape[0], layer_out.shape[-1])

        return predpose

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):

        # Get the encoded latent vector
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)

        # Get the previous pose
        prevpose = x[:, -self.n_features:]

        # Use the latent vector and previous pose to predict the next pose with the mixed decoder
        predpose = self.decode(z, prevpose)  

        return predpose, mu, log_var, x
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_mse.reset()
        self.val_kl.reset()

    def loss_function(self, ppred, x, mu, log_var):
        # Reconstruction loss
        mse = self.mse(ppred, x[:, -ppred.shape[1]:])

        # KL divergence loss
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), axis=1))
        return (mse + self.beta * kld, mse, kld)

    
    def training_step(self, batch, batch_idx):

        # Resample with fixed probability 0.2, resampled loss is averaged accross rollout before passing through the gradient
        if torch.bernoulli(torch.tensor(self.p)):
            loss = self.training_step_regular(batch, batch_idx)
        else:
            loss = self.training_step_resample(batch, batch_idx)

        return loss

    def training_step_regular(self, batch, batch_idx):
        x = batch
        predpose, mu, log_var, x = self.forward(x)
        loss, mse, kld = self.loss_function(predpose, x, mu, log_var)

        # update and log metrics
        self.train_mse(mse)
        self.log("train/mse", self.train_mse, on_step=True, on_epoch=True, prog_bar=True)

        self.train_kl(kld)
        self.log("train/kl", self.train_kl, on_step=True, on_epoch=True, prog_bar=True)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def training_step_resample(self, batch, batch_idx):
        x = batch
        losses, mses, klds = [], [], []

        for i in range(self.prediction_length):
            predpose, mu, log_var, x = self.forward(x)
            loss, mse, kld = self.loss_function(predpose, x, mu, log_var)
            losses.append(loss)
            mses.append(mse)
            klds.append(kld)
            x = torch.cat((x[:, -self.n_features:], predpose), dim=1)

        loss, mse, kld  = torch.stack(losses).mean(), torch.stack(mses).mean(), torch.stack(klds).mean()

        # update and log metrics
        self.train_mse_resampled(mse)
        self.log("train/resampled/mse", self.train_mse_resampled, on_step=True, on_epoch=True, prog_bar=True)

        self.train_kl_resampled(kld)
        self.log("train/resampled/kl", self.train_kl_resampled, on_step=True, on_epoch=True, prog_bar=True)

        self.train_loss_resampled(loss)
        self.log("train/resampled/loss", self.train_loss_resampled, on_step=True, on_epoch=True, prog_bar=True)
        return loss



    def on_train_epoch_end(self):
        self.current_epoch
        schedule = 20*[1.0] + [1-i/20 for i in range(20)] + 140*[0.0]
        self.p = schedule[self.current_epoch]
    
    def validation_step(self, batch, batch_idx):
        x = batch
        predpose, mu, log_var, x = self.forward(x)
        loss, mse, kld = self.loss_function(predpose, x, mu, log_var)

        # update and log metrics
        self.val_mse(mse)
        self.log("val/mse", self.val_mse, on_step=True, on_epoch=True, prog_bar=True)

        self.val_kl(kld)
        self.log("val/kl", self.val_kl, on_step=True, on_epoch=True, prog_bar=True)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x = batch
        predpose, mu, log_var, x = self.forward(x)
        loss, mse, kld = self.loss_function(predpose, x, mu, log_var)

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
    _ = MVAEModule(None, None, None, None, None, None, None)