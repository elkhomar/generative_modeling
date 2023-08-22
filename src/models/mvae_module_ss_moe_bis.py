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
        self.prediction_length = self.hparams.prediction_length
        self.n_features = input_dim//self.prediction_length

        self.n_experts = self.hparams.n_experts
        self.coefficients = None
        self.hidden_size = self.hparams.hidden_size

        self.p = 1.0

        self.initial_lr = 1e-4
        self.final_lr = 1e-7


        self.beta = self.hparams.beta
        self.mse = nn.MSELoss(reduction="mean")

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(2*self.n_features, 256),
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

        self.experts = nn.ModuleList([

            # For each expert, we must define the following layersin a non sequential way to feed back z in each layer

            nn.ModuleList([
            nn.Sequential(
            nn.Linear(self.n_features + self.latent_dim, 256),
            nn.LeakyReLU()),

            nn.Sequential(
            nn.Linear(256 + self.latent_dim, 256),
            nn.LeakyReLU()),

            nn.Sequential(
            nn.Linear(256 + self.latent_dim, 256),
            nn.LeakyReLU()),

            nn.Linear(256 + self.latent_dim, self.n_features)])

            for i in range(self.n_experts)])
        
        """self.experts = [
            nn.Sequential(nn.Linear(self.n_features + self.latent_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.n_features), 
            nn.LeakyReLU()).to("cuda") for i in range(self.n_experts)]"""

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

        self.mse_wrt_prev = MeanMetric()

        self.automatic_optimization = False


    def encode(self, x):
        mu = self.mu(self.encoder(x))
        log_var = self.log_var(self.encoder(x))
        return mu, log_var

    def decode(self, z, c):
        # Predicts a pose from the latent vector z and the previous pose c, give in addition the prediction of each expert

        # Gating network
        self.coefficients = self.gate(torch.cat((c, z), dim=1))

        # Expert networks
        expert_poses = []
        for expert in self.experts :
            # Go throught the expert with z concatenated at each layer
            layer = c
            for operation in expert:
                layer = operation(torch.cat((z, layer), dim=1))
            expert_poses += [layer]

        # Mixed predicted pose
        mean_predpose = sum([self.coefficients[:, i].view(-1, 1)*expert_poses[i] for i in range(self.n_experts)])

        return mean_predpose, expert_poses

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # X is supposed to be a sequence of 2 poses [previous_pose, pose_to_be_predicted], returns the predicted pose
        # previous_pose can be the ground truth (teacher mode) or the previous prediction (student mode)

        # Get the encoded latent vector
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)

        # Get the previous pose
        previous_pose = x[:, :self.n_features]

        # Go through the decoder with the latent vector and the previous pose
        mean_pose, expert_poses = self.decode(z, previous_pose)

        return mean_pose, expert_poses, mu, log_var, x
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_mse.reset()
        self.val_kl.reset()
        
    def loss_function(self, predicted_poses, x, mu, log_var):
        # Reconstruction loss, MSE between predicted and true pose
        # Average for each expert's prediction mse with the gate's coefficients weighted by the gate coefficients
        gt = x[:, self.n_features:2*self.n_features]
        mse_experts = [torch.mean(pow(predicted_pose - gt, 2), dim=1) for predicted_pose in predicted_poses]
        mse_weighted_batch = [self.coefficients[:,i] * mse_experts[i][:] for i in range(self.n_experts)]
        mse_weighted_avg_batch = sum(mse_weighted_batch)
        mse_total = torch.mean(mse_weighted_avg_batch)

        # KL divergence loss
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), axis=1))
        return (mse_total + self.beta * kld, mse_total, kld)

    
    def training_step(self, batch, batch_idx):
        # Resample with probability p,
        # The backward at every step of the loop
        c = batch[:, :self.n_features]
        for i in range(0, self.prediction_length-1):
            # ground truth pi+1, x is (pi, pi+1)(Bernoulli(p)=True) (pi_hat, pi+1)(Bernoulli(p)=False)

            # Extract ground truth pi+1 from the batch
            x_gt = batch[:, (i+1)*self.n_features:(i+2)*self.n_features]
            x = torch.cat((c, x_gt), dim=1)

            # Get the predictions
            predicted_pose, predposes, mu, log_var, x = self.forward(x)
            # Evaluate the loss
            loss, mse, kld = self.loss_function(predposes, x, mu, log_var)
            # Recompute the gradient 
            self.optimizer.zero_grad()
            self.manual_backward(loss)
            self.optimizer.step()

            
            # Resample with probability 1-p
            if torch.bernoulli(torch.tensor(self.p)):
                c = x_gt
            else:
                c = predicted_pose.detach()

            # update and log metrics
            self.train_mse(mse)
            self.log("train/mse", self.train_mse, on_step=True, on_epoch=True, prog_bar=True)

            self.train_kl(kld)
            self.log("train/kl", self.train_kl, on_step=True, on_epoch=True, prog_bar=True)

            self.train_loss(loss)
            self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

            # Compute mse wrt previous pose, Only for visualization purposes
            diff = torch.pow(batch[:, i*self.n_features:(i+1)*self.n_features] - batch[:, (i+1)*self.n_features:(i+2)*self.n_features], 2)
            diff_mean_features= torch.mean(diff, dim=1)
            diff_avg_batch = torch.mean(diff_mean_features)
            mse_wrt_prev = mse/(diff_avg_batch + 1e-12)
            self.mse_wrt_prev(mse_wrt_prev)
            self.log("train/mse_wrt_prev", self.mse_wrt_prev, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def on_train_epoch_end(self):
        self.current_epoch
        schedule = 20*[1.0] + [1-i/10 for i in range(10)] + 140*[0.0]
        #schedule =140*[0.0]
        self.log("p", self.p, on_step=False, on_epoch=True, prog_bar=True)

        lr = self.initial_lr - (self.initial_lr - self.final_lr) * self.current_epoch / float(len(schedule))
        self.optimizer.param_groups[0]['lr'] = lr

        self.log("lr", self.optimizer.param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True)

        self.p = schedule[self.current_epoch]


    
    def validation_step(self, batch, batch_idx):
        x = batch[:, :2*self.n_features]
        mean_pose, predposes, mu, log_var, x = self.forward(x)
        loss, mse, kld = self.loss_function(predposes, x, mu, log_var)

        # update and log metrics
        self.val_mse(mse)
        self.log("val/mse", self.val_mse, on_step=True, on_epoch=True, prog_bar=True)

        self.val_kl(kld)
        self.log("val/kl", self.val_kl, on_step=True, on_epoch=True, prog_bar=True)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x = batch[:, :2*self.n_features]
        mean_pose, predposes, mu, log_var, x = self.forward(x)
        loss, mse, kld = self.loss_function(predposes, x, mu, log_var)

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
            self.optimizer = self.hparams.optimizer(params=self.parameters())
            if self.hparams.scheduler is not None:
                self.scheduler = self.hparams.scheduler(optimizer=self.optimizer)
                return {
                    "optimizer": self.optimizer,
                    "lr_scheduler": {
                        "scheduler": self.scheduler,
                        "monitor": "val/loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            return {"optimizer": self.optimizer}

if __name__ == "__main__":
    _ = MVAEModule(None, None, None, None, None, None, None, None)