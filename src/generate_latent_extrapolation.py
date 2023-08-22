from typing import List, Tuple

import os
import hydra
import pyrootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper 
def visualize(cfg: DictConfig):
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    ### Model and datamodule instantiation

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    cfg.model.input_dim = datamodule.n_features
    
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.get_class(cfg.model._target_).load_from_checkpoint(checkpoint_path = cfg.ckpt_path)

    log.info("Starting visualisation!")

    datamodule.setup()

    ### Trainer instantiation
    # for predictions use trainer.predict(...)

    # from mu-3 sigma to mu+3 sigma, note that we are using the predicted sigma of the pose
    def translation(mu, sigma, k, s) :
        n = s * 30
        weights = torch.tensor([6*torch.exp(sigma[k])*(i-n/2)/(n) for i in range(n)])
        mus = mu.repeat(n, 1)
        mus[:, k] += weights
        return mus

    # Disable gradient calculation
    torch.set_grad_enabled(False)

    ### Create extrapolation anim for s seconds

    s = 30

    directory = os.path.dirname(cfg.ckpt_path)

    # Base poses
    latent_zero_pose = torch.zeros_like(torch.load(directory + '/mu_test.pt')[0])
    T_pose = torch.load(directory + '/mu_test.pt')[0]

    base_mus = [latent_zero_pose, T_pose]
    #base_sigma = [torch.load(directory + '/log_sigma_test.pt')[0], torch.zeros_like(torch.load(directory + '/mu_test.pt')[0])]      
    # sigma = one
    base_sigma = torch.ones_like(latent_zero_pose)

    for i, base_mu in enumerate(base_mus) :
        # Create directory for the extrapolation
        os.mkdir(directory + f'/latent_extrapolation_{cfg.base_pose[i]}')


        for k in range(len(base_mu)) :
            extrapolation_mu = translation(base_mu, base_sigma, k, s)
            extrapolation_xhat = torch.cat([model.decode(extrapolation_mu)])
            extrapolation_anim = datamodule.train_dataloader().dataset.model_output_to_pose(y=extrapolation_xhat)
            torch.save(extrapolation_anim, directory + f'/latent_extrapolation_{cfg.base_pose[i]}/extrapolate_latent_{k}')
    

@hydra.main(version_base="1.3", config_path="../configs", config_name="generate_latent_extrapolation.yaml")
def main(cfg: DictConfig) -> None:
    # set precision for matrix multiplication
    torch.set_float32_matmul_precision('medium')
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    visualize(cfg)


if __name__ == "__main__":
    main()
