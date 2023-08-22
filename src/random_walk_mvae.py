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
def infer(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    cfg.model.input_dim = datamodule.n_features
    
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.get_class(cfg.model._target_).load_from_checkpoint(checkpoint_path = cfg.ckpt_path)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }

    log.info("Starting inference!")
    datamodule.setup()

    # Generate sequence of z that are latent_dim dimensional, first is standard normal, second is zero, third to latent_dim + 2 is ones along a latent dimension
    z_single = -torch.eye(model.hparams.latent_dim, model.hparams.latent_dim).unsqueeze(1)
    z_single = z_single.repeat(1, 1000, 1)
    z_random = torch.randn(1000, model.hparams.latent_dim).reshape(1, 1000, model.hparams.latent_dim)


    z_zero = torch.zeros(1000, model.hparams.latent_dim).reshape(1, 1000, model.hparams.latent_dim)
    zs = torch.cat([z_random, z_zero, z_single])

    # Define the pose from which to start the random walk
    dataset = datamodule.train_dataloader().dataset
    base_pose = dataset[0][:dataset.n_features]

    # Disable gradient calculation
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)

    zs = zs.to("cuda")
    model.to('cuda')

    # Generate the random walk with the decoder of the model

    for i, trajectory in enumerate(zs):
        poses = base_pose.unsqueeze(0)
        poses = poses.to("cuda")
        for z in trajectory:
            z = z.to("cuda").unsqueeze(0)
            previous_pose = poses[-1:]
            poses = torch.cat([poses, model.decode(z, previous_pose)])

        anim = datamodule.train_dataloader().dataset.model_output_to_pose(y=poses)
        
        # Define the output directory
        directory = os.path.dirname(cfg.ckpt_path)
        file_path = os.path.join(directory, f"walk_{i}.pt")
        torch.save(anim, file_path)


    metric_dict = trainer.callback_metrics 

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="random_walk.yaml")
def main(cfg: DictConfig) -> None:
    # set precision for matrix multiplication
    torch.set_float32_matmul_precision('medium')
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)


    infer(cfg)


if __name__ == "__main__":
    main()