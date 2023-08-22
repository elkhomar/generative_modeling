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
    # for predictions use trainer.predict(...)
    datamodule.setup()

    # This function extracts a specific index tensor from a list of predictions. Here we get the tensors of the forward pass.
    def extract_tensors_from_prediction(prediction, index):
        tensor_list = [batch[index] for batch in prediction]
        return torch.cat(tensor_list, dim=0)
    
    # This function saves the given tensor to a file with appropriate naming,
    # and optionally saves the associated animation tensor.
    def save_tensors(tensor, subdir, base_dir, prefix, datamodule=None):
        file_path = os.path.join(base_dir, f"{prefix}_{subdir}.pt")
        torch.save(tensor, file_path)
        
        if datamodule:
            anim_tensor = datamodule.train_dataloader().dataset.model_output_to_pose(y=tensor)
            anim_path = os.path.join(base_dir, f"{prefix}_anim_{subdir}.pt")
            torch.save(anim_tensor, anim_path)

    # Get model predictions
    prediction = trainer.predict(model, dataloaders=(datamodule.train_dataloader(shuffle=False), datamodule.check_dataloader()))

    # Extract the tensors from the predictions using list comprehension
    train_tensors, test_tensors = [[extract_tensors_from_prediction(group, i) for i in range(4)] for group in prediction]

    # Split the tensors into separate lists
    xhat_train, mu_train, sigma_train, x_train = train_tensors
    xhat_test, mu_test, sigma_test, x_test = test_tensors

    # Disable gradient calculation
    torch.set_grad_enabled(False)

    # Calculate xdeter tensor
    xdeter_train = torch.cat([model.decode(batch[1]) for batch in prediction[0]], dim=0)
    xdeter_test = torch.cat([model.decode(batch[1]) for batch in prediction[1]], dim=0)

    # Define the output directory
    directory = os.path.dirname(cfg.ckpt_path)

    # Bundle the relevant data in lists for easier organization
    tensor_list = [xhat_train, mu_train, sigma_train, x_train, xdeter_train, xhat_test, mu_test, sigma_test, x_test, xdeter_test]
    prefix_list = ["xhat", "mu", "log_sigma", "x", "xdeter"] * 2
    subdirs = ["train"] * 5 + ["test"] * 5
    datamodules = [datamodule if prefix in ["xhat", "x", "xdeter"] else None for prefix in prefix_list]

    # Iterate over the bundled data and save the respective tensors
    for tensor, subdir, prefix, dm in zip(tensor_list, subdirs, prefix_list, datamodules):
        save_tensors(tensor, subdir, directory, prefix, dm)


    metric_dict = trainer.callback_metrics 

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig) -> None:
    # set precision for matrix multiplication
    torch.set_float32_matmul_precision('medium')
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)


    infer(cfg)


if __name__ == "__main__":
    main()