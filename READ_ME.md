# Generative modeling project by Omar El khalifi and Sebastien Vol

This repository contains the code of a full ML pipeline that generates samples of financial data similar to the provided data_train_log_return.csv

It is based on the following repo : https://github.com/ashleve/lightning-hydra-template that allows to build complex ML pipelines with hydra and takes into account bayesian hyper parameter tuning using optuna.

The files are separated into config .yaml files and src .py files

the metrics for the project are located in src/custom_metrics.py
the models are found in src/models

gpu/cpu can be changed in config/trainer/default