#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py experiment=locomotion_mvae_beta_e1_linear_lr
python src/train.py experiment=locomotion_mvae_beta_e2_linear_lr
python src/train.py experiment=locomotion_mvae_beta_e3_linear_lr
python src/train.py experiment=locomotion_mvae_beta_e4_linear_lr
python src/train.py experiment=locomotion_mvae_beta_e5_linear_lr