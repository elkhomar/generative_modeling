#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

#python src/train.py -m experiment=all_lafan_mvae_root_space_base_moe_linear_lr model.beta=2e0,2e-1,2e-2,2e-3
python src/train.py -m experiment=locomotion_mvae_root_space_base_moe_ss_bis_linear_lr data.data_dir="data/LAFAN/" model.n_experts=16