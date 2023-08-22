#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

#python src/train.py -m experiment=all_lafan_mvae_root_space_base_moe_linear_lr model.beta=2e0,2e-1,2e-2,2e-3
python src/train.py -m experiment=all_lafan_mvae_root_space_base_moe_linear_lr model.n_experts=16 model.beta=2e-3,2e0
