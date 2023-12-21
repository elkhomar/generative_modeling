#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py -m experiment=f_test model.latent_dim=1,2,3,4,5,6,7,8,16,32,64
#python src/train.py -m experiment=f_test model.encoder_dims=[32,16,8],[64,32,16],[128,64,32] model.latent_dim=4 model.decoder_dims=[8,16,32],[16,32,64],[32,64,128]