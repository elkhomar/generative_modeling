#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

#python src/train.py -m experiment=f_test model.latent_dim=4,6,8,16,32,64
#python src/train.py -m experiment=f_test model.optimizer.lr=1e-1,1e-2,1e-3,1e-4,1e-5,1e-6
python src/train.py -m experiment=f_test model.activation="ELU","ReLU","Sigmoid","Softmax","Softplus","Softsign","Tanh"
#python src/train.py -m experiment=f_test model.encoder_dims=[32,16,8],[64,32,16],[128,64,32] model.latent_dim=4 model.decoder_dims=[8,16,32],[16,32,64],[32,64,128]