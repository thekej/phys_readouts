#!/bin/bash
export HDF5_USE_FILE_LOCKING=FALSE
CUDA_VISIBLE_DEVICES=$1 python train_new.py  --env TDWdominoes  --model_name SGNN --log_per_iter 1 --training_fpt 4 --ckp_per_iter 13000 --floor_cheat 1 --outf "" --resume_epoch 140 --resume_iter 13000
