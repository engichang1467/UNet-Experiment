#!/bin/bash  

# set the name of datasets you would like to use for traning/validation
train_dataset_name="rb2d_ra1e6_s1_bpvv.npz"
eval_dataset_name="rb2d_ra1e6_s1_bpvv.npz"

log_dir_name="./log/Exp1_baseline"
mkdir -p $log_dir_name

# please see the train.py for further tunable arguments during the training process
echo "[!] If you run into OOM error, try reducing batch_size_per_gpu or n_samp_pts_per_crop..."
CUDA_VISIBLE_DEVICES=0 python3 train_baseline.py --epochs=10 --data_folder=data --log_dir=$log_dir_name --train_data=$train_dataset_name --eval_data=$eval_dataset_name --batch_size_per_gpu=10
