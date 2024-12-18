#!/bin/bash

#SBATCH -J 500
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8

python main.py --dataset=ml-1m --train_dir=default


exit 0