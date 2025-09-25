#!/bin/bash

#SBATCH --job-name=ffcv        # Job name
#SBATCH --output=/path/to/ReprSimCauses/vit_resnet_different_model_size/log_dir/slurm_out/%j.out            # Output file name (%j expands to jobID)
#SBATCH --error=/path/to/ReprSimCauses/vit_resnet_different_model_size/log_dir/slurm_out/%j.err             # Error file name (%j expands to jobID)
#SBATCH --partition=your_account-gpu           # Partition name
#SBATCH --time=96:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1        # Number of tasks per node
#SBATCH --cpus-per-task=6          # Number of CPU cores per task
#SBATCH --mem=48G                   # Memory required per node
#SBATCH --gres=gpu:1
#SBATCH --account=your_account



# Load any required modules here

# info: run like SEED=int sbatch run_train.sh
mkdir -p /path/to/ReprSimCauses/vit_resnet_different_model_size/log_dir/slurm_out

source ~/.bashrc  # Ensure bash profile is loaded (only needed if using bash)
eval "$(conda shell.bash hook)"  # Initialize Conda


conda activate ffcv2; cd /path/to/ReprSimCauses/vit_resnet_different_model_size


# python ds_overlap_run_metrics_ffcv.py --dataset tinyimagenet --model resnet152
# python ds_overlap_run_metrics_ffcv.py --dataset tinyimagenet --model vit_large_patch32_224


## ds overlap
# python ds_overlap_run_metrics_ffcv.py --dataset $DATASET --model $MODEL

## task overlap
# python task_overlap_run_metrics_ffcv.py --dataset $DATASET --model $MODEL

## pretrained models
# python pretrained_models_run_metrics_ffcv.py

## ColorShapeDigit800k
python partition_run_metrics_ffcv.py --model $MODEL
