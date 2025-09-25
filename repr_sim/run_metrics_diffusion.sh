#!/bin/bash

#SBATCH --job-name=ffcv        # Job name
#SBATCH --output=/path/to/ReprSimCauses/repr_sim/log_dir/slurm_out/%j.out            # Output file name (%j expands to jobID)
#SBATCH --error=/path/to/ReprSimCauses/repr_sim/log_dir/slurm_out/%j.err             # Error file name (%j expands to jobID)
#SBATCH --partition=your_account-gpu           # Partition name
#SBATCH --time=16:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1        # Number of tasks per node
#SBATCH --cpus-per-task=10          # Number of CPU cores per task
#SBATCH --mem=40G                   # Memory required per node
#SBATCH --gres=gpu:1
#SBATCH --account=your_account


# #SBATCH --partition=your_account-gpu           # Partition name

# Load any required modules here

# info: run like SEED=int sbatch run_train.sh
mkdir -p /path/to/ReprSimCauses/repr_sim/log_dir/slurm_out

source ~/.bashrc  # Ensure bash profile is loaded (only needed if using bash)
eval "$(conda shell.bash hook)"  # Initialize Conda


conda activate ffcv2; cd /path/to/ReprSimCauses/repr_sim

## diffusion, cifar10 dataset overlap
python diffusion_ds_run_metrics.py --log_base_dir ../train_diffusion_unet/log_dir/ds_overlap_train/cifar10 --dsname cifar10

# python diffusion_task_run_metrics.py --log_base_dir ../train_diffusion_unet/log_dir/task_overlap_train/cifar10 --dsname cifar10