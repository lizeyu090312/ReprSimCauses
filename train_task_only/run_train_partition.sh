#!/bin/sh

#SBATCH --job-name=ffcv        # Job name
#SBATCH --output=/path/to/ReprSimCauses/train_task_only/log_dir_partition/slurm_out/%j.out            # Output file name (%j expands to jobID)
#SBATCH --error=/path/to/ReprSimCauses/train_task_only/log_dir_partition/slurm_out/%j.err             # Error file name (%j expands to jobID)
#SBATCH --partition=gpu-common           # Partition name
#SBATCH --time=16:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node
#SBATCH --cpus-per-task=10          # Number of CPU cores per task
#SBATCH --mem=16G                   # Memory required per node
#SBATCH --gres=gpu:1
#SBATCH --account=your_account
#SBATCH --gres-flags=disable-binding

# Load any required modules here

# info: run like SEED=int sbatch run_train.sh
mkdir -p /path/to/ReprSimCauses/train_task_only/log_dir_partition/slurm_out

source ~/.bashrc  # Ensure bash profile is loaded (only needed if using bash)
eval "$(conda shell.bash hook)"  # Initialize Conda
conda activate ffcv2; cd /path/to/ReprSimCauses/train_task_only/train_task_only

python ffcv_partition1_train.py --training.seed $SEED --output.output_root ./log_dir_partition/partition1 --training.epochs 20
python ffcv_partition2_train.py --training.seed $SEED --output.output_root ./log_dir_partition/partition2 --training.epochs 20
python ffcv_partition3_train.py --training.seed $SEED --output.output_root ./log_dir_partition/partition3 --training.epochs 20
