#!/bin/sh

#SBATCH --job-name=ffcv        # Job name
#SBATCH --output=/path/to/ReprSimCauses/train_task_only/partition_datagen_%j.out            # Output file name (%j expands to jobID)
#SBATCH --error=/path/to/ReprSimCauses/train_task_only/partition_datagen_%j.err             # Error file name (%j expands to jobID)
#SBATCH --partition=common           # Partition name
#SBATCH --time=23:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1        # Number of tasks per node
#SBATCH --cpus-per-task=10          # Number of CPU cores per task
#SBATCH --mem=28G                   # Memory required per node
#SBATCH --account=your_account

# Load any required modules here

source ~/.bashrc  # Ensure bash profile is loaded (only needed if using bash)
eval "$(conda shell.bash hook)"  # Initialize Conda


conda activate ffcv2; cd /path/to/ReprSimCauses/train_task_only

# python partitioning_final.py val
# python partitioning_final.py train
python ffcv_partition_write_data.py
echo Done