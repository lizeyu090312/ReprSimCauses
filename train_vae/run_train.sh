#!/bin/sh

#SBATCH --job-name=ffcv        # Job name
#SBATCH --output=/hpc/group/wengerlab/zl310/data_overlap/train_vae/log_dir/slurm_out/%j.out            # Output file name (%j expands to jobID)
#SBATCH --error=/hpc/group/wengerlab/zl310/data_overlap/train_vae/log_dir/slurm_out/%j.err             # Error file name (%j expands to jobID)
#SBATCH --partition=gpu-common           # Partition name
#SBATCH --time=16:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node
#SBATCH --cpus-per-task=10          # Number of CPU cores per task
#SBATCH --mem=26G                   # Memory required per node
#SBATCH --gres=gpu:1
#SBATCH --account=wengerlab
#SBATCH --nodelist=dcc-core-gpu-09,dcc-core-gpu-10,dcc-core-gpu-11,dcc-core-gpu-26,dcc-core-gpu-27,dcc-core-gpu-28,dcc-core-gpu-29,dcc-core-gpu-30,dcc-core-gpu-31,dcc-core-gpu-32,dcc-core-gpu-33,dcc-core-gpu-34,dcc-core-gpu-35,dcc-core-gpu-36,dcc-core-gpu-37,dcc-core-gpu-38,dcc-core-gpu-39,dcc-core-gpu-40,dcc-core-gpu-41,dcc-core-gpu-42,dcc-core-gpu-43,dcc-core-gpu-44,dcc-core-gpu-45,dcc-core-gpu-46

# Load any required modules here

# info: run like SEED=int sbatch run_train.sh
mkdir -p /hpc/group/wengerlab/zl310/data_overlap/train_vae/log_dir/slurm_out

source ~/.bashrc  # Ensure bash profile is loaded (only needed if using bash)
eval "$(conda shell.bash hook)"  # Initialize Conda
conda activate ffcv2; cd /hpc/group/wengerlab/zl310/data_overlap/train_vae


# Horizontal
python cifar10_train_ds_overlap.py --frac_overlap 0 --seed $SEED --output_root ./log_dir/ds_overlap_cifar10
python cifar10_train_ds_overlap.py --frac_overlap 0.1 --seed $SEED --output_root ./log_dir/ds_overlap_cifar10
python cifar10_train_ds_overlap.py --frac_overlap 0.2 --seed $SEED --output_root ./log_dir/ds_overlap_cifar10
python cifar10_train_ds_overlap.py --frac_overlap 0.3 --seed $SEED --output_root ./log_dir/ds_overlap_cifar10
python cifar10_train_ds_overlap.py --frac_overlap 0.4 --seed $SEED --output_root ./log_dir/ds_overlap_cifar10
python cifar10_train_ds_overlap.py --frac_overlap 0.5 --seed $SEED --output_root ./log_dir/ds_overlap_cifar10
python cifar10_train_ds_overlap.py --frac_overlap 0.6 --seed $SEED --output_root ./log_dir/ds_overlap_cifar10
python cifar10_train_ds_overlap.py --frac_overlap 0.7 --seed $SEED --output_root ./log_dir/ds_overlap_cifar10
python cifar10_train_ds_overlap.py --frac_overlap 0.8 --seed $SEED --output_root ./log_dir/ds_overlap_cifar10
python cifar10_train_ds_overlap.py --frac_overlap 0.9 --seed $SEED --output_root ./log_dir/ds_overlap_cifar10
python cifar10_train_ds_overlap.py --frac_overlap 1.0 --seed $SEED --output_root ./log_dir/ds_overlap_cifar10

# Vertical fixed dataset length
python cifar10_train_task_overlap.py --frac_overlap 0 --seed $SEED --output_root ./log_dir/task_overlap_cifar10
python cifar10_train_task_overlap.py --frac_overlap 0.2 --seed $SEED --output_root ./log_dir/task_overlap_cifar10
python cifar10_train_task_overlap.py --frac_overlap 0.4 --seed $SEED --output_root ./log_dir/task_overlap_cifar10
python cifar10_train_task_overlap.py --frac_overlap 0.6 --seed $SEED --output_root ./log_dir/task_overlap_cifar10
python cifar10_train_task_overlap.py --frac_overlap 0.8 --seed $SEED --output_root ./log_dir/task_overlap_cifar10
python cifar10_train_task_overlap.py --frac_overlap 1.0 --seed $SEED --output_root ./log_dir/task_overlap_cifar10
