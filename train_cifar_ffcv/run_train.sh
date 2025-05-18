#!/bin/sh

#SBATCH --job-name=ffcv        # Job name
#SBATCH --output=/path/to/ReprSimCauses/train_cifar_ffcv/log_dir/slurm_out/%j.out            # Output file name (%j expands to jobID)
#SBATCH --error=/path/to/ReprSimCauses/train_cifar_ffcv/log_dir/slurm_out/%j.err             # Error file name (%j expands to jobID)
#SBATCH --partition=gpu-common           # Partition name
#SBATCH --time=16:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node
#SBATCH --cpus-per-task=10          # Number of CPU cores per task
#SBATCH --mem=26G                   # Memory required per node
#SBATCH --gres=gpu:1
#SBATCH --account=your_account


# Load any required modules here

# info: run like SEED=int sbatch run_train.sh
mkdir -p /path/to/ReprSimCauses/train_cifar_ffcv/log_dir/slurm_out

source ~/.bashrc  # Ensure bash profile is loaded (only needed if using bash)
eval "$(conda shell.bash hook)"  # Initialize Conda
conda activate ffcv2; cd /path/to/ReprSimCauses/train_cifar_ffcv

# CIFAR10
# ds_overlap
python ffcv_cifar10_train_ds_overlap.py --data.class_frac_overlap 0 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar10
python ffcv_cifar10_train_ds_overlap.py --data.class_frac_overlap 0.1 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar10
python ffcv_cifar10_train_ds_overlap.py --data.class_frac_overlap 0.2 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar10
python ffcv_cifar10_train_ds_overlap.py --data.class_frac_overlap 0.3 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar10
python ffcv_cifar10_train_ds_overlap.py --data.class_frac_overlap 0.4 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar10
python ffcv_cifar10_train_ds_overlap.py --data.class_frac_overlap 0.5 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar10
python ffcv_cifar10_train_ds_overlap.py --data.class_frac_overlap 0.6 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar10
python ffcv_cifar10_train_ds_overlap.py --data.class_frac_overlap 0.7 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar10
python ffcv_cifar10_train_ds_overlap.py --data.class_frac_overlap 0.8 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar10
python ffcv_cifar10_train_ds_overlap.py --data.class_frac_overlap 0.9 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar10
python ffcv_cifar10_train_ds_overlap.py --data.class_frac_overlap 1.0 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar10

# task_overlap
python ffcv_cifar10_train_task_overlap.py --data.class_frac_overlap 0 --training.seed $SEED --output.output_root ./log_dir/task_overlap_cifar10
python ffcv_cifar10_train_task_overlap.py --data.class_frac_overlap 0.2 --training.seed $SEED --output.output_root ./log_dir/task_overlap_cifar10
python ffcv_cifar10_train_task_overlap.py --data.class_frac_overlap 0.4 --training.seed $SEED --output.output_root ./log_dir/task_overlap_cifar10
python ffcv_cifar10_train_task_overlap.py --data.class_frac_overlap 0.6 --training.seed $SEED --output.output_root ./log_dir/task_overlap_cifar10
python ffcv_cifar10_train_task_overlap.py --data.class_frac_overlap 0.8 --training.seed $SEED --output.output_root ./log_dir/task_overlap_cifar10
python ffcv_cifar10_train_task_overlap.py --data.class_frac_overlap 1.0 --training.seed $SEED --output.output_root ./log_dir/task_overlap_cifar10

# CIFAR100
# ds_overlap
python ffcv_cifar100_train_ds_overlap.py --data.class_frac_overlap 0 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar100
python ffcv_cifar100_train_ds_overlap.py --data.class_frac_overlap 0.1 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar100
python ffcv_cifar100_train_ds_overlap.py --data.class_frac_overlap 0.2 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar100
python ffcv_cifar100_train_ds_overlap.py --data.class_frac_overlap 0.3 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar100
python ffcv_cifar100_train_ds_overlap.py --data.class_frac_overlap 0.4 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar100
python ffcv_cifar100_train_ds_overlap.py --data.class_frac_overlap 0.5 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar100
python ffcv_cifar100_train_ds_overlap.py --data.class_frac_overlap 0.6 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar100
python ffcv_cifar100_train_ds_overlap.py --data.class_frac_overlap 0.7 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar100
python ffcv_cifar100_train_ds_overlap.py --data.class_frac_overlap 0.8 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar100
python ffcv_cifar100_train_ds_overlap.py --data.class_frac_overlap 0.9 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar100
python ffcv_cifar100_train_ds_overlap.py --data.class_frac_overlap 1.0 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_cifar100

# task_overlap
python ffcv_cifar100_train_task_overlap.py --data.class_frac_overlap 0 --training.seed $SEED --output.output_root ./log_dir/task_overlap_cifar100
python ffcv_cifar100_train_task_overlap.py --data.class_frac_overlap 0.2 --training.seed $SEED --output.output_root ./log_dir/task_overlap_cifar100
python ffcv_cifar100_train_task_overlap.py --data.class_frac_overlap 0.4 --training.seed $SEED --output.output_root ./log_dir/task_overlap_cifar100
python ffcv_cifar100_train_task_overlap.py --data.class_frac_overlap 0.6 --training.seed $SEED --output.output_root ./log_dir/task_overlap_cifar100
python ffcv_cifar100_train_task_overlap.py --data.class_frac_overlap 0.8 --training.seed $SEED --output.output_root ./log_dir/task_overlap_cifar100
python ffcv_cifar100_train_task_overlap.py --data.class_frac_overlap 1.0 --training.seed $SEED --output.output_root ./log_dir/task_overlap_cifar100
