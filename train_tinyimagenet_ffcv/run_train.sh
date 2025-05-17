#!/bin/bash

#SBATCH --job-name=ffcv        # Job name
#SBATCH --output=/hpc/group/wengerlab/zl310/data_overlap/train_tinyimagenet_ffcv/log_dir/slurm_out/%j.out            # Output file name (%j expands to jobID)
#SBATCH --error=/hpc/group/wengerlab/zl310/data_overlap/train_tinyimagenet_ffcv/log_dir/slurm_out/%j.err             # Error file name (%j expands to jobID)
#SBATCH --partition=wengerlab-gpu           # Partition name
#SBATCH --time=16:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node
#SBATCH --cpus-per-task=10          # Number of CPU cores per task
#SBATCH --mem=26G                   # Memory required per node
#SBATCH --gres=gpu:1
#SBATCH --account=wengerlab


# #!/bin/bash
# a 
# #SBATCH --job-name=ffcv        # Job name
# #SBATCH --output=/hpc/group/wengerlab/zl310/data_overlap/train_tinyimagenet_ffcv/log_dir/slurm_out/%j.out            # Output file name (%j expands to jobID)
# #SBATCH --error=/hpc/group/wengerlab/zl310/data_overlap/train_tinyimagenet_ffcv/log_dir/slurm_out/%j.err             # Error file name (%j expands to jobID)
# #SBATCH --partition=h200ea           # Partition name
# #SBATCH --time=16:00:00            # Maximum runtime (HH:MM:SS)
# #SBATCH --nodes=1                  # Number of nodes
# #SBATCH --ntasks-per-node=1        # Number of tasks per node
# #SBATCH --cpus-per-task=10          # Number of CPU cores per task
# #SBATCH --mem=26G                   # Memory required per node
# #SBATCH --gres=gpu:h200_1g.18gb:1
# #SBATCH --account=h200ea



# Load any required modules here

# info: run like SEED=int sbatch run_train.sh
mkdir -p /hpc/group/wengerlab/zl310/data_overlap/train_tinyimagenet_ffcv/log_dir/slurm_out

source ~/.bashrc  # Ensure bash profile is loaded (only needed if using bash)
eval "$(conda shell.bash hook)"  # Initialize Conda
conda activate ffcv2; cd /hpc/group/wengerlab/zl310/data_overlap/train_tinyimagenet_ffcv

# ds_overlap
python ffcv_tinyimagenet_train_ds_overlap.py --data.class_frac_overlap 0 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_tinyimagenet
python ffcv_tinyimagenet_train_ds_overlap.py --data.class_frac_overlap 0.1 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_tinyimagenet
python ffcv_tinyimagenet_train_ds_overlap.py --data.class_frac_overlap 0.2 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_tinyimagenet
python ffcv_tinyimagenet_train_ds_overlap.py --data.class_frac_overlap 0.3 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_tinyimagenet
python ffcv_tinyimagenet_train_ds_overlap.py --data.class_frac_overlap 0.4 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_tinyimagenet
python ffcv_tinyimagenet_train_ds_overlap.py --data.class_frac_overlap 0.5 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_tinyimagenet
python ffcv_tinyimagenet_train_ds_overlap.py --data.class_frac_overlap 0.6 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_tinyimagenet
python ffcv_tinyimagenet_train_ds_overlap.py --data.class_frac_overlap 0.7 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_tinyimagenet
python ffcv_tinyimagenet_train_ds_overlap.py --data.class_frac_overlap 0.8 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_tinyimagenet
python ffcv_tinyimagenet_train_ds_overlap.py --data.class_frac_overlap 0.9 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_tinyimagenet
python ffcv_tinyimagenet_train_ds_overlap.py --data.class_frac_overlap 1.0 --training.seed $SEED --output.output_root ./log_dir/ds_overlap_tinyimagenet


# task_overlap fixed dataset length
python ffcv_tinyimagenet_train_task_overlap.py --data.class_frac_overlap 0 --training.seed $SEED --output.output_root ./log_dir/task_overlap_tinyimagenet
python ffcv_tinyimagenet_train_task_overlap.py --data.class_frac_overlap 0.2 --training.seed $SEED --output.output_root ./log_dir/task_overlap_tinyimagenet
python ffcv_tinyimagenet_train_task_overlap.py --data.class_frac_overlap 0.4 --training.seed $SEED --output.output_root ./log_dir/task_overlap_tinyimagenet
python ffcv_tinyimagenet_train_task_overlap.py --data.class_frac_overlap 0.6 --training.seed $SEED --output.output_root ./log_dir/task_overlap_tinyimagenet
python ffcv_tinyimagenet_train_task_overlap.py --data.class_frac_overlap 0.8 --training.seed $SEED --output.output_root ./log_dir/task_overlap_tinyimagenet
python ffcv_tinyimagenet_train_task_overlap.py --data.class_frac_overlap 1.0 --training.seed $SEED --output.output_root ./log_dir/task_overlap_tinyimagenet
