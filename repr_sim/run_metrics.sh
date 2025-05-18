#!/bin/sh

#SBATCH --job-name=ffcv        # Job name
#SBATCH --output=/path/to/ReprSimCauses/platonic_rep/log_dir/slurm_out/%j.out            # Output file name (%j expands to jobID)
#SBATCH --error=/path/to/ReprSimCauses/platonic_rep/log_dir/slurm_out/%j.err             # Error file name (%j expands to jobID)
#SBATCH --partition=your_account-gpu           # Partition name
#SBATCH --time=16:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1        # Number of tasks per node
#SBATCH --cpus-per-task=10          # Number of CPU cores per task
#SBATCH --mem=28G                   # Memory required per node
#SBATCH --gres=gpu:1
#SBATCH --account=your_account
#SBATCH --gres-flags=disable-binding

# Load any required modules here

# info: run like SEED=int sbatch run_train.sh
mkdir -p /path/to/ReprSimCauses/platonic_rep/log_dir/slurm_out

source ~/.bashrc  # Ensure bash profile is loaded (only needed if using bash)
eval "$(conda shell.bash hook)"  # Initialize Conda


# conda activate data_overlap; cd /path/to/ReprSimCauses/platonic_rep
conda activate ffcv2; cd /path/to/ReprSimCauses/platonic_rep
python partition_run_metrics_ffcv.py --log_base_dir ../train_task_only/log_dir_partition


python cifar10_ds_overlap_run_metrics.py --log_base_dir ../train_cifar_ffcv/log_dir/ds_overlap_cifar10
python cifar10_task_overlap_run_metrics.py --log_base_dir ../train_cifar_ffcv/log_dir/task_overlap_cifar10


python cifar100_ds_overlap_run_metrics.py --log_base_dir ../train_cifar_ffcv/log_dir/ds_overlap_cifar100
python cifar100_task_overlap_run_metrics.py --log_base_dir ../train_cifar_ffcv/log_dir/task_overlap_cifar100


python tinyiamgenet_ds_overlap_run_metrics.py --log_base_dir ../train_tinyiamgenet_ffcv/log_dir/ds_overlap_tinyiamgenet_shakespeare_all
python tinyiamgenet_ds_overlap_run_metrics.py --log_base_dir ../train_tinyiamgenet_ffcv/log_dir/ds_overlap_tinyiamgenet_tinystories
python tinyiamgenet_task_overlap_run_metrics.py --log_base_dir ../train_tinyiamgenet_ffcv/log_dir/task_overlap_tinyiamgenet


conda activate data_overlap
python vae_cifar10_ds_overlap_run_metrics.py --log_base_dir ../train_vae/log_dir/ds_overlap_cifar10
python vae_cifar10_task_overlap_run_metrics.py --log_base_dir ../train_vae/log_dir/task_overlap_cifar10


python nanogpt_ds_run_metrics.py --log_base_dir ../train_nanoGPT/log_dir/nanogpt_ds_overlap
python nanogpt_task_run_metrics.py --log_base_dir ../train_nanoGPT/log_dir/nanogpt_task_overlap


# train_nanoGPT dataset overlap
# python gpt2_ds_run_metrics.py --log_base_dir ../train_nanoGPT/log_dir/nanogpt_horizontal_babyGPT_shakespeare_all
# python gpt2_ds_run_metrics.py --log_base_dir ../train_nanoGPT/log_dir/nanogpt_horizontal_babyGPT_tinystories

# # train_nanoGPT task overlap
# python gpt2_task_run_metrics.py --log_base_dir ../train_nanoGPT/log_dir/nanogpt_task_babyGPT

# # partitioned datasets
# python partition_run_metrics_ffcv.py --log_base_dir ../combined_dataset_ffcv/log_dir_partition