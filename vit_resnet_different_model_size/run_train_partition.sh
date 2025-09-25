#!/bin/bash

#SBATCH --job-name=ffcv        # Job name
#SBATCH --output=/path/to/ReprSimCauses/vit_resnet_different_model_size/log_dir_partition/slurm_out/%j.out            # Output file name (%j expands to jobID)
#SBATCH --error=/path/to/ReprSimCauses/vit_resnet_different_model_size/log_dir_partition/slurm_out/%j.err             # Error file name (%j expands to jobID)
#SBATCH --partition=your_account-gpu           # Partition name
#SBATCH --time=128:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node
#SBATCH --cpus-per-task=10          # Number of CPU cores per task
#SBATCH --mem=80G                   # Memory required per node
#SBATCH --gres=gpu:1
#SBATCH --account=your_account


# Load any required modules here

# info: run like SEED=int sbatch run_train.sh
mkdir -p /path/to/ReprSimCauses/vit_resnet_different_model_size/log_dir_partition/slurm_out

source ~/.bashrc  # Ensure bash profile is loaded (only needed if using bash)
eval "$(conda shell.bash hook)"  # Initialize Conda
conda activate ffcv2; cd /path/to/ReprSimCauses/vit_resnet_different_model_size


### ResNet
# models=("resnet18" "resnet50" "resnet101" "resnet152")
models=("resnet152")
for model in "${models[@]}"; do
    export MODEL=$model
    # python ffcv_partition3_train.py --model.model_name $MODEL --training.seed $SEED --output.output_root ./log_dir_partition/ --training.lr 0.2 --training.batch_size 256 --training.epochs 5
    # python ffcv_partition2_train.py --model.model_name $MODEL --training.seed $SEED --output.output_root ./log_dir_partition/ --training.lr 0.2 --training.batch_size 256 --training.epochs 5
    python ffcv_partition1_train.py --model.model_name $MODEL --training.seed $SEED --output.output_root ./log_dir_partition/ --training.lr 0.2 --training.batch_size 256 --training.epochs 5
done


### ViT
models=("vit_tiny_patch16_224" "vit_small_patch32_224" "vit_base_patch32_224" "vit_large_patch32_224")
for model in "${models[@]}"; do
    export MODEL=$model
    python ffcv_partition3_train.py --model.model_name $MODEL --training.seed $SEED --output.output_root ./log_dir_partition/ --training.lr 0.0003 --training.batch_size 256 --training.weight_decay 0.05 --training.epochs 5
    python ffcv_partition2_train.py --model.model_name $MODEL --training.seed $SEED --output.output_root ./log_dir_partition/ --training.lr 0.0003 --training.batch_size 256 --training.weight_decay 0.05 --training.epochs 5
    python ffcv_partition1_train.py --model.model_name $MODEL --training.seed $SEED --output.output_root ./log_dir_partition/ --training.lr 0.0003 --training.batch_size 256 --training.weight_decay 0.05 --training.epochs 5
done
