#!/bin/bash

#SBATCH --job-name=ffcv        # Job name
#SBATCH --output=/path/to/ReprSimCauses/vit_resnet_different_model_size/log_dir/slurm_out/%j.out            # Output file name (%j expands to jobID)
#SBATCH --error=/path/to/ReprSimCauses/vit_resnet_different_model_size/log_dir/slurm_out/%j.err             # Error file name (%j expands to jobID)
#SBATCH --partition=your_account-gpu           # Partition name
#SBATCH --time=16:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node
#SBATCH --cpus-per-task=10          # Number of CPU cores per task
#SBATCH --mem=40G                   # Memory required per node
#SBATCH --gres=gpu:1
#SBATCH --account=your_account



# Load any required modules here

# info: run like SEED=int sbatch run_train.sh
mkdir -p /path/to/ReprSimCauses/vit_resnet_different_model_size/log_dir/slurm_out

source ~/.bashrc  # Ensure bash profile is loaded (only needed if using bash)
eval "$(conda shell.bash hook)"  # Initialize Conda
conda activate ffcv2; cd /path/to/ReprSimCauses/vit_resnet_different_model_size


# # -------------- dataaset overlap
# export DS="tinyimagenet"
# ### ResNet
# models=("resnet18" "resnet50" "resnet101" "resnet152")
# for model in "${models[@]}"; do
#     export MODEL=$model
#     python transfer_att_ds_overlap.py --dataset $DS --model $model --max_adv_images 500
# done

# ### ViT
# models=("vit_tiny_patch16_224" "vit_small_patch32_224" "vit_base_patch32_224" "vit_large_patch32_224")
# for model in "${models[@]}"; do
#     export MODEL=$model
#     python transfer_att_ds_overlap.py --dataset $DS --model $MODEL --max_adv_images 500
# done


# -------------- task overlap
export DS="tinyimagenet"
### ResNet
models=("resnet18" "resnet50" "resnet101" "resnet152")
for model in "${models[@]}"; do
    export MODEL=$model
    python transfer_att_task_overlap.py --dataset $DS --model $model --max_adv_images 500
done

### ViT
models=("vit_tiny_patch16_224" "vit_small_patch32_224" "vit_base_patch32_224" "vit_large_patch32_224")
for model in "${models[@]}"; do
    export MODEL=$model
    python transfer_att_task_overlap.py --dataset $DS --model $MODEL --max_adv_images 500
done