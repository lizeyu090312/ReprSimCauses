#!/bin/bash

#SBATCH --job-name=train_nanoGPT        # Job name
#SBATCH --output=/path/to/ReprSimCauses/train_nanoGPT/log_dir/slurm_out/%j.out            # Output file name (%j expands to jobID)
#SBATCH --error=/path/to/ReprSimCauses/train_nanoGPT/log_dir/slurm_out/%j.err             # Error file name (%j expands to jobID)
#SBATCH --partition=gpu-common           # Partition name
#SBATCH --time=2-00:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node
#SBATCH --cpus-per-task=10          # Number of CPU cores per task
#SBATCH --mem=55G                   # Memory required per node
#SBATCH --gres=gpu:1
#SBATCH --account=your_account

# Load any required modules here

# info: run like SEED=int sbatch run_train.sh
mkdir -p /path/to/ReprSimCauses/train_nanoGPT/log_dir/slurm_out

source ~/.bashrc  # Ensure bash profile is loaded (only needed if using bash)
eval "$(conda shell.bash hook)"  # Initialize Conda
conda activate data_overlap; cd /path/to/ReprSimCauses/train_nanoGPT

# Set the MASTER_ADDR and MASTER_PORT for distributed training. Specifically this 
# avoids an error in training.distributed_mode.init_distributed_mode's torch distributed
# mode init. PyTorch expects these two environment variables to be present, but somehow
# I need to manually set it.
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export TOKENIZERS_PARALLELISM=false

export N_LAYER=6
export N_EMBD=384
export N_HEAD=6
export DROPOUT=0.2

export GRAD_ACCUM_ITERS=2
export BSZ=128
export BLOCK_SZ=256
export LR=0.0005

export LOG_DIR_BASE="./log_dir/nanogpt_ds_overlap"

python gpt2_train_ds_overlap.py --dataset $DATASET --seed $SEED --frac_overlap $FRAC_OVERLAP --max_iters 10000 --output_dir_base $LOG_DIR_BASE \
    --n_layer $N_LAYER --n_embd $N_EMBD --n_head $N_HEAD --dropout $DROPOUT  \
    --grad_accum_iters $GRAD_ACCUM_ITERS --batch_size $BSZ --block_size $BLOCK_SZ --max_lr $LR

python sample_ds_overlap.py --dataset $DATASET --seed $SEED --frac_overlap $FRAC_OVERLAP --split 0 --output_dir_base $LOG_DIR_BASE
python sample_ds_overlap.py --dataset $DATASET --seed $SEED --frac_overlap $FRAC_OVERLAP --split 1 --output_dir_base $LOG_DIR_BASE