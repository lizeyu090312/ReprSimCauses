#!/bin/bash

#SBATCH --job-name=nanoGPT        # Job name
#SBATCH --output=/path/to/ReprSimCauses/llm_finetune/log_dir/slurm_out/%j.out            # Output file name (%j expands to jobID)
#SBATCH --error=/path/to/ReprSimCauses/llm_finetune/log_dir/slurm_out/%j.err             # Error file name (%j expands to jobID)
#SBATCH --partition=your_account-gpu           # Partition name
#SBATCH --time=3-00:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1        # Number of tasks per node
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=80G                   # Memory required per node
#SBATCH --gres=gpu:1
#SBATCH --account=your_account

# info: run like SEED=int sbatch run_train.sh
mkdir -p /path/to/ReprSimCauses/llm_finetune/log_dir/slurm_out

source ~/.bashrc  # Ensure bash profile is loaded (only needed if using bash)
eval "$(conda shell.bash hook)"  # Initialize Conda
conda activate data_overlap; cd /path/to/ReprSimCauses/llm_finetune

# Set the MASTER_ADDR and MASTER_PORT for distributed training. Specifically this 
# avoids an error in training.distributed_mode.init_distributed_mode's torch distributed
# mode init. PyTorch expects these two environment variables to be present, but somehow
# I need to manually set it.
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export TOKENIZERS_PARALLELISM=false


export LOG_DIR_BASE="./log_dir/ds_overlap_test_bfloat16"

# export HF_DATASET="nampdn-ai/tiny-webtext"
export HF_DATASET="nampdn-ai/tiny-codes"

export MAX_STEPS=3
export LOGGING_STEPS=1

# export FRAC_OVERLAP=1   # comment these out when using bash script looping through these
# export SEED=0

# ### largest
# python llm_ds_overlap.py \
#     --hf_dataset $HF_DATASET --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
#     --block_size 512 --max_steps $MAX_STEPS --logging_steps $LOGGING_STEPS --save_strategy no \
#     --per_device_train_batch_size 4 --gradient_accumulation_steps 8 --precision bfloat16 \
#     --bfloat16_compute --auto_preset \
#     --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
#     --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
#     --output_dir_base $LOG_DIR_BASE --seed $SEED --frac_overlap $FRAC_OVERLAP
# echo "passed meta-llama/Llama-3.2-11B-Vision-Instruct $FRAC_OVERLAP" >> test-log.txt

# python llm_ds_overlap.py \
#     --hf_dataset $HF_DATASET --model_name google/gemma-3-12b-it \
#     --block_size 512 --max_steps $MAX_STEPS --logging_steps $LOGGING_STEPS --save_strategy no \
#     --per_device_train_batch_size 1 --gradient_accumulation_steps 64 --precision bfloat16 \
#     --bfloat16_compute --auto_preset \
#     --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
#     --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
#     --output_dir_base $LOG_DIR_BASE --seed $SEED --frac_overlap $FRAC_OVERLAP
# echo "passed google/gemma-3-12b-it $FRAC_OVERLAP" >> test-log.txt




# ### smallest
# python llm_ds_overlap.py \
#     --hf_dataset $HF_DATASET --model_name meta-llama/Llama-3.2-1B-Instruct \
#     --block_size 512 --max_steps $MAX_STEPS --logging_steps $LOGGING_STEPS --save_strategy no \
#     --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --precision bfloat16 \
#     --bfloat16_compute --auto_preset \
#     --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
#     --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
#     --output_dir_base $LOG_DIR_BASE --seed $SEED --frac_overlap $FRAC_OVERLAP
# echo "passed meta-llama/Llama-3.2-1B-Instruct $FRAC_OVERLAP" >> test-log.txt

# python llm_ds_overlap.py \
#     --hf_dataset $HF_DATASET --model_name google/gemma-3-1b-it \
#     --block_size 512 --max_steps $MAX_STEPS --logging_steps $LOGGING_STEPS --save_strategy no \
#     --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --precision bfloat16 \
#     --bfloat16_compute --auto_preset \
#     --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
#     --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
#     --output_dir_base $LOG_DIR_BASE --seed $SEED --frac_overlap $FRAC_OVERLAP
# echo "passed google/gemma-3-1b-it $FRAC_OVERLAP" >> test-log.txt


# ### medium
# python llm_ds_overlap.py \
#     --hf_dataset $HF_DATASET --model_name meta-llama/Llama-3.2-3B-Instruct \
#     --block_size 512 --max_steps $MAX_STEPS --logging_steps $LOGGING_STEPS --save_strategy no \
#     --per_device_train_batch_size 4 --gradient_accumulation_steps 8 --precision bfloat16 \
#     --bfloat16_compute --auto_preset \
#     --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
#     --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
#     --output_dir_base $LOG_DIR_BASE --seed $SEED --frac_overlap $FRAC_OVERLAP


# python llm_ds_overlap.py \
#     --hf_dataset $HF_DATASET --model_name google/gemma-3-4b-it \
#     --block_size 512 --max_steps $MAX_STEPS --logging_steps $LOGGING_STEPS --save_strategy no \
#     --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --precision bfloat16 \
#     --bfloat16_compute --auto_preset \
#     --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
#     --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
#     --output_dir_base $LOG_DIR_BASE --seed $SEED --frac_overlap $FRAC_OVERLAP
