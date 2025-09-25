#!/bin/sh

#SBATCH --job-name=ffcv        # Job name
#SBATCH --output=/path/to/ReprSimCauses/repr_sim/log_dir/slurm_out/%j.out            # Output file name (%j expands to jobID)
#SBATCH --error=/path/to/ReprSimCauses/repr_sim/log_dir/slurm_out/%j.err             # Error file name (%j expands to jobID)
#SBATCH --partition=your_account-gpu           # Partition name
#SBATCH --time=6-23:59:59            # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node
#SBATCH --cpus-per-task=6          # Number of CPU cores per task
#SBATCH --mem=150G                   # Memory required per node
#SBATCH --gres=gpu:1
#SBATCH --account=your_account

# Load any required modules here

# info: run like SEED=int sbatch run_train.sh
mkdir -p /path/to/ReprSimCauses/repr_sim/log_dir/slurm_out

source ~/.bashrc  # Ensure bash profile is loaded (only needed if using bash)
eval "$(conda shell.bash hook)"  # Initialize Conda


conda activate data_overlap; cd /path/to/ReprSimCauses/repr_sim


# # Llama, tinycodes, ds_overlap
# # python llm_ds_run_metrics.py \
# #     --log_base_dir "../llm_finetune/log_dir/ds_overlap_bfloat16_tinycodes/meta-llama_Llama-3.2-11B-Vision-Instruct" \
# #     --model_name meta-llama/Llama-3.2-11B-Vision-Instruct --hf_dataset nampdn-ai/tiny-codes --batch_size 20 --max_num_batches 500

# python llm_ds_run_metrics.py \
#     --log_base_dir "../llm_finetune/log_dir/ds_overlap_bfloat16_tinycodes/meta-llama_Llama-3.2-3B-Instruct" \
#     --model_name meta-llama/Llama-3.2-3B-Instruct --hf_dataset nampdn-ai/tiny-codes --batch_size 50 --max_num_batches 200

# python llm_ds_run_metrics.py \
#     --log_base_dir "../llm_finetune/log_dir/ds_overlap_bfloat16_tinycodes/meta-llama_Llama-3.2-1B-Instruct" \
#     --model_name meta-llama/Llama-3.2-1B-Instruct --hf_dataset nampdn-ai/tiny-codes --batch_size 100 --max_num_batches 100
    

# Llama, task overlap
# python llm_task_run_metrics.py \
#     --log_base_dir "../llm_finetune/log_dir/task_overlap_bfloat16/meta-llama_Llama-3.2-11B-Vision-Instruct" \
#     --model_name meta-llama/Llama-3.2-11B-Vision-Instruct --batch_size 20 --max_num_batches 250

# python llm_task_run_metrics.py \
#     --log_base_dir "../llm_finetune/log_dir/task_overlap_bfloat16/meta-llama_Llama-3.2-3B-Instruct" \
#     --model_name meta-llama/Llama-3.2-3B-Instruct --batch_size 50 --max_num_batches 100

# python llm_task_run_metrics.py \
#     --log_base_dir "../llm_finetune/log_dir/task_overlap_bfloat16/meta-llama_Llama-3.2-1B-Instruct" \
#     --model_name meta-llama/Llama-3.2-1B-Instruct --batch_size 100 --max_num_batches 50