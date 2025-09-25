## llama
export FRAC_OVERLAP=0.2
export SEED=0
export LOGGING_STEPS=1
export MAX_STEPS=2
export LOG_DIR_BASE="./log_dir/data_writing"

source ~/.bashrc
eval "$(conda shell.bash hook)"  # Initialize Conda
conda activate data_overlap

python llm_ds_overlap.py \
    --hf_dataset nampdn-ai/tiny-textbooks --model_name meta-llama/Llama-3.2-1B-Instruct \
    --block_size 512 --max_steps $MAX_STEPS --logging_steps $LOGGING_STEPS --save_strategy no \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --precision bfloat16 \
    --bfloat16_compute --auto_preset \
    --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
    --output_dir_base $LOG_DIR_BASE --seed $SEED --frac_overlap 0
echo "tiny-textbooks" > out.txt

python llm_ds_overlap.py \
    --hf_dataset Salesforce/wikitext --model_name meta-llama/Llama-3.2-1B-Instruct \
    --block_size 512 --max_steps $MAX_STEPS --logging_steps $LOGGING_STEPS --save_strategy no \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --precision bfloat16 \
    --bfloat16_compute --auto_preset \
    --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
    --output_dir_base $LOG_DIR_BASE --seed $SEED --frac_overlap $FRAC_OVERLAP
echo "wikitext" >> out.txt

python llm_ds_overlap.py \
    --hf_dataset roneneldan/TinyStories --model_name meta-llama/Llama-3.2-1B-Instruct \
    --block_size 512 --max_steps $MAX_STEPS --logging_steps $LOGGING_STEPS --save_strategy no \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --precision bfloat16 \
    --bfloat16_compute --auto_preset \
    --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
    --output_dir_base $LOG_DIR_BASE --seed $SEED --frac_overlap $FRAC_OVERLAP
echo "TinyStories" >> out.txt

python llm_ds_overlap.py \
    --hf_dataset Trelis/tiny-shakespeare --model_name meta-llama/Llama-3.2-1B-Instruct \
    --block_size 512 --max_steps $MAX_STEPS --logging_steps $LOGGING_STEPS --save_strategy no \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --precision bfloat16 \
    --bfloat16_compute --auto_preset \
    --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
    --output_dir_base $LOG_DIR_BASE --seed $SEED --frac_overlap $FRAC_OVERLAP
echo "tiny-shakespeare" >> out.txt
