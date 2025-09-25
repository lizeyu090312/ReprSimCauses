#!/bin/bash

# Define arrays
# seeds=(0 1 2 3)
# frac_overlaps=(1.0 0.8 0.6 0.4 0.2 0.0)

# # Loop over combinations
# for seed in "${seeds[@]}"; do
#   for frac in "${frac_overlaps[@]}"; do
#     export SEED=$seed
#     export FRAC_OVERLAP=$frac
#     sbatch run_train_ds_overlap_slurm.sh
#   done
# done

# Define arrays
seeds=(0 1 2 3)
frac_overlaps=(0.33333 0.0 0.66666 1.0)

# Loop over combinations
for seed in "${seeds[@]}"; do
  for frac in "${frac_overlaps[@]}"; do
    export SEED=$seed
    export FRAC_OVERLAP=$frac
    sbatch run_train_task_overlap_slurm.sh
  done
done
