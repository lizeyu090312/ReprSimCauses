#!/bin/bash

# Define arrays, ds overlap (shakespeare)
seeds=(0 1 2 3 4 5)
frac_overlaps=(0.0 0.2 0.4 0.6 0.8 1.0)

# Loop over combinations
for seed in "${seeds[@]}"; do
  for frac in "${frac_overlaps[@]}"; do
    export SEED=$seed
    export FRAC_OVERLAP=$frac
    export DATASET="shakespeare_all"
    sbatch run_train_ds_overlap_slurm.sh
  done
done


# Define arrays, ds overlap (tinystories)
seeds=(0 1 2 3 4 5)
frac_overlaps=(0.0 0.2 0.4 0.6 0.8 1.0)

# Loop over combinations
for seed in "${seeds[@]}"; do
  for frac in "${frac_overlaps[@]}"; do
    export SEED=$seed
    export FRAC_OVERLAP=$frac
    export DATASET="tinystories"
    sbatch run_train_ds_overlap_slurm.sh
  done
done


# Define arrays, task overlap
seeds=(0 1 2 3 4 5)
frac_overlaps=(0.0 0.33333 0.66666 1.0)

# Loop over combinations
for seed in "${seeds[@]}"; do
  for frac in "${frac_overlaps[@]}"; do
    export SEED=$seed
    export FRAC_OVERLAP=$frac
    sbatch run_train_task_overlap_slurm.sh
  done
done
