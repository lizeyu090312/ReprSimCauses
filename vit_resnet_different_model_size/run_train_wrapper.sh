#!/bin/bash


for SEED in {0..3}; do
    # SEED=$SEED sbatch run_train_task_overlap.sh
    # SEED=$SEED sbatch run_train_ds_overlap.sh
    SEED=$SEED sbatch run_train_partition.sh
done
