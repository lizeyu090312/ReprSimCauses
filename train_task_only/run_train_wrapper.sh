#!/bin/bash

for SEED in {0..9}; do
    SEED=$SEED sbatch run_train_partition.sh
done
