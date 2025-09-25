#!/bin/bash

for SEED in {0..3}; do
    SEED=$SEED sbatch run_train.sh
done
