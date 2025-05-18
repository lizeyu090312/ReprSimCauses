# Official Implementation of **Exploring Causes of Representational Similarity in Machine Learning Models**
Under review.

### Introduction
This repository contains the official implementation of **Exploring Causes of Representational Similarity in Machine Learning Models**. Please install two environments: one containing ffcv (refer to https://ffcv.io) and the other using `pip install -r requirements.txt`. 

### Training models
The training scripts can be found under `train**/`. Please run `bash run_write_ffcv_data.sh` if this file is available and then `bash run_train_wrapper.sh` to train the models using task and dataset splitting. Remember to change the paths appropriately. 

**Note for `train_nanoGPT/`**: please manually download any appropriate text files (required for `shakespeare_all`) and run `python prepare.py` for all six directories under `train_nanoGPT/`. 

### Measuring representational similarity
Please run `sbatch run_metrics.sh` in `repr_sim/`. We base our code for computing representational similarity on this repository: https://github.com/minyoungg/platonic-rep. 

### Notes about the TinyImageNet dataset
Please download the dataset from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`. Then, unzip using `unzip path/to/tiny-imagenet-200.zip`. Then, copy `ReprSimCauses/train_tinyimagenet_ffcv/tinyimagenet_dataset_reorg.py` to `path/to/tiny-imagenet-200` and run `cd path/to/tiny-imagenet-200; python tinyimagenet_dataset_reorg.py`. 