# Official Implementation of *Causes and Consequences of Representational Similarity in Machine Learning Models*
Under review.

### Introduction
This repository contains the official implementation of *Causes and Consequences of Representational Similarity in Machine Learning Models*. Please create two environments: one containing ffcv (refer to https://ffcv.io) and the other containing packages in `pip install -r requirements.txt`. 

### Training ResNet-18, nanoGPT, and diffusion UNet
The training scripts can be found under `train**/`. Please run `bash run_write_ffcv_data.sh` if this file is available and then `bash run_train_wrapper.sh` to train the models using task and dataset splitting. Remember to change the paths appropriately. For training the diffusion UNet, remember to change all occurrences of `/path/to/tiny_imagenet_200` and copy over `./log_dir/<dataset>/labels.pkl` files generated using `ReprSimCauses/vit_resnet_different_model_size/save_labels.py`. 

**Note for `train_nanoGPT/`**: please manually download the text files for `shakespeare_all` (see `git@github.com:cobanov/shakespeare-dataset.git`) and aggregate the texts into a single file called `ReprSimCauses/train_nanoGPT/data/shakespeare_all/input.txt`; then run `python prepare.py`. For the five other text datasets, simply run `cd ReprSimCauses/train_nanoGPT/data/<dataset>; python prepare.py` (data should be downloaded automatically from HuggingFace). 

### Tuning models
To finetune ResNet and ViT models, please refer to `ReprSimCauses/vit_resnet_different_model_size`. Before you start, ensure you download the [ColorShapeDigit800k](https://drive.google.com/file/d/1N1xWKNWm3rD-a7EOmMwQn86pyus8Xn0P/view?usp=share_link) dataset and the [TinyImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) dataset and set the correct paths to these two datasets. First, run `**write_data.py`. Then, run `save_labels.py` to generate labels of datasets. Then, edit the SLURM script `run_train_ds_overlap.sh` or `run_train_task_overlap.sh` to run the desired experiments. 

To finetune Llama models, please refer to `ReprSimCauses/llm_finetune`. Edit `run_train**.sh` to run the desired experiments. 

### Transfer adversarial attacks and transfer jailbreaking
Please refer to `ReprSimCauses/vit_resnet_different_model_size/transfer_att_<ds-or-task>_overlap.py` and `ReprSimCauses/llm_finetune/jailbreak.py` for code regarding transfer attacks. 

### Measuring representational similarity
Please run the bash files in `repr_sim/` to compute the representational similarity metrics. We base our code for computing representational similarity on this repository: https://github.com/minyoungg/platonic-rep. 

### Reproducing figures
To reproduce the box plots in the main paper as well as the mutual information table, please refer to `plotting/final_plotting.ipynb`. The raw data we used is stored in `plotting/plotting_results`. To plot the representational similarity values that you produce, please move the appropriate file to `plotting/plotting_results`. Take the example of `repr_sim/cifar10_ds_overlap_run_metrics_ffcv.py`: the representational similarity values output by `cifar10_ds_overlap_run_metrics_ffcv.py` are stored in `train_cifar_ffcv/log_dir/ds_overlap_cifar10/platonic_ffcv.csv`. After copying the `csv` file to `plotting/plotting_results` and renaming the file appropriately, you can plot the results using the first code cell in `plotting/final_plotting.ipynb`. 

### Notes about the TinyImageNet dataset
Please download the dataset from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`. Then, unzip using `unzip path/to/tiny-imagenet-200.zip`. Then, copy `ReprSimCauses/train_tinyimagenet_ffcv/tinyimagenet_dataset_reorg.py` to `path/to/tiny-imagenet-200` and run `cd path/to/tiny-imagenet-200; python tinyimagenet_dataset_reorg.py`. 