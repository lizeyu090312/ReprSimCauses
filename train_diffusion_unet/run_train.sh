#!/bin/bash
#SBATCH --job-name=ffcv        # Job name
#SBATCH --output=/path/to/ReprSimCauses/train_diffusion_unet/log_dir/slurm_out/%j.out            # Output file name (%j expands to jobID)
#SBATCH --error=/path/to/ReprSimCauses/train_diffusion_unet/log_dir/slurm_out/%j.err             # Error file name (%j expands to jobID)
#SBATCH --partition=your_account-gpu           # Partition name
#SBATCH --time=9-23:59:59            # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=36G                   # Memory required per node
#SBATCH --gres=gpu:1
#SBATCH --account=your_account


# Load any required modules here

# info: run like SEED=int sbatch run_train.sh
mkdir -p /path/to/ReprSimCauses/train_diffusion_unet/log_dir/slurm_out

source ~/.bashrc  # Ensure bash profile is loaded (only needed if using bash)
eval "$(conda shell.bash hook)"  # Initialize Conda
conda activate ffcv2; cd /path/to/ReprSimCauses/train_diffusion_unet

export LOG_DIR="./log_dir/task_overlap_train"

# # tinyimagenet
# export DS_NAME="tinyimagenet"
# export GRAD_ACCUM=2
# export BSZ=256
# export IMG_SIZE=64
# export STEPS=20
# export LOG_EVERY=5

# CIFAR10
export DS_NAME="cifar10"
export GRAD_ACCUM=1
export BSZ=1280
export IMG_SIZE=32
export STEPS=30000
export LOG_EVERY=300

# export N_IMGS=10000
# # dataset overlap
# python ds_overlap_train.py --training.batch_size $BSZ --training.grad_accum_iters $GRAD_ACCUM --data.class_frac_overlap 0 --output.n_fid $N_IMGS --training.steps $STEPS --output.log_every $LOG_EVERY --data.img_size $IMG_SIZE --data.dataset_name $DS_NAME --training.seed $SEED --output.output_root $LOG_DIR
# python -m pytorch_fid ./ds_overlap_fid/${DS_NAME}_saved_png.npz ${LOG_DIR}/${DS_NAME}/seed_${SEED}/frac_overlap_0/split_0/generated --device cuda:0 >> "${LOG_DIR}/${DS_NAME}/seed_${SEED}/frac_overlap_0/split_0/log.txt"
# export N_IMGS=0
# python ds_overlap_train.py --training.batch_size $BSZ --training.grad_accum_iters $GRAD_ACCUM --data.class_frac_overlap 0.25 --output.n_fid $N_IMGS --training.steps $STEPS --output.log_every $LOG_EVERY --data.img_size $IMG_SIZE --data.dataset_name $DS_NAME --training.seed $SEED --output.output_root $LOG_DIR
# python ds_overlap_train.py --training.batch_size $BSZ --training.grad_accum_iters $GRAD_ACCUM --data.class_frac_overlap 0.5 --output.n_fid $N_IMGS --training.steps $STEPS --output.log_every $LOG_EVERY --data.img_size $IMG_SIZE --data.dataset_name $DS_NAME --training.seed $SEED --output.output_root $LOG_DIR
# python ds_overlap_train.py --training.batch_size $BSZ --training.grad_accum_iters $GRAD_ACCUM --data.class_frac_overlap 0.75 --output.n_fid $N_IMGS --training.steps $STEPS --output.log_every $LOG_EVERY --data.img_size $IMG_SIZE --data.dataset_name $DS_NAME --training.seed $SEED --output.output_root $LOG_DIR
# python ds_overlap_train.py --training.batch_size $BSZ --training.grad_accum_iters $GRAD_ACCUM --data.class_frac_overlap 1.0 --output.n_fid $N_IMGS --training.steps $STEPS --output.log_every $LOG_EVERY --data.img_size $IMG_SIZE --data.dataset_name $DS_NAME --training.seed $SEED --output.output_root $LOG_DIR

export N_IMGS=10000
# task overlap
python task_overlap_train.py --training.batch_size $BSZ --training.grad_accum_iters $GRAD_ACCUM --data.class_frac_overlap 0 --output.n_fid $N_IMGS --training.steps $STEPS --output.log_every $LOG_EVERY --data.img_size $IMG_SIZE --data.dataset_name $DS_NAME --training.seed $SEED --output.output_root $LOG_DIR
python -m pytorch_fid ./ds_overlap_fid/${DS_NAME}_saved_png.npz ${LOG_DIR}/${DS_NAME}/seed_${SEED}/frac_overlap_0/split_0/generated --device cuda:0 >> "${LOG_DIR}/${DS_NAME}/seed_${SEED}/frac_overlap_0/split_0/log.txt"
export N_IMGS=0
python task_overlap_train.py --training.batch_size $BSZ --training.grad_accum_iters $GRAD_ACCUM --data.class_frac_overlap 0.2 --output.n_fid $N_IMGS --training.steps $STEPS --output.log_every $LOG_EVERY --data.img_size $IMG_SIZE --data.dataset_name $DS_NAME --training.seed $SEED --output.output_root $LOG_DIR
python task_overlap_train.py --training.batch_size $BSZ --training.grad_accum_iters $GRAD_ACCUM --data.class_frac_overlap 0.4 --output.n_fid $N_IMGS --training.steps $STEPS --output.log_every $LOG_EVERY --data.img_size $IMG_SIZE --data.dataset_name $DS_NAME --training.seed $SEED --output.output_root $LOG_DIR
python task_overlap_train.py --training.batch_size $BSZ --training.grad_accum_iters $GRAD_ACCUM --data.class_frac_overlap 0.6 --output.n_fid $N_IMGS --training.steps $STEPS --output.log_every $LOG_EVERY --data.img_size $IMG_SIZE --data.dataset_name $DS_NAME --training.seed $SEED --output.output_root $LOG_DIR
python task_overlap_train.py --training.batch_size $BSZ --training.grad_accum_iters $GRAD_ACCUM --data.class_frac_overlap 0.8 --output.n_fid $N_IMGS --training.steps $STEPS --output.log_every $LOG_EVERY --data.img_size $IMG_SIZE --data.dataset_name $DS_NAME --training.seed $SEED --output.output_root $LOG_DIR
python task_overlap_train.py --training.batch_size $BSZ --training.grad_accum_iters $GRAD_ACCUM --data.class_frac_overlap 1.0 --output.n_fid $N_IMGS --training.steps $STEPS --output.log_every $LOG_EVERY --data.img_size $IMG_SIZE --data.dataset_name $DS_NAME --training.seed $SEED --output.output_root $LOG_DIR



# ## testing script

# export SEED=2
# export LOG_DIR="./log_dir/ds_overlap_train"

# # # CIFAR10
# export DS_NAME="cifar10"
# export GRAD_ACCUM=1
# export BSZ=1280
# export IMG_SIZE=32
# export STEPS=30000
# export LOG_EVERY=300

# export N_IMGS=0
# # dataset overlap
# python ds_overlap_train.py --training.batch_size $BSZ --training.grad_accum_iters $GRAD_ACCUM --data.class_frac_overlap 0.5 --output.n_fid $N_IMGS --training.steps $STEPS --output.log_every $LOG_EVERY --data.img_size $IMG_SIZE --data.dataset_name $DS_NAME --training.seed $SEED --output.output_root $LOG_DIR
# # python -m pytorch_fid ./ds_overlap_fid/${DS_NAME}_saved_png.npz ${LOG_DIR}/${DS_NAME}/seed_${SEED}/frac_overlap_0/split_0/generated --device cuda:0 >> "${LOG_DIR}/${DS_NAME}/seed_${SEED}/frac_overlap_0/split_0/log.txt"

# ## testing script

# export SEED=0
# export LOG_DIR="./log_dir/task_overlap_train_test"

# # # CIFAR10
# export DS_NAME="cifar10"
# export GRAD_ACCUM=1
# export BSZ=128
# export IMG_SIZE=32
# export STEPS=2
# export LOG_EVERY=1

# export N_IMGS=256
# # dataset overlap
# python task_overlap_train.py --training.num_workers 4 --training.batch_size $BSZ --training.grad_accum_iters $GRAD_ACCUM --data.class_frac_overlap 0 --output.n_fid $N_IMGS --training.steps $STEPS --output.log_every $LOG_EVERY --data.img_size $IMG_SIZE --data.dataset_name $DS_NAME --training.seed $SEED --output.output_root $LOG_DIR
# python -m pytorch_fid ./ds_overlap_fid/${DS_NAME}_saved_png.npz ${LOG_DIR}/${DS_NAME}/seed_${SEED}/frac_overlap_0/split_0/generated --device cuda:0 >> "${LOG_DIR}/${DS_NAME}/seed_${SEED}/frac_overlap_0/split_0/log.txt"