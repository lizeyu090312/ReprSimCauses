python ffcv_ood_write_data.py \
    --data.train_dataset "./ood_dataset/cifar10/train.beton" \
    --data.val_dataset "./ood_dataset/cifar10/val.beton"

python ffcv_ood_write_data.py \
    --data.train_dataset "./ood_dataset/mnist/train.beton" \
    --data.val_dataset "./ood_dataset/mnist/val.beton"

python ffcv_ood_write_data.py \
    --data.train_dataset "./ood_dataset/svhn/train.beton" \
    --data.val_dataset "./ood_dataset/svhn/val.beton"

python ffcv_partition_write_data.py  # this takes many disk IO ops. 