import ffcv
import time, os

import torch
from torchvision import datasets
from torchvision.datasets import ImageFolder

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from PIL import Image


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class CollapsedImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, collapse_by, transform=None):
        """
        root: path to dataset structured as ImageFolder
        collapse_by: tuple of labels to collapse by, e.g. ('shape', 'digit')
        transform: optional torchvision transform
        """
        self.root = root
        self.transform = transform
        self.collapse_by = collapse_by
        self.original_dataset = datasets.ImageFolder(root=root)

        # Map from original class name to (shape, digit, color)
        self.orig_classname_map = {}
        self.classname_to_label = {}
        self.samples = []

        # Parse and generate collapsed labels
        self._prepare()

    def _prepare(self):
        label_set = set()

        for img_path, _ in self.original_dataset.samples:
            class_name = os.path.basename(os.path.dirname(img_path))  # e.g., "circle_7_blue"

            try:
                shape, digit, color = class_name.split("_")
            except ValueError:
                print(f"Skipping malformed class name: {class_name}")
                continue

            label_parts = {'shape': shape, 'digit': digit, 'color': color}
            collapsed_key = "_".join([label_parts[k] for k in self.collapse_by])
            label_set.add(collapsed_key)
            self.samples.append((img_path, collapsed_key))

        # Create label-to-index mapping
        sorted_labels = sorted(list(label_set))
        self.classname_to_label = {name: idx for idx, name in enumerate(sorted_labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, collapsed_classname = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.classname_to_label[collapsed_classname]
        return img, label

    
def main():
    partition_base = "./all_partitions"
    # move ColorShapeDigit800k.zip under partition_base and run unzip ColorShapeDigit800k.zip
    ffcv_partition_base = "./ffcv_all_partitions"
    for _t_ in ["train", "val"]:
        partition3_path = f"{partition_base}/partition3/{_t_}"
        path_to_beton = f"{ffcv_partition_base}/partition3/all/{_t_}.beton"
        if os.path.exists(path_to_beton) == False:
            print(f"writing to {path_to_beton}")
            ds = ImageFolder(root=partition3_path)
            os.makedirs(f"{ffcv_partition_base}/partition3/all", exist_ok=True)
            writer = DatasetWriter(path_to_beton, {
                'image': RGBImageField(),
                'label': IntField()
            })
            writer.from_indexed_dataset(ds)
        
        collapse_strategies = ["shape_digit", "shape_color", "digit_color"]
        for c in collapse_strategies:
            path_to_beton = f"{ffcv_partition_base}/partition2/{c}/{_t_}.beton"
            if os.path.exists(path_to_beton) == False:
                print(f"writing to {path_to_beton}")
                ds = CollapsedImageFolder(partition3_path, collapse_by=c.split('_'))
                os.makedirs(f"{ffcv_partition_base}/partition2/{c}", exist_ok=True)
                writer = DatasetWriter(path_to_beton, {
                    'image': RGBImageField(),
                    'label': IntField()
                })
                writer.from_indexed_dataset(ds)
        
        for name in ["shape", "digit"]:
            path_to_beton = f"{ffcv_partition_base}/partition1/{name}/{_t_}.beton"
            if os.path.exists(path_to_beton) == False:
                print(f"writing to {path_to_beton}")
                ds = CollapsedImageFolder(partition3_path, collapse_by=[name])
                os.makedirs(f"{ffcv_partition_base}/partition1/{name}", exist_ok=True)
                writer = DatasetWriter(path_to_beton, {
                    'image': RGBImageField(),
                    'label': IntField()
                })
                writer.from_indexed_dataset(ds)
       
if __name__ == "__main__":
    main()