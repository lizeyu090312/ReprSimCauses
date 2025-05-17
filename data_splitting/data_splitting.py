import torch, torchvision
import numpy as np
from torch.utils.data import Dataset, Subset
from collections import defaultdict


def horizontal_split(dataset, fractional_overlap, mode="constrained"):
    """
    Splits dataset horizontally by randomly splitting the examples in each class
    so that the two subsets (s1 and s2) have identical size per class and the
    fraction of overlapping examples in each class is fractional_overlap.

    In "constrained" mode the maximum number of examples per class is capped at
    what you would get if there were zero overlap (i.e. half the available examples).
    In "unconstrained" mode we try to use as many examples as possible.
    """
    # Group indices by class.
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_to_indices[label].append(idx)
    
    s1_indices = []
    s2_indices = []
    
    for label, indices in class_to_indices.items():
        indices = np.array(indices)
        np.random.shuffle(indices)
        if mode == "constrained":
            N = len(indices) // 2  # no overlap => half the dataset is in each subset
        elif mode == "unconstrained":
            # Find the maximum N (#datapoints per subset) s.t.
            # 2*N - overlap_count <= len(indices), where overlap_count = round(fractional_overlap * N).
            best_N = 0
            for possible_N in range(1, len(indices) + 1):
                overlap_count = int(round(fractional_overlap * possible_N))
                if 2 * possible_N - overlap_count <= len(indices):
                    best_N = possible_N
            N = best_N
        else:
            raise ValueError(f"Unknown mode: {mode}")
        overlap_count = int(round(fractional_overlap * N))
        non_overlap_count = N - overlap_count
        # Ensure there are enough examples to sample the unique parts.
        if 2 * N - overlap_count > len(indices):
            raise ValueError(f"Not enough examples in class {label} for mode {mode} with fractional_overlap={fractional_overlap}")
        
        # random assignment
        #   First overlap_count will be the overlap
        #   Next non_overlap_count for s1 only
        #   Next non_overlap_count for s2 only
        np.random.shuffle(indices)
        overlap = indices[:overlap_count]
        s1_unique = indices[overlap_count:overlap_count + non_overlap_count]
        s2_unique = indices[overlap_count + non_overlap_count:overlap_count + 2 * non_overlap_count]
        
        s1_class_indices = np.concatenate([overlap, s1_unique])
        s2_class_indices = np.concatenate([overlap, s2_unique])
        assert len(s1_class_indices) == len(s2_class_indices)  # each subset has the same size
        assert len(set(s1_class_indices)) == len(s1_class_indices)  # check for duplicates
        assert len(set(s2_class_indices)) == len(s2_class_indices)
        
        s1_indices.extend(s1_class_indices.tolist())
        s2_indices.extend(s2_class_indices.tolist())
    
    return {"s1_indices": s1_indices, "s2_indices": s2_indices}

def vertical_split(dataset, fractional_overlap, mode="constrained"):
    """
    Splits dataset vertically by selecting classes for each subset.
    
    In "constrained" mode, each subset is capped to half of all available classes,
    and the number of classes that appear in both subsets is exactly fractional_overlap *
    (number of classes per subset).
    
    In "unconstrained" mode we try to use as many classes as possible while achieving an
    overlap as close as possible to the desired fraction.
    """
    # Group indices by class.
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_to_indices[label].append(idx)
    classes = sorted(class_to_indices.keys())
    num_classes = len(classes)
    
    # Decide how many classes per subset.
    if mode == "constrained":
        classes_per_subset = num_classes // 2
    elif mode == "unconstrained":
        # We choose the largest k such that 2*k - round(fractional_overlap * k) <= num_classes.
        best_k = 0
        for possible_k in range(2, num_classes + 1):
            common = int(round(fractional_overlap * possible_k))
            if 2 * possible_k - common <= num_classes:
                best_k = possible_k
        classes_per_subset = best_k
        if classes_per_subset <= 2:
            raise ValueError(f"classes_per_subset <= 2 (undesirable), classes_per_subset: {classes_per_subset}; choose another fractional_overlap")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    common_classes_count = int(round(fractional_overlap * classes_per_subset))
    all_classes = classes.copy()
    np.random.shuffle(all_classes)
    
    if 2 * classes_per_subset - common_classes_count > num_classes:
        raise ValueError("Not enough classes to form subsets with the desired overlap in mode {}".format(mode))
    
    if mode == "constrained":
        common = all_classes[:common_classes_count]
        remaining = all_classes[common_classes_count:]
        s1_only = remaining[:classes_per_subset - common_classes_count]
        s2_only = remaining[classes_per_subset - common_classes_count: 2 * (classes_per_subset - common_classes_count)]
    else:  # unconstrained
        # Pick the classes that will be used (total selected classes = 2*k - common)
        selected = all_classes[:2 * classes_per_subset - common_classes_count]
        np.random.shuffle(selected)
        common = selected[:common_classes_count]
        s1_only = selected[common_classes_count: common_classes_count + (classes_per_subset - common_classes_count)]
        s2_only = selected[common_classes_count + (classes_per_subset - common_classes_count):]
    
    assert len(s1_only) == len(s2_only)  # each subset has the same size
    s1_classes = set(common + s1_only)
    s2_classes = set(common + s2_only)
    # assert len(s1_classes) == len(s2_classes)  # unnecessary
    assert len(common + s1_only) == len(s1_classes)  # check for duplicates
    assert len(common + s2_only) == len(s2_classes)
    if mode == "constrained":
        assert np.isclose(len(s1_classes & s2_classes) / len(s2_classes), fractional_overlap), f"{len(s1_classes & s2_classes) / len(s2_classes)}, {fractional_overlap}"
    
    s1_indices = []
    s2_indices = []
    for label, indices in class_to_indices.items():
        if label in s1_classes:
            s1_indices.extend(indices)
        if label in s2_classes:
            s2_indices.extend(indices)
    
    return {"s1_indices": s1_indices, "s2_indices": s2_indices, "s1_classes": s1_classes, "s2_classes": s2_classes} 

def split_dataset(dataset, fractional_overlap, split_method, mode):
    """
    Splits a dataset into two identically sized subsets using either a horizontal
    (per-example) or vertical (per-class) split.
    
    Parameters:
      - dataset: the original PyTorch dataset.
      - fractional_overlap: desired overlap as a fraction (0 to 1).
      - split_method: "horizontal" or "vertical".
      - mode: "constrained" or "unconstrained".
    
    Returns:
      A tuple (s1, s2) of lists that can be used to index into datasets
    """
    if split_method == "horizontal":
        return horizontal_split(dataset, fractional_overlap, mode)
    elif split_method == "vertical":
        return vertical_split(dataset, fractional_overlap, mode)
    else:
        raise ValueError("Unknown split method: {}".format(split_method))

## Testing script
class DummyDataset(Dataset):
    """
    A dummy dataset for testing that simulates an image classification dataset.
    Each sample is a tuple (data, label). Here, data is just a tensor with one value.
    """
    def __init__(self, num_samples=100, num_classes=10):
        self.data = []
        self.targets = []
        samples_per_class = num_samples // num_classes
        for c in range(num_classes):
            for i in range(samples_per_class):
                #  tensor containing a unique value.
                self.data.append(torch.tensor([i + c * samples_per_class], dtype=torch.float))
                self.targets.append(c)
        # Handle remainder
        for i in range(num_samples - samples_per_class * num_classes):
            self.data.append(torch.tensor([i+num_samples], dtype=torch.float))
            # use i+num_samples since no data in the mian for loop has value greater than num_samples
            self.targets.append(i % num_classes)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def test_horizontal_split(mode):
    print(f"\n=== Testing Horizontal Split, mode={mode} ===")
    dataset = DummyDataset(num_samples=10000, num_classes=10)
    fractional_overlap = 0.2
    
    fos = [i*0.1 for i in range(10+1)]
    
    for fractional_overlap in fos:
        print(f"\n\nfractional_overlap {fractional_overlap}")
        ret_dict = split_dataset(dataset, fractional_overlap, split_method="horizontal", mode=mode)
        s1, s2 = Subset(dataset, ret_dict["s1_indices"]), Subset(dataset, ret_dict["s1_indices"])
        print("Total samples in original dataset:", len(dataset))
        print("Subset sizes: s1 =", len(s1), "s2 =", len(s2))
        
        # Group original dataset indices by class.
        class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            class_to_indices[label].append(idx)
        
        # For each class, compute the number of examples in each subset and the overlap.
        for label, indices in sorted(class_to_indices.items()):
            s1_set = set(s1.indices)
            s2_set = set(s2.indices)
            class_s1 = set(indices).intersection(s1_set)
            class_s2 = set(indices).intersection(s2_set)
            overlap = class_s1.intersection(class_s2)
            if len(class_s1) == 0:
                continue
            calculated_overlap = len(overlap) / len(class_s1)
            print(f"class {label:2d}: s1_count={len(class_s1):2d}, s2_count={len(class_s2):2d}, "
                f"overlap_count={len(overlap):2d}, overlap_fraction={calculated_overlap:.2f}")


def test_vertical_split(mode):
    print(f"\n=== Testing Vertical Split, mode={mode} ===")
    dataset = DummyDataset(num_samples=10000, num_classes=10)
    if mode == "constrained":
        fos = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    else:
        fos = [i*0.1 for i in range(10+1)]
    for fractional_overlap in fos:
        print(f"\n\nfractional_overlap {fractional_overlap}")
        ret_dict = split_dataset(dataset, fractional_overlap, split_method="vertical", mode=mode)
        s1, s2 = Subset(dataset, ret_dict["s1_indices"]), Subset(dataset, ret_dict["s1_indices"])
        print("Total samples in original dataset:", len(dataset))
        print("Subset sizes: s1 =", len(s1), "s2 =", len(s2))
        
        # Determine which classes are in each subset.
        s1_classes = set()
        for idx in s1.indices:
            _, label = dataset[idx]
            s1_classes.add(label)
        s2_classes = set()
        for idx in s2.indices:
            _, label = dataset[idx]
            s2_classes.add(label)
        
        common_classes = s1_classes.intersection(s2_classes)
        if len(s1_classes) > 0:
            actual_overlap_fraction = len(common_classes) / len(s1_classes)
        else:
            actual_overlap_fraction = 0.0
        
        print("Classes in s1:", sorted(s1_classes))
        print("Classes in s2:", sorted(s2_classes))
        print("Common classes:", sorted(common_classes))
        print("Desired overlap fraction:", fractional_overlap)
        print("Actual overlap fraction (by class):", actual_overlap_fraction)


def main():
    test_horizontal_split("constrained")
    test_vertical_split("constrained")
    
    test_horizontal_split("unconstrained")
    test_vertical_split("unconstrained")

if __name__ == "__main__":
    main()