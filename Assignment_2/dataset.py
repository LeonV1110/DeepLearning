import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Class name to index mapping
CLASS_NAMES = ["rest", "task_motor", "task_story_math", "task_working_memory"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class MEGDataset(Dataset):
    def __init__(self, file_paths, labels, person_ids=None, cache_data=True):
        self.file_paths = file_paths
        self.labels = labels
        self.person_ids = person_ids
        self.cache_data = cache_data
        self._cache = {}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        if self.cache_data and idx in self._cache:
            x = self._cache[idx]
        else:
            x = pd.read_csv(
                self.file_paths[idx],
                index_col=0,
            ).values.astype(np.float32)

            # CSV shape: (time, sensors)
            x = x.T  # Convert to: (sensors, time)
            x = torch.from_numpy(x)

            if self.cache_data:
                self._cache[idx] = x

        y = torch.tensor(self.labels[idx]).long()

        if self.person_ids is not None:
            person_id = torch.tensor(self.person_ids[idx]).long()
            return x, y, person_id

        return x, y


def load_file_paths(data_dir):

    train_files = []
    test_files = []
    train_labels = []
    test_labels = []

    # Load data from preprocessed_data directory
    for split_type in os.listdir(data_dir):
        split_path = os.path.join(data_dir, split_type)
        if not os.path.isdir(split_path):
            continue

        print(f"Loading {split_type} data...")

        for file_name in sorted(os.listdir(split_path)):
            if not file_name.endswith(".csv"):
                continue

            file_path = os.path.join(split_path, file_name)

            # Extract class name from filename
            # Format: class_name_..._subject_id.csv
            parts = file_name.split("_")
            # Remove the last part (subject_id) and extension
            class_name = "_".join(parts[:-2]).replace(".csv", "")

            if class_name not in CLASS_TO_IDX:
                print(f"Warning: Unknown class '{class_name}'")
                continue

            class_idx = CLASS_TO_IDX[class_name]

            if split_type == "train":
                train_files.append(file_path)
                train_labels.append(class_idx)
            else:
                test_files.append(file_path)
                test_labels.append(class_idx)

    print(f"Loaded {len(train_files)} training samples")
    print(f"Loaded {len(test_files)} test samples")
    print(f"Class distribution in training: {np.bincount(train_labels)}")
    print(f"Class distribution in test: {np.bincount(test_labels)}")

    return (train_files, train_labels, test_files, test_labels)


def create_dataloaders(
    data_dir,
    batch_size,
    add_person_id=False,
    cache_data=True,
    num_workers=0,
    pin_memory=None,
):

    train_files, train_labels, test_files, test_labels = load_file_paths(data_dir)

    if add_person_id:
        train_person_ids = [
            int(os.path.splitext(f)[0].split("_")[-2]) for f in train_files
        ]
        test_person_ids = [
            int(os.path.splitext(f)[0].split("_")[-2]) for f in test_files
        ]
    else:
        train_person_ids = None
        test_person_ids = None

    train_dataset = MEGDataset(
        train_files,
        train_labels,
        train_person_ids,
        cache_data=cache_data,
    )
    test_dataset = MEGDataset(
        test_files,
        test_labels,
        test_person_ids,
        cache_data=cache_data,
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, test_loader
