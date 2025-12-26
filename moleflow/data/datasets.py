"""
Dataset utilities for MoLE-Flow continual learning.
"""

from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset

from moleflow.data.mvtec import MVTEC


class TaskDataset(Dataset):
    """Dataset wrapper for continual learning tasks."""

    def __init__(self, data: List[Tuple], class_names: List[str], class_to_idx: Dict[str, int]):
        """
        Args:
            data: List of (image, class_id, mask, name, path) tuples
            class_names: List of class names in this task
            class_to_idx: Mapping from class name to global class index
        """
        self.data = data
        self.class_names = class_names
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        return self.data[idx]


def create_task_dataset(
    args,
    task_classes: List[str],
    global_class_to_idx: Dict[str, int],
    train: bool = True
) -> TaskDataset:
    """Create dataset for specific task classes.

    Args:
        args: Configuration object with data_path, img_size, msk_size attributes
        task_classes: List of class names for this task
        global_class_to_idx: Global mapping from class name to index
        train: If True, load training data, otherwise test data

    Returns:
        TaskDataset containing data for the specified classes
    """
    filtered_data = []

    for class_name in task_classes:
        # Create dataset for specific class
        class_dataset = MVTEC(
            root=args.data_path,
            class_name=class_name,
            train=train,
            img_size=args.img_size,
            crp_size=args.img_size,
            msk_size=args.msk_size
        )

        # Add all data from this class
        for i in range(len(class_dataset)):
            image, label, mask, name, path = class_dataset[i]
            # Use global class ID
            final_class_id = int(global_class_to_idx[class_name])
            filtered_data.append((image, final_class_id, mask, name, path))

    # Create task-specific class mapping
    task_class_to_idx = {cls: global_class_to_idx[cls] for cls in task_classes}

    return TaskDataset(filtered_data, task_classes, task_class_to_idx)
