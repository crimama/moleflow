"""
MoLE-Flow Data Module

Contains dataset classes and utilities for loading data.
"""

from moleflow.data.mvtec import MVTEC, MVTEC_CLASS_NAMES
from moleflow.data.visa import VISA, VISA_CLASS_NAMES
from moleflow.data.mpdd import MPDD, MPDD_CLASS_NAMES
from moleflow.data.datasets import (
    create_task_dataset,
    TaskDataset,
    get_dataset_class,
    get_class_names,
    DATASET_REGISTRY,
)

__all__ = [
    # MVTec
    "MVTEC",
    "MVTEC_CLASS_NAMES",
    # VisA
    "VISA",
    "VISA_CLASS_NAMES",
    # MPDD
    "MPDD",
    "MPDD_CLASS_NAMES",
    # Dataset utilities
    "create_task_dataset",
    "TaskDataset",
    "get_dataset_class",
    "get_class_names",
    "DATASET_REGISTRY",
]
