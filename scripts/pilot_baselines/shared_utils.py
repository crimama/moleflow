"""
Shared utilities for pilot baseline experiments.

This module provides common functionality for all baseline models:
- Feature extraction (reusing MoLE-Flow's backbone)
- Dataset/DataLoader creation
- Evaluation metrics (Image AUC, Pixel AP, Forgetting Measure)
- Results logging and saving utilities
- Common training loop utilities

Author: MoLE-Flow Team
"""

import os
import sys
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from moleflow.extractors import create_feature_extractor, get_backbone_type
from moleflow.data.mvtec import MVTEC


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a pilot experiment."""

    # Task settings
    task_classes: List[str] = field(default_factory=lambda: ['leather', 'grid', 'transistor'])

    # Training settings
    num_epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4

    # Backbone settings
    backbone: str = 'wide_resnet50_2'
    target_embed_dim: int = 1024

    # Data settings
    data_path: str = '/Data/MVTecAD'
    img_size: int = 256
    msk_size: int = 256

    # Logging settings
    log_base_dir: str = './logs/PilotExperiment'
    experiment_name: str = ''

    # Reproducibility
    seeds: List[int] = field(default_factory=lambda: [42])

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TaskMetrics:
    """Metrics for a single task evaluation."""

    task_id: int
    task_class: str
    image_auc: float
    pixel_auc: float = 0.0
    pixel_ap: float = 0.0
    num_samples: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingState:
    """Tracks training state for checkpointing and analysis."""

    current_task: int = 0
    current_epoch: int = 0
    loss_history: List[float] = field(default_factory=list)
    task_losses: Dict[int, List[float]] = field(default_factory=dict)

    # Forgetting tracking: task_id -> list of AUCs after each subsequent task
    forgetting_history: Dict[int, List[float]] = field(default_factory=dict)


# =============================================================================
# Feature Extractor Wrapper
# =============================================================================

class FeatureExtractorWrapper(nn.Module):
    """
    Wrapper around MoLE-Flow's feature extractor for baseline models.

    This provides a unified interface for feature extraction that:
    1. Keeps the backbone frozen (shared across all tasks)
    2. Provides both patch-level and image-level features
    3. Handles different backbone types (ViT/CNN) transparently
    """

    def __init__(
        self,
        backbone_name: str = 'wide_resnet50_2',
        input_shape: Tuple[int, int, int] = (3, 256, 256),
        target_embed_dim: int = 1024,
        device: str = 'cuda'
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.device = device
        self.backbone_type = get_backbone_type(backbone_name)

        # Create feature extractor using MoLE-Flow's factory
        self.extractor = create_feature_extractor(
            backbone_name=backbone_name,
            input_shape=input_shape,
            target_embed_dimension=target_embed_dim,
            device=device
        )

        # Freeze the backbone
        for param in self.extractor.parameters():
            param.requires_grad = False

        self.extractor.eval()
        self.target_embed_dim = target_embed_dim

        # Cache spatial shape
        self._spatial_shape = None

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        return_spatial_shape: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[int, int]]]:
        """
        Extract patch-level features from images.

        Args:
            images: (B, C, H, W) input images
            return_spatial_shape: If True, also return (H_patch, W_patch)

        Returns:
            features: (B, H_patch, W_patch, D) patch embeddings
            spatial_shape (optional): (H_patch, W_patch) tuple
        """
        images = images.to(self.device)
        return self.extractor(images, return_spatial_shape=return_spatial_shape)

    @torch.no_grad()
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image-level features via global average pooling.

        Args:
            images: (B, C, H, W) input images

        Returns:
            features: (B, D) image-level embeddings
        """
        images = images.to(self.device)
        return self.extractor.get_image_level_features(images)

    @torch.no_grad()
    def get_flatten_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract flattened patch features.

        Args:
            images: (B, C, H, W) input images

        Returns:
            features: (B*H*W, D) flattened patch embeddings
        """
        images = images.to(self.device)
        return self.extractor.get_flatten_embeddings(images)

    def get_spatial_shape(self, images: torch.Tensor) -> Tuple[int, int]:
        """Get the spatial shape of feature maps for given input."""
        if self._spatial_shape is None:
            _, self._spatial_shape = self.forward(images[:1], return_spatial_shape=True)
        return self._spatial_shape


# =============================================================================
# Dataset Utilities
# =============================================================================

def create_dataloader(
    config: ExperimentConfig,
    class_name: str,
    train: bool = True,
    shuffle: bool = None
) -> DataLoader:
    """
    Create a DataLoader for a specific class.

    Args:
        config: Experiment configuration
        class_name: MVTec class name (e.g., 'leather')
        train: If True, load training data
        shuffle: Override default shuffle behavior

    Returns:
        DataLoader for the specified class
    """
    dataset = MVTEC(
        root=config.data_path,
        class_name=class_name,
        train=train,
        img_size=config.img_size,
        crp_size=config.img_size,
        msk_size=config.msk_size
    )

    if shuffle is None:
        shuffle = train

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=train and len(dataset) >= config.batch_size
    )


def create_task_dataloaders(
    config: ExperimentConfig,
    task_id: int
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders for a task.

    Args:
        config: Experiment configuration
        task_id: Task index

    Returns:
        (train_loader, test_loader) tuple
    """
    class_name = config.task_classes[task_id]
    train_loader = create_dataloader(config, class_name, train=True)
    test_loader = create_dataloader(config, class_name, train=False)
    return train_loader, test_loader


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_image_auc(
    labels: np.ndarray,
    scores: np.ndarray
) -> float:
    """
    Compute Image-level AUC.

    Args:
        labels: Binary labels (0=normal, 1=anomaly)
        scores: Anomaly scores (higher = more anomalous)

    Returns:
        AUC score
    """
    if len(np.unique(labels)) < 2:
        return 0.5  # Can't compute AUC with single class
    return roc_auc_score(labels, scores)


def compute_pixel_metrics(
    masks: np.ndarray,
    anomaly_maps: np.ndarray
) -> Tuple[float, float]:
    """
    Compute Pixel-level AUC and AP.

    Args:
        masks: Ground truth masks (B, H, W), binary
        anomaly_maps: Predicted anomaly maps (B, H, W)

    Returns:
        (pixel_auc, pixel_ap) tuple
    """
    # Flatten for metrics computation
    masks_flat = masks.reshape(-1)
    maps_flat = anomaly_maps.reshape(-1)

    # Skip if no anomalous pixels
    if masks_flat.sum() == 0:
        return 0.5, 0.0

    pixel_auc = roc_auc_score(masks_flat, maps_flat)
    pixel_ap = average_precision_score(masks_flat, maps_flat)

    return pixel_auc, pixel_ap


def compute_forgetting_measure(
    initial_auc: float,
    final_auc: float
) -> float:
    """
    Compute Forgetting Measure (FM).

    FM = AUC_initial - AUC_after_learning_new_tasks

    Positive FM means forgetting occurred.

    Args:
        initial_auc: AUC immediately after learning the task
        final_auc: AUC after learning subsequent tasks

    Returns:
        Forgetting measure (positive = forgetting)
    """
    return initial_auc - final_auc


def compute_backward_transfer(
    auc_history: List[float]
) -> float:
    """
    Compute Backward Transfer (BWT).

    BWT measures how much learning new tasks affects old tasks.
    BWT = (1/(T-1)) * sum(AUC_final - AUC_initial) for tasks 0..T-2

    Args:
        auc_history: List of AUCs for a task after each subsequent task

    Returns:
        Backward transfer (negative = forgetting)
    """
    if len(auc_history) < 2:
        return 0.0

    # BWT is the difference between final and initial performance
    return auc_history[-1] - auc_history[0]


# =============================================================================
# Evaluation Runner
# =============================================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    feature_extractor: FeatureExtractorWrapper,
    test_loader: DataLoader,
    device: str = 'cuda',
    compute_pixel: bool = True,
    score_fn: Callable = None
) -> TaskMetrics:
    """
    Evaluate a model on a test set.

    Args:
        model: The anomaly detection model
        feature_extractor: Feature extractor wrapper
        test_loader: Test DataLoader
        device: Device to use
        compute_pixel: Whether to compute pixel-level metrics
        score_fn: Optional custom scoring function(model, features) -> (image_scores, anomaly_maps)

    Returns:
        TaskMetrics with evaluation results
    """
    model.eval()

    all_labels = []
    all_scores = []
    all_masks = []
    all_anomaly_maps = []

    for batch in test_loader:
        images, labels, masks, _, _ = batch
        images = images.to(device)

        # Get features
        features, spatial_shape = feature_extractor(images, return_spatial_shape=True)

        # Get anomaly scores
        if score_fn is not None:
            image_scores, anomaly_maps = score_fn(model, features)
        else:
            # Default: expect model to have compute_anomaly_score method
            image_scores, anomaly_maps = model.compute_anomaly_score(features)

        all_labels.append(labels.cpu().numpy())
        all_scores.append(image_scores.cpu().numpy())

        if compute_pixel:
            all_masks.append(masks.cpu().numpy())
            # Resize anomaly maps to mask size if needed
            if anomaly_maps.shape[-2:] != masks.shape[-2:]:
                anomaly_maps = F.interpolate(
                    anomaly_maps.unsqueeze(1),
                    size=masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
            all_anomaly_maps.append(anomaly_maps.cpu().numpy())

    # Concatenate
    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)

    # Compute image AUC
    image_auc = compute_image_auc(labels, scores)

    # Compute pixel metrics
    pixel_auc, pixel_ap = 0.0, 0.0
    if compute_pixel and len(all_masks) > 0:
        masks = np.concatenate(all_masks)
        anomaly_maps = np.concatenate(all_anomaly_maps)
        pixel_auc, pixel_ap = compute_pixel_metrics(masks, anomaly_maps)

    return TaskMetrics(
        task_id=-1,  # To be filled by caller
        task_class=test_loader.dataset.class_name,
        image_auc=image_auc,
        pixel_auc=pixel_auc,
        pixel_ap=pixel_ap,
        num_samples=len(labels)
    )


def evaluate_all_tasks(
    model: nn.Module,
    feature_extractor: FeatureExtractorWrapper,
    config: ExperimentConfig,
    current_task: int,
    score_fn: Callable = None
) -> Dict[int, TaskMetrics]:
    """
    Evaluate model on all tasks up to and including current_task.

    Args:
        model: The anomaly detection model
        feature_extractor: Feature extractor wrapper
        config: Experiment configuration
        current_task: Current task index (evaluate tasks 0..current_task)
        score_fn: Optional custom scoring function

    Returns:
        Dict mapping task_id to TaskMetrics
    """
    results = {}

    for task_id in range(current_task + 1):
        _, test_loader = create_task_dataloaders(config, task_id)

        metrics = evaluate_model(
            model=model,
            feature_extractor=feature_extractor,
            test_loader=test_loader,
            device=config.device,
            score_fn=score_fn
        )
        metrics.task_id = task_id
        results[task_id] = metrics

    return results


# =============================================================================
# Logging and Results Management
# =============================================================================

class ExperimentLogger:
    """
    Logger for experiment results and training progress.

    Handles:
    - Training loss logging to CSV
    - Metric logging
    - Forgetting analysis
    - Summary generation
    """

    def __init__(
        self,
        log_dir: str,
        model_name: str,
        config: ExperimentConfig
    ):
        self.log_dir = Path(log_dir)
        self.model_name = model_name
        self.config = config

        # Create directories
        self.model_dir = self.log_dir / f"{model_name}_baseline"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.model_dir / 'config.json', 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        # Initialize training log
        self.training_log_path = self.model_dir / 'training_log.csv'
        with open(self.training_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['task_id', 'epoch', 'loss', 'timestamp'])

        # Results storage
        self.results: Dict[str, any] = {
            'model_name': model_name,
            'config': config.to_dict(),
            'task_metrics': {},
            'forgetting': {},
            'training_time': {}
        }

    def log_training_step(
        self,
        task_id: int,
        epoch: int,
        loss: float
    ):
        """Log a training step."""
        with open(self.training_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([task_id, epoch, loss, datetime.now().isoformat()])

    def log_task_metrics(
        self,
        task_id: int,
        metrics: Dict[int, TaskMetrics],
        after_task: int
    ):
        """
        Log metrics for all tasks after training on a specific task.

        Args:
            task_id: Which task was just trained
            metrics: Dict of task_id -> TaskMetrics for all evaluated tasks
            after_task: Task ID that was just completed
        """
        key = f"after_task_{after_task}"
        if key not in self.results['task_metrics']:
            self.results['task_metrics'][key] = {}

        for tid, m in metrics.items():
            self.results['task_metrics'][key][tid] = m.to_dict()

    def log_forgetting(
        self,
        task_id: int,
        initial_auc: float,
        current_auc: float,
        after_task: int
    ):
        """Log forgetting for a task."""
        if task_id not in self.results['forgetting']:
            self.results['forgetting'][task_id] = {
                'initial_auc': initial_auc,
                'auc_history': [initial_auc]
            }

        self.results['forgetting'][task_id]['auc_history'].append(current_auc)
        self.results['forgetting'][task_id]['final_auc'] = current_auc
        self.results['forgetting'][task_id]['forgetting_measure'] = compute_forgetting_measure(
            initial_auc, current_auc
        )

    def log_training_time(self, task_id: int, time_seconds: float):
        """Log training time for a task."""
        self.results['training_time'][task_id] = time_seconds

    def save_results(self):
        """Save all results to JSON."""
        with open(self.model_dir / 'results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

    def print_summary(self):
        """Print a summary of results."""
        print(f"\n{'='*60}")
        print(f"Results for {self.model_name}")
        print(f"{'='*60}")

        # Final metrics
        final_key = max(self.results['task_metrics'].keys())
        final_metrics = self.results['task_metrics'][final_key]

        print("\nFinal Task Metrics:")
        print(f"{'Task':<15} {'I-AUC':<10} {'P-AUC':<10} {'P-AP':<10}")
        print("-" * 45)

        total_iauc = 0
        for tid, m in sorted(final_metrics.items(), key=lambda x: int(x[0])):
            print(f"{m['task_class']:<15} {m['image_auc']:.4f}     {m['pixel_auc']:.4f}     {m['pixel_ap']:.4f}")
            total_iauc += m['image_auc']

        avg_iauc = total_iauc / len(final_metrics)
        print("-" * 45)
        print(f"{'Average':<15} {avg_iauc:.4f}")

        # Forgetting analysis
        if self.results['forgetting']:
            print("\nForgetting Analysis:")
            print(f"{'Task':<15} {'Initial':<10} {'Final':<10} {'FM':<10}")
            print("-" * 45)

            total_fm = 0
            for tid, fdata in sorted(self.results['forgetting'].items()):
                fm = fdata.get('forgetting_measure', 0)
                print(f"Task {tid:<10} {fdata['initial_auc']:.4f}     {fdata.get('final_auc', fdata['initial_auc']):.4f}     {fm:+.4f}")
                total_fm += fm

            avg_fm = total_fm / len(self.results['forgetting']) if self.results['forgetting'] else 0
            print("-" * 45)
            print(f"{'Avg FM':<15} {'':<10} {'':<10} {avg_fm:+.4f}")


# =============================================================================
# Training Utilities
# =============================================================================

class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, loss: float) -> bool:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# Abstract Base Class for Baseline Models
# =============================================================================

class BaselineModel(nn.Module):
    """
    Abstract base class for baseline anomaly detection models.

    All baseline models should implement:
    - add_task(task_id): Prepare model for a new task
    - train_task(task_id, train_loader, ...): Train on a task
    - compute_anomaly_score(features): Compute anomaly scores
    - get_trainable_params(): Get parameters to train
    """

    def __init__(self, feature_dim: int, device: str = 'cuda'):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device
        self.current_task = -1
        self.task_components: Dict[int, nn.Module] = {}

    def add_task(self, task_id: int):
        """Prepare model for a new task. Override in subclass."""
        raise NotImplementedError

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        feature_extractor: FeatureExtractorWrapper,
        num_epochs: int,
        lr: float,
        logger: ExperimentLogger = None
    ) -> List[float]:
        """
        Train on a task. Override in subclass.

        Returns:
            List of epoch losses
        """
        raise NotImplementedError

    def compute_anomaly_score(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute anomaly scores from features.

        Args:
            features: (B, H, W, D) patch features

        Returns:
            image_scores: (B,) image-level scores
            anomaly_maps: (B, H, W) spatial anomaly maps
        """
        raise NotImplementedError

    def get_trainable_params(self, task_id: int = None):
        """Get trainable parameters. Override in subclass."""
        raise NotImplementedError

    def set_active_task(self, task_id: int):
        """Set the active task for inference."""
        self.current_task = task_id


# =============================================================================
# Summary Report Generation
# =============================================================================

def generate_comparison_table(
    log_dir: str,
    model_names: List[str]
) -> str:
    """
    Generate a comparison table from experiment results.

    Args:
        log_dir: Base log directory
        model_names: List of model names to compare

    Returns:
        Markdown table string
    """
    log_path = Path(log_dir)

    # Collect results
    all_results = {}
    for name in model_names:
        result_path = log_path / f"{name}_baseline" / 'results.json'
        if result_path.exists():
            with open(result_path) as f:
                all_results[name] = json.load(f)

    if not all_results:
        return "No results found."

    # Build table
    lines = ["# Pilot Experiment Comparison\n"]
    lines.append("| Model | Avg I-AUC | Avg FM | Task 0 Final | Task 1 Final | Task 2 Final |")
    lines.append("|-------|-----------|--------|--------------|--------------|--------------|")

    for name, results in all_results.items():
        # Get final metrics
        final_key = max(results['task_metrics'].keys())
        final_metrics = results['task_metrics'][final_key]

        # Calculate averages
        aucs = [m['image_auc'] for m in final_metrics.values()]
        avg_auc = np.mean(aucs)

        # Calculate average forgetting
        fms = [f.get('forgetting_measure', 0) for f in results.get('forgetting', {}).values()]
        avg_fm = np.mean(fms) if fms else 0

        # Get per-task final AUC
        task_aucs = [final_metrics.get(str(i), {}).get('image_auc', 0) for i in range(3)]

        lines.append(
            f"| {name:<6} | {avg_auc:.4f}    | {avg_fm:+.4f} | "
            f"{task_aucs[0]:.4f}       | {task_aucs[1]:.4f}       | {task_aucs[2]:.4f}       |"
        )

    return "\n".join(lines)


def generate_forgetting_report(
    log_dir: str,
    model_names: List[str]
) -> str:
    """Generate a detailed forgetting analysis report."""
    log_path = Path(log_dir)

    lines = ["\n## Forgetting Analysis\n"]
    lines.append("Forgetting Measure (FM) = Initial AUC - Final AUC\n")
    lines.append("Positive FM indicates performance degradation.\n")

    for name in model_names:
        result_path = log_path / f"{name}_baseline" / 'results.json'
        if not result_path.exists():
            continue

        with open(result_path) as f:
            results = json.load(f)

        lines.append(f"\n### {name}\n")
        lines.append("| Task | Initial AUC | Final AUC | FM |")
        lines.append("|------|-------------|-----------|-----|")

        for tid, fdata in sorted(results.get('forgetting', {}).items()):
            fm = fdata.get('forgetting_measure', 0)
            lines.append(
                f"| {tid} | {fdata['initial_auc']:.4f} | "
                f"{fdata.get('final_auc', fdata['initial_auc']):.4f} | {fm:+.4f} |"
            )

    return "\n".join(lines)
