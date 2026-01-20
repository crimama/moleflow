#!/usr/bin/env python
"""
Memory Bank Baseline for Continual Anomaly Detection (Pilot Experiment).

This implements a PatchCore-style Memory Bank approach as a baseline to demonstrate
that Memory Bank methods have NO learnable parameters for "Base + Task-specific"
decomposition. The only way to handle new tasks is to store their features,
which is equivalent to data replay.

Key Point:
-----------
Unlike MoLE-Flow (which decomposes parameters into Base NF + Task-specific LoRA),
Memory Bank methods must:
  - Option A (Task-Separated): Store separate memory banks per task
              -> Requires task ID at inference (or router)
  - Option B (Accumulated): Accumulate all features into single memory bank
              -> This IS replay - storing previous task data

Architecture:
-------------
- Input: Patch embeddings from frozen WideResNet-50 backbone (768-dim)
- Memory Bank: Store representative patch embeddings per task
- Anomaly Score: Distance to nearest neighbor in memory bank

Usage:
------
    # Task-separated mode (each task has its own memory bank)
    python memory_bank_baseline.py --mode task_separated \\
        --task_classes leather grid transistor

    # Accumulated mode (single memory bank, features from all tasks)
    python memory_bank_baseline.py --mode accumulated \\
        --task_classes leather grid transistor
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from moleflow.data.datasets import get_dataset_class, create_task_dataset
from moleflow.extractors.cnn_extractor import CNNPatchCoreExtractor


# =============================================================================
# Memory Bank Classes
# =============================================================================

class MemoryBank:
    """
    Memory Bank for storing and retrieving patch embeddings.

    Core component of PatchCore-style anomaly detection.
    Stores representative patch embeddings and provides kNN-based anomaly scoring.

    Key Insight for Continual Learning:
        This class has NO learnable parameters. It only stores data (features).
        Therefore, handling new tasks requires either:
        1. Storing task-specific features (equivalent to replay)
        2. Accumulating all features (explicit replay)

    Attributes:
        memory: (N, D) tensor of stored patch embeddings
        task_id: Optional task identifier for task-separated mode
        max_size: Maximum number of patches to store
        sampling_ratio: Ratio of patches to sample from each image
        device: Device to store memory on
    """

    def __init__(
        self,
        max_size: int = 100000,
        sampling_ratio: float = 0.1,
        task_id: Optional[int] = None,
        device: str = 'cuda'
    ):
        """
        Initialize Memory Bank.

        Args:
            max_size: Maximum number of patches to store in memory
            sampling_ratio: Fraction of patches to sample from each image
            task_id: Optional task identifier for task-separated mode
            device: Device to store memory on
        """
        self.max_size = max_size
        self.sampling_ratio = sampling_ratio
        self.task_id = task_id
        self.device = device

        self.memory: Optional[torch.Tensor] = None
        self.n_samples = 0

    def update(self, features: torch.Tensor) -> int:
        """
        Add features to memory bank with random sampling.

        Args:
            features: (B, H, W, D) or (N, D) patch embeddings

        Returns:
            Number of patches actually added
        """
        # Flatten spatial dimensions if needed
        if features.dim() == 4:
            B, H, W, D = features.shape
            features = features.reshape(-1, D)  # (B*H*W, D)

        features = features.to(self.device)
        N = features.shape[0]

        # Random sampling
        n_samples = max(1, int(N * self.sampling_ratio))
        indices = torch.randperm(N)[:n_samples]
        sampled_features = features[indices]

        # Add to memory
        if self.memory is None:
            self.memory = sampled_features
        else:
            self.memory = torch.cat([self.memory, sampled_features], dim=0)

        self.n_samples = self.memory.shape[0]

        # Subsample if exceeds max_size
        if self.n_samples > self.max_size:
            indices = torch.randperm(self.n_samples)[:self.max_size]
            self.memory = self.memory[indices]
            self.n_samples = self.max_size

        return n_samples

    def get_memory(self) -> Optional[torch.Tensor]:
        """Return stored memory tensor."""
        return self.memory

    def compute_distances(self, query: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute k-nearest neighbor distances from query to memory.

        Args:
            query: (N, D) query patch embeddings
            k: Number of nearest neighbors

        Returns:
            distances: (N, k) distances to k nearest neighbors
            indices: (N, k) indices of k nearest neighbors
        """
        if self.memory is None or self.n_samples == 0:
            raise ValueError("Memory bank is empty. Call update() first.")

        query = query.to(self.device)

        # Compute pairwise L2 distances
        # query: (N, D), memory: (M, D)
        # distances: (N, M)
        distances = torch.cdist(query, self.memory, p=2)

        # Get k smallest distances
        top_k_distances, top_k_indices = torch.topk(distances, k, dim=1, largest=False)

        return top_k_distances, top_k_indices

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return f"MemoryBank(n_samples={self.n_samples}, max_size={self.max_size}, task_id={self.task_id})"


class MultiTaskMemoryBank:
    """
    Manager for multiple task-specific memory banks.

    Supports two modes:
    1. Task-Separated: Each task has its own memory bank
    2. Accumulated: Single memory bank accumulates all tasks' features

    Key Insight:
        Both modes require storing features from each task.
        - Task-Separated: Stores features per task (implicit replay via storage)
        - Accumulated: Explicitly accumulates all features (explicit replay)

        Neither mode has learnable parameters that can decompose into
        "base + task-specific" like MoLE-Flow's LoRA approach.
    """

    def __init__(
        self,
        mode: str = 'task_separated',
        max_size_per_task: int = 50000,
        total_max_size: int = 200000,
        sampling_ratio: float = 0.1,
        device: str = 'cuda'
    ):
        """
        Initialize MultiTaskMemoryBank.

        Args:
            mode: 'task_separated' or 'accumulated'
            max_size_per_task: Max patches per task (for task_separated mode)
            total_max_size: Total max patches (for accumulated mode)
            sampling_ratio: Fraction of patches to sample
            device: Device to store memory on
        """
        if mode not in ['task_separated', 'accumulated']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'task_separated' or 'accumulated'")

        self.mode = mode
        self.max_size_per_task = max_size_per_task
        self.total_max_size = total_max_size
        self.sampling_ratio = sampling_ratio
        self.device = device

        # Task-separated: dict of memory banks
        self.task_memories: Dict[int, MemoryBank] = {}

        # Accumulated: single memory bank
        self.accumulated_memory: Optional[MemoryBank] = None
        if mode == 'accumulated':
            self.accumulated_memory = MemoryBank(
                max_size=total_max_size,
                sampling_ratio=sampling_ratio,
                task_id=None,
                device=device
            )

    def add_task(self, task_id: int) -> None:
        """
        Initialize memory bank for a new task (task_separated mode only).

        Args:
            task_id: Task identifier
        """
        if self.mode == 'task_separated':
            if task_id not in self.task_memories:
                self.task_memories[task_id] = MemoryBank(
                    max_size=self.max_size_per_task,
                    sampling_ratio=self.sampling_ratio,
                    task_id=task_id,
                    device=self.device
                )

    def update(self, features: torch.Tensor, task_id: int) -> int:
        """
        Add features to appropriate memory bank.

        Args:
            features: (B, H, W, D) or (N, D) patch embeddings
            task_id: Task identifier

        Returns:
            Number of patches added
        """
        if self.mode == 'task_separated':
            if task_id not in self.task_memories:
                self.add_task(task_id)
            return self.task_memories[task_id].update(features)
        else:  # accumulated
            return self.accumulated_memory.update(features)

    def get_memory(self, task_id: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Get memory bank for scoring.

        Args:
            task_id: Task identifier (required for task_separated mode)

        Returns:
            Memory tensor or None if empty
        """
        if self.mode == 'task_separated':
            if task_id is None:
                raise ValueError("task_id required for task_separated mode")
            if task_id not in self.task_memories:
                return None
            return self.task_memories[task_id].get_memory()
        else:  # accumulated
            return self.accumulated_memory.get_memory()

    def compute_distances(
        self,
        query: torch.Tensor,
        k: int = 1,
        task_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute k-NN distances to memory bank.

        Args:
            query: (N, D) query embeddings
            k: Number of nearest neighbors
            task_id: Task identifier (required for task_separated mode)

        Returns:
            distances: (N, k) distances
            indices: (N, k) indices
        """
        if self.mode == 'task_separated':
            if task_id is None:
                raise ValueError("task_id required for task_separated mode")
            return self.task_memories[task_id].compute_distances(query, k)
        else:  # accumulated
            return self.accumulated_memory.compute_distances(query, k)

    def get_stats(self) -> Dict:
        """Get memory usage statistics."""
        if self.mode == 'task_separated':
            return {
                'mode': 'task_separated',
                'num_tasks': len(self.task_memories),
                'task_sizes': {t: len(m) for t, m in self.task_memories.items()},
                'total_size': sum(len(m) for m in self.task_memories.values())
            }
        else:
            return {
                'mode': 'accumulated',
                'total_size': len(self.accumulated_memory) if self.accumulated_memory else 0
            }


# =============================================================================
# Memory Bank Model
# =============================================================================

class MemoryBankModel(nn.Module):
    """
    Complete Memory Bank model for anomaly detection.

    Combines:
    1. Frozen CNN backbone for feature extraction
    2. MultiTaskMemoryBank for storing/retrieving features
    3. kNN-based anomaly scoring

    Key Design Point:
        This model has NO learnable parameters (backbone is frozen).
        The only way to adapt to new tasks is to store new features.
        This is fundamentally different from MoLE-Flow's parameter decomposition.
    """

    def __init__(
        self,
        backbone_name: str = 'wide_resnet50_2',
        embed_dim: int = 768,
        img_size: int = 224,
        memory_mode: str = 'task_separated',
        max_memory_per_task: int = 50000,
        total_max_memory: int = 200000,
        sampling_ratio: float = 0.1,
        k_neighbors: int = 9,
        device: str = 'cuda'
    ):
        """
        Initialize MemoryBankModel.

        Args:
            backbone_name: CNN backbone name (from timm)
            embed_dim: Target embedding dimension
            img_size: Input image size
            memory_mode: 'task_separated' or 'accumulated'
            max_memory_per_task: Max patches per task (task_separated mode)
            total_max_memory: Total max patches (accumulated mode)
            sampling_ratio: Fraction of patches to sample
            k_neighbors: Number of neighbors for kNN scoring
            device: Device for computation
        """
        super().__init__()

        self.backbone_name = backbone_name
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.memory_mode = memory_mode
        self.k_neighbors = k_neighbors
        self.device = device

        # Initialize frozen backbone
        self.feature_extractor = CNNPatchCoreExtractor(
            backbone_name=backbone_name,
            input_shape=(3, img_size, img_size),
            target_embed_dimension=embed_dim,
            device=device
        )
        self.feature_extractor.eval()

        # Freeze all backbone parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Initialize memory bank manager
        self.memory_bank = MultiTaskMemoryBank(
            mode=memory_mode,
            max_size_per_task=max_memory_per_task,
            total_max_size=total_max_memory,
            sampling_ratio=sampling_ratio,
            device=device
        )

        # Store task info
        self.task_classes: Dict[int, List[str]] = {}
        self.current_task_id: int = 0

    def add_task(self, task_id: int, task_classes: List[str]) -> None:
        """Register a new task."""
        self.task_classes[task_id] = task_classes
        self.memory_bank.add_task(task_id)
        self.current_task_id = task_id

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract patch features from images.

        Args:
            images: (B, C, H, W) input images

        Returns:
            features: (B, H_patch, W_patch, D) patch embeddings
        """
        with torch.no_grad():
            features = self.feature_extractor(images, return_spatial_shape=False)
        return features

    def update_memory(self, images: torch.Tensor, task_id: int) -> int:
        """
        Extract features and add to memory bank.

        Args:
            images: (B, C, H, W) input images
            task_id: Task identifier

        Returns:
            Number of patches added
        """
        features = self.extract_features(images)
        return self.memory_bank.update(features, task_id)

    def compute_anomaly_scores(
        self,
        images: torch.Tensor,
        task_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute anomaly scores using kNN distance.

        Args:
            images: (B, C, H, W) input images
            task_id: Task identifier (required for task_separated mode)

        Returns:
            patch_scores: (B, H_patch, W_patch) patch-level anomaly scores
            image_scores: (B,) image-level anomaly scores
        """
        features = self.extract_features(images)  # (B, H, W, D)
        B, H, W, D = features.shape

        # Flatten for kNN
        flat_features = features.reshape(-1, D)  # (B*H*W, D)

        # Get kNN distances
        distances, _ = self.memory_bank.compute_distances(
            flat_features, k=self.k_neighbors, task_id=task_id
        )

        # Use mean of k nearest distances as anomaly score
        patch_scores = distances.mean(dim=1)  # (B*H*W,)
        patch_scores = patch_scores.reshape(B, H, W)

        # Image-level score: max of top-k patches
        flat_scores = patch_scores.reshape(B, -1)
        top_k = min(self.k_neighbors, flat_scores.shape[1])
        top_k_scores, _ = torch.topk(flat_scores, top_k, dim=1)
        image_scores = top_k_scores.mean(dim=1)

        return patch_scores, image_scores

    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics."""
        return self.memory_bank.get_stats()


# =============================================================================
# Continual Trainer for Memory Bank
# =============================================================================

class MBContinualTrainer:
    """
    Continual Learning Trainer for Memory Bank baseline.

    Key Difference from MoLE-Flow:
        MoLE-Flow trains parameters: Base NF + Task-specific LoRA
        Memory Bank stores data: Features from each task

        Memory Bank has NO training loop - it only accumulates features.
        This demonstrates the fundamental limitation: no parameter decomposition.
    """

    def __init__(
        self,
        model: MemoryBankModel,
        device: str = 'cuda'
    ):
        """
        Initialize trainer.

        Args:
            model: MemoryBankModel instance
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.task_classes: Dict[int, List[str]] = {}

    def train_task(
        self,
        task_id: int,
        task_classes: List[str],
        train_loader: DataLoader,
        verbose: bool = True
    ) -> Dict:
        """
        "Train" on a new task by accumulating features to memory bank.

        Note: This is NOT training in the traditional sense.
        We simply extract features and store them. This demonstrates that
        Memory Bank methods require storing data (replay) for continual learning.

        Args:
            task_id: Task identifier
            task_classes: List of class names in this task
            train_loader: DataLoader for training data
            verbose: Print progress

        Returns:
            Dict with task statistics
        """
        self.task_classes[task_id] = task_classes
        self.model.add_task(task_id, task_classes)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Memory Bank: Accumulating features for Task {task_id}")
            print(f"Classes: {task_classes}")
            print(f"Mode: {self.model.memory_mode}")
            print(f"{'='*60}")

        total_patches = 0
        n_batches = len(train_loader)

        self.model.eval()

        for batch_idx, batch in enumerate(train_loader):
            images = batch[0].to(self.device)

            # Extract and store features
            n_added = self.model.update_memory(images, task_id)
            total_patches += n_added

            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx+1}/{n_batches}] Patches added: {total_patches}")

        stats = self.model.get_memory_stats()

        if verbose:
            print(f"\nTask {task_id} completed:")
            print(f"  Total patches added: {total_patches}")
            print(f"  Memory stats: {stats}")
            print(f"{'='*60}")

        return {
            'task_id': task_id,
            'task_classes': task_classes,
            'patches_added': total_patches,
            'memory_stats': stats
        }

    def evaluate_task(
        self,
        task_id: int,
        class_name: str,
        test_loader: DataLoader,
        target_size: int = 256
    ) -> Dict:
        """
        Evaluate anomaly detection performance on a single class.

        Args:
            task_id: Task identifier for memory bank selection
            class_name: Class name being evaluated
            test_loader: DataLoader for test data
            target_size: Size for pixel-level evaluation

        Returns:
            Dict with evaluation metrics
        """
        self.model.eval()

        gt_labels = []
        gt_masks = []
        pred_scores = []
        pred_image_scores = []

        for batch in test_loader:
            images, labels, masks = batch[0], batch[1], batch[2]
            images = images.to(self.device)

            # Compute anomaly scores
            patch_scores, image_scores = self.model.compute_anomaly_scores(
                images, task_id=task_id
            )

            # Resize patch scores to target size
            patch_scores_resized = F.interpolate(
                patch_scores.unsqueeze(1),
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=True
            ).squeeze(1)

            gt_labels.extend(labels.cpu().numpy())
            gt_masks.extend(masks.cpu().numpy())
            pred_scores.append(patch_scores_resized.cpu())
            pred_image_scores.append(image_scores.cpu())

        # Concatenate results
        pred_scores = torch.cat(pred_scores, dim=0).numpy()
        pred_image_scores = torch.cat(pred_image_scores, dim=0).numpy()
        gt_labels = np.array(gt_labels, dtype=bool)
        gt_masks = np.squeeze(np.array(gt_masks, dtype=bool), axis=1)

        # Apply Gaussian smoothing
        for i in range(pred_scores.shape[0]):
            pred_scores[i] = gaussian_filter(pred_scores[i], sigma=4)

        # Sanitize scores
        pred_image_scores = np.nan_to_num(pred_image_scores, nan=0.0, posinf=1.0, neginf=0.0)
        pred_scores = np.nan_to_num(pred_scores, nan=0.0, posinf=1.0, neginf=0.0)

        # Compute metrics
        img_auc = roc_auc_score(gt_labels, pred_image_scores)
        pixel_auc = roc_auc_score(gt_masks.flatten(), pred_scores.flatten())
        img_ap = average_precision_score(gt_labels, pred_image_scores)
        pixel_ap = average_precision_score(gt_masks.flatten(), pred_scores.flatten())

        return {
            'class_name': class_name,
            'img_auc': img_auc,
            'pixel_auc': pixel_auc,
            'img_ap': img_ap,
            'pixel_ap': pixel_ap,
            'n_samples': len(gt_labels)
        }


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_all_tasks(
    trainer: MBContinualTrainer,
    args,
    target_size: int = 256
) -> Dict:
    """
    Evaluate all learned tasks.

    Args:
        trainer: MBContinualTrainer instance
        args: Configuration object
        target_size: Size for pixel-level evaluation

    Returns:
        Dict with class-wise and average metrics
    """
    results = {
        'classes': [],
        'task_ids': [],
        'img_aucs': [],
        'pixel_aucs': [],
        'img_aps': [],
        'pixel_aps': [],
        'n_samples': []
    }

    print(f"\n{'='*70}")
    print(f"Evaluating All Tasks (Mode: {trainer.model.memory_mode})")
    print(f"{'='*70}")

    DatasetClass = get_dataset_class(args.dataset)

    for task_id, task_classes in trainer.task_classes.items():
        print(f"\nTask {task_id}: {task_classes}")

        for class_name in task_classes:
            # Create test dataset
            test_dataset = DatasetClass(
                root=args.data_path,
                class_name=class_name,
                train=False,
                img_size=args.img_size,
                crp_size=args.img_size,
                msk_size=target_size
            )
            test_loader = DataLoader(
                test_dataset, batch_size=8, shuffle=False,
                num_workers=4, pin_memory=True
            )

            # Evaluate
            # For task_separated mode: use task-specific memory
            # For accumulated mode: task_id is ignored (uses single memory)
            eval_task_id = task_id if trainer.model.memory_mode == 'task_separated' else None

            class_results = trainer.evaluate_task(
                eval_task_id, class_name, test_loader, target_size
            )

            results['classes'].append(class_name)
            results['task_ids'].append(task_id)
            results['img_aucs'].append(class_results['img_auc'])
            results['pixel_aucs'].append(class_results['pixel_auc'])
            results['img_aps'].append(class_results['img_ap'])
            results['pixel_aps'].append(class_results['pixel_ap'])
            results['n_samples'].append(class_results['n_samples'])

            print(f"  {class_name}: I-AUC={class_results['img_auc']:.4f}, "
                  f"P-AUC={class_results['pixel_auc']:.4f}, "
                  f"I-AP={class_results['img_ap']:.4f}, "
                  f"P-AP={class_results['pixel_ap']:.4f}")

    # Compute averages
    results['mean_img_auc'] = np.mean(results['img_aucs'])
    results['mean_pixel_auc'] = np.mean(results['pixel_aucs'])
    results['mean_img_ap'] = np.mean(results['img_aps'])
    results['mean_pixel_ap'] = np.mean(results['pixel_aps'])

    print(f"\n{'='*70}")
    print(f"Summary (Memory Bank - {trainer.model.memory_mode})")
    print(f"{'='*70}")
    print(f"  Mean Image AUC:  {results['mean_img_auc']:.4f}")
    print(f"  Mean Pixel AUC:  {results['mean_pixel_auc']:.4f}")
    print(f"  Mean Image AP:   {results['mean_img_ap']:.4f}")
    print(f"  Mean Pixel AP:   {results['mean_pixel_ap']:.4f}")
    print(f"  Memory Stats:    {trainer.model.get_memory_stats()}")
    print(f"{'='*70}")

    return results


def save_results(
    results: Dict,
    config: Dict,
    save_dir: str
) -> None:
    """
    Save evaluation results to files.

    Args:
        results: Evaluation results dict
        config: Experiment configuration dict
        save_dir: Directory to save results
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(save_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save results as JSON
    results_json = {k: v if not isinstance(v, np.ndarray) else v.tolist()
                    for k, v in results.items()}
    with open(save_path / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    # Save results as CSV
    import csv
    with open(save_path / 'results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Task', 'Image_AUC', 'Pixel_AUC', 'Image_AP', 'Pixel_AP'])
        for i, cls in enumerate(results['classes']):
            writer.writerow([
                cls,
                results['task_ids'][i],
                f"{results['img_aucs'][i]:.4f}",
                f"{results['pixel_aucs'][i]:.4f}",
                f"{results['img_aps'][i]:.4f}",
                f"{results['pixel_aps'][i]:.4f}"
            ])
        writer.writerow([])
        writer.writerow(['Mean', '',
                        f"{results['mean_img_auc']:.4f}",
                        f"{results['mean_pixel_auc']:.4f}",
                        f"{results['mean_img_ap']:.4f}",
                        f"{results['mean_pixel_ap']:.4f}"])

    print(f"\nResults saved to: {save_path}")


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Memory Bank Baseline for Continual Anomaly Detection'
    )

    # Data arguments
    parser.add_argument('--data_path', type=str, default='/Data/MVTecAD',
                        help='Path to dataset root')
    parser.add_argument('--dataset', type=str, default='mvtec',
                        choices=['mvtec', 'visa', 'mpdd'],
                        help='Dataset to use')
    parser.add_argument('--task_classes', type=str, nargs='+',
                        default=['leather', 'grid', 'transistor'],
                        help='Classes to learn sequentially')

    # Model arguments
    parser.add_argument('--backbone_name', type=str, default='wide_resnet50_2',
                        help='Backbone model name')
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='Embedding dimension')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--msk_size', type=int, default=256,
                        help='Mask size for evaluation')

    # Memory Bank arguments
    parser.add_argument('--mode', type=str, default='task_separated',
                        choices=['task_separated', 'accumulated'],
                        help='Memory bank mode')
    parser.add_argument('--max_memory_per_task', type=int, default=50000,
                        help='Max patches per task (task_separated mode)')
    parser.add_argument('--total_max_memory', type=int, default=200000,
                        help='Total max patches (accumulated mode)')
    parser.add_argument('--sampling_ratio', type=float, default=0.1,
                        help='Ratio of patches to sample')
    parser.add_argument('--k_neighbors', type=int, default=9,
                        help='Number of neighbors for kNN')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    # Output arguments
    parser.add_argument('--log_dir', type=str,
                        default='/Volume/MoLeFlow/logs/PilotExperiment/MemoryBank_baseline',
                        help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Setup experiment name and save directory
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_str = "_".join(args.task_classes[:3])
        args.experiment_name = f"MB_{args.mode}_{task_str}_{timestamp}"

    save_dir = Path(args.log_dir) / args.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print(f"\n{'='*70}")
    print("Memory Bank Baseline - Continual Anomaly Detection")
    print(f"{'='*70}")
    print(f"  Mode: {args.mode}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Tasks: {args.task_classes}")
    print(f"  Backbone: {args.backbone_name}")
    print(f"  Embed Dim: {args.embed_dim}")
    print(f"  Image Size: {args.img_size}")
    print(f"  k-Neighbors: {args.k_neighbors}")
    print(f"  Max Memory/Task: {args.max_memory_per_task}")
    print(f"  Sampling Ratio: {args.sampling_ratio}")
    print(f"  Save Dir: {save_dir}")
    print(f"{'='*70}")

    # Initialize model
    model = MemoryBankModel(
        backbone_name=args.backbone_name,
        embed_dim=args.embed_dim,
        img_size=args.img_size,
        memory_mode=args.mode,
        max_memory_per_task=args.max_memory_per_task,
        total_max_memory=args.total_max_memory,
        sampling_ratio=args.sampling_ratio,
        k_neighbors=args.k_neighbors,
        device=device
    )
    print("\nMemory Bank Model initialized")
    print(f"  Feature Extractor: {args.backbone_name} (Frozen)")
    print(f"  Memory Mode: {args.mode}")

    # Initialize trainer
    trainer = MBContinualTrainer(model, device)

    # Setup continual learning tasks (1 class per task)
    ALL_CLASSES = args.task_classes
    GLOBAL_CLASS_TO_IDX = {cls: i for i, cls in enumerate(ALL_CLASSES)}
    CONTINUAL_TASKS = [[cls] for cls in ALL_CLASSES]  # 1 class per task

    print(f"\nContinual Learning Setup:")
    for t_id, t_classes in enumerate(CONTINUAL_TASKS):
        print(f"  Task {t_id}: {t_classes}")

    # Training loop (feature accumulation)
    print(f"\n{'#'*70}")
    print("# Phase 1: Feature Accumulation (\"Training\")")
    print(f"{'#'*70}")

    DatasetClass = get_dataset_class(args.dataset)

    for task_id, task_classes in enumerate(CONTINUAL_TASKS):
        # Create training dataset
        train_dataset = create_task_dataset(
            args, task_classes, GLOBAL_CLASS_TO_IDX, train=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=False
        )

        # Accumulate features
        trainer.train_task(task_id, task_classes, train_loader)

    # Evaluation
    print(f"\n{'#'*70}")
    print("# Phase 2: Evaluation")
    print(f"{'#'*70}")

    results = evaluate_all_tasks(trainer, args, target_size=args.msk_size)

    # Save results
    config = {
        'experiment_name': args.experiment_name,
        'mode': args.mode,
        'dataset': args.dataset,
        'task_classes': args.task_classes,
        'backbone_name': args.backbone_name,
        'embed_dim': args.embed_dim,
        'img_size': args.img_size,
        'k_neighbors': args.k_neighbors,
        'max_memory_per_task': args.max_memory_per_task,
        'sampling_ratio': args.sampling_ratio,
        'seed': args.seed,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'memory_stats': model.get_memory_stats()
    }

    save_results(results, config, str(save_dir))

    # Print key insight
    print(f"\n{'='*70}")
    print("KEY INSIGHT: Memory Bank vs MoLE-Flow")
    print(f"{'='*70}")
    print("""
Memory Bank (this baseline):
  - NO learnable parameters (backbone frozen)
  - Handles new tasks by STORING features
  - Task-Separated mode: Stores features per task (implicit replay)
  - Accumulated mode: Stores ALL features (explicit replay)
  => Both modes require data storage = DATA REPLAY

MoLE-Flow:
  - Learnable parameters: Base NF + Task-specific LoRA
  - Handles new tasks by TRAINING new LoRA adapters
  - No feature storage required
  - Parameters decompose into: Shared (Base NF) + Task-specific (LoRA)
  => PARAMETER DECOMPOSITION (no replay needed)

This demonstrates why Memory Bank methods cannot achieve true
continual learning without replay: they have no parameters to decompose.
""")
    print(f"{'='*70}")

    return results


if __name__ == '__main__':
    main()
