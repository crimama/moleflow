#!/usr/bin/env python3
"""
AutoEncoder Baseline for Continual Anomaly Detection (Pilot Experiment)

This script implements an AutoEncoder baseline to demonstrate that the
"Base frozen + Task-specific trainable" parameter decomposition works for
Normalizing Flows but NOT for reconstruction-based methods.

Key Hypothesis:
- NF (MoLE-Flow): Base captures the shared transformation structure,
  task-specific LoRA adapts the density estimation. Frozen encoder still
  provides meaningful latent space.
- AE: Encoder-Decoder are tightly coupled. Freezing encoder after Task 0
  means Task 1+ cannot adapt the latent space representation, leading to
  poor reconstruction for new tasks.

Architecture:
- Input: Patch embeddings from frozen WideResNet-50 backbone (768-dim)
- Encoder: 768 -> 512 -> latent_dim (e.g., 256)
- Decoder: latent_dim -> 512 -> 768
- Anomaly Score: Reconstruction error ||x - x_hat||^2

Continual Learning Setup:
- Task 0: Train both Encoder + Decoder jointly
- After Task 0: FREEZE Encoder (this is the "Base")
- Task 1+: Only train Decoder (task-specific)

Expected Result:
- Task 0 performance: Good (both E & D trained together)
- Task 1+ performance: Poor (frozen E cannot adapt to new task distribution)
- Catastrophic forgetting: After Task 1+, Task 0 performance may degrade
  because Decoder changed without Encoder adaptation

Usage:
    python ae_baseline.py --task_classes leather grid transistor \\
                          --num_epochs 40 --latent_dim 256

Author: MoLE-Flow Pilot Experiment
"""

import os
import sys
import json
import argparse
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import gaussian_filter

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from moleflow.extractors import create_feature_extractor
from moleflow.data.mvtec import MVTEC


# =============================================================================
# AutoEncoder Model
# =============================================================================

class Encoder(nn.Module):
    """
    Encoder network: Maps patch embeddings to a latent space.

    Architecture:
        input_dim -> hidden_dim -> latent_dim

    This component is FROZEN after Task 0 in the continual learning setup.
    The hypothesis is that freezing the encoder prevents the model from
    adapting to new task distributions, unlike NF where the frozen base
    still provides meaningful transformations.

    Args:
        input_dim: Input dimension (e.g., 768 for WideResNet-50)
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
        dropout: Dropout rate for regularization
    """

    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 512,
                 latent_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode patch embeddings to latent space.

        Args:
            x: (B, H, W, D) or (B*H*W, D) patch embeddings

        Returns:
            z: Latent representation with same batch structure
        """
        original_shape = x.shape

        # Flatten spatial dimensions if needed
        if len(original_shape) == 4:
            B, H, W, D = original_shape
            x = x.reshape(B * H * W, D)

        z = self.encoder(x)

        # Restore spatial dimensions if needed
        if len(original_shape) == 4:
            z = z.reshape(B, H, W, -1)

        return z


class Decoder(nn.Module):
    """
    Decoder network: Reconstructs patch embeddings from latent space.

    Architecture:
        latent_dim -> hidden_dim -> output_dim

    This component is trained for ALL tasks in the continual learning setup.
    After Task 0, only the Decoder is trainable (task-specific).

    The hypothesis is that without encoder adaptation, the decoder alone
    cannot learn meaningful reconstructions for new tasks whose feature
    distributions differ from Task 0.

    Args:
        latent_dim: Latent space dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (should match input_dim of Encoder)
        dropout: Dropout rate for regularization
    """

    def __init__(self,
                 latent_dim: int = 256,
                 hidden_dim: int = 512,
                 output_dim: int = 768,
                 dropout: float = 0.1):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to patch embeddings.

        Args:
            z: (B, H, W, latent_dim) or (B*H*W, latent_dim) latent codes

        Returns:
            x_hat: Reconstructed patch embeddings with same batch structure
        """
        original_shape = z.shape

        # Flatten spatial dimensions if needed
        if len(original_shape) == 4:
            B, H, W, D = original_shape
            z = z.reshape(B * H * W, D)

        x_hat = self.decoder(z)

        # Restore spatial dimensions if needed
        if len(original_shape) == 4:
            x_hat = x_hat.reshape(B, H, W, -1)

        return x_hat


class AEModel(nn.Module):
    """
    AutoEncoder Model for Anomaly Detection.

    Combines Encoder and Decoder for reconstruction-based anomaly detection.
    Anomaly score is computed as reconstruction error ||x - x_hat||^2.

    Continual Learning Strategy (Base + Task-specific decomposition):
    - Task 0: Train Encoder (E) + Decoder (D) jointly
    - Task 0 complete: FREEZE Encoder (Base component)
    - Task 1+: Train only Decoder (Task-specific component)

    This mirrors MoLE-Flow's strategy:
    - NF Base = Encoder (frozen after Task 0)
    - Task LoRA = Decoder (trainable for all tasks)

    Hypothesis: This decomposition FAILS for AE because:
    1. Encoder maps Task 0 distribution to latent space
    2. For Task 1+ with different distribution, frozen Encoder
       produces suboptimal latent codes
    3. Decoder cannot compensate for poor latent representations

    Args:
        input_dim: Input/output dimension of patch embeddings
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
        dropout: Dropout rate
    """

    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 512,
                 latent_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode.

        Args:
            x: (B, H, W, D) patch embeddings

        Returns:
            x_hat: (B, H, W, D) reconstructed embeddings
            z: (B, H, W, latent_dim) latent representations
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        return self.decoder(z)

    def freeze_encoder(self):
        """
        Freeze encoder parameters (called after Task 0).

        This is the critical step that differentiates the CL strategy:
        - Encoder becomes "Base" (frozen, shared across tasks)
        - Decoder becomes "Task-specific" (trainable)
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("[AE] Encoder FROZEN - only Decoder trainable from now on")

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters (for ablation studies)."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("[AE] Encoder UNFROZEN")

    def get_trainable_params(self, task_id: int) -> List[torch.nn.Parameter]:
        """
        Get trainable parameters based on task ID.

        Args:
            task_id: Current task ID

        Returns:
            List of trainable parameters
        """
        if task_id == 0:
            # Task 0: Both encoder and decoder
            return list(self.parameters())
        else:
            # Task 1+: Only decoder (encoder frozen)
            return list(self.decoder.parameters())

    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute patch-wise reconstruction error (anomaly score).

        Args:
            x: (B, H, W, D) patch embeddings

        Returns:
            error: (B, H, W) reconstruction error per patch
        """
        x_hat, _ = self.forward(x)
        # MSE per patch: sum over feature dimension
        error = ((x - x_hat) ** 2).sum(dim=-1)  # (B, H, W)
        return error


# =============================================================================
# Continual Trainer
# =============================================================================

class AEContinualTrainer:
    """
    Continual Learning Trainer for AutoEncoder Baseline.

    Implements the "Base frozen + Task-specific trainable" strategy:
    - Task 0: Train full AE (Encoder + Decoder)
    - Task 0 complete: Freeze Encoder
    - Task 1+: Train only Decoder

    Tracks:
    - Training loss convergence per task
    - Image AUC on current task
    - Forgetting Measure (performance on previous tasks)

    Args:
        feature_extractor: Frozen backbone for feature extraction
        ae_model: AutoEncoder model
        device: Device for computation
        log_dir: Directory for saving logs and results
    """

    def __init__(self,
                 feature_extractor,
                 ae_model: AEModel,
                 device: str = 'cuda',
                 log_dir: str = './logs'):

        self.feature_extractor = feature_extractor
        self.ae_model = ae_model
        self.device = device
        self.log_dir = log_dir

        # Task tracking
        self.task_classes: Dict[int, List[str]] = {}
        self.current_task_id = -1

        # Performance tracking
        self.training_history: Dict[int, List[float]] = defaultdict(list)
        self.task_performance: Dict[int, Dict[str, float]] = {}
        self.forgetting_measures: Dict[int, Dict[int, float]] = defaultdict(dict)

        # Initial performance after each task (for forgetting calculation)
        self.initial_performance: Dict[int, float] = {}

        # Move model to device
        self.ae_model.to(device)
        self.feature_extractor.to(device)
        self.feature_extractor.eval()

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

    def train_task(self,
                   task_id: int,
                   task_classes: List[str],
                   train_loader: DataLoader,
                   num_epochs: int = 40,
                   lr: float = 1e-3,
                   log_interval: int = 10):
        """
        Train on a single task.

        Task 0: Train both Encoder and Decoder
        Task 1+: Train only Decoder (Encoder frozen)

        Args:
            task_id: Task identifier
            task_classes: List of class names in this task
            train_loader: DataLoader for training data
            num_epochs: Number of training epochs
            lr: Learning rate
            log_interval: Batch interval for logging
        """
        print(f"\n{'='*70}")
        print(f"[AE Baseline] Training Task {task_id}: {task_classes}")
        print(f"{'='*70}")

        self.task_classes[task_id] = task_classes
        self.current_task_id = task_id

        # Freeze encoder after Task 0
        if task_id > 0 and task_id == 1:
            self.ae_model.freeze_encoder()

        # Get trainable parameters
        trainable_params = self.ae_model.get_trainable_params(task_id)
        num_params = sum(p.numel() for p in trainable_params if p.requires_grad)

        phase = "Full AE (E+D)" if task_id == 0 else "Decoder Only (E frozen)"
        print(f"  Phase: {phase}")
        print(f"  Trainable Parameters: {num_params:,}")
        print(f"  Learning Rate: {lr:.2e}")
        print(f"  Epochs: {num_epochs}")

        # Optimizer
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=lr * 0.01
        )

        # Training loop
        self.ae_model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (images, labels, masks, names, paths) in enumerate(train_loader):
                images = images.to(self.device)

                # Extract features using frozen backbone
                with torch.no_grad():
                    patch_embeddings, spatial_shape = self.feature_extractor(
                        images, return_spatial_shape=True
                    )
                    B = patch_embeddings.shape[0]
                    H, W = spatial_shape
                    patch_embeddings = patch_embeddings.reshape(B, H, W, -1)

                # Forward pass
                x_hat, z = self.ae_model(patch_embeddings)

                # Reconstruction loss (MSE)
                loss = F.mse_loss(x_hat, patch_embeddings)

                # Check for NaN
                if torch.isnan(loss):
                    print(f"  WARNING: NaN loss at batch {batch_idx}, skipping...")
                    continue

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # Logging
                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"  Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {loss.item():.6f} | Avg: {avg_loss:.6f}")

            scheduler.step()

            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            self.training_history[task_id].append(avg_epoch_loss)

            # Log epoch summary
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  [Epoch {epoch+1}/{num_epochs}] Avg Loss: {avg_epoch_loss:.6f} | LR: {current_lr:.2e}")

        print(f"\n  Task {task_id} training completed!")
        print(f"  Final Loss: {self.training_history[task_id][-1]:.6f}")

    def evaluate(self,
                 task_id: int,
                 test_loader: DataLoader,
                 target_size: int = 224) -> Dict[str, float]:
        """
        Evaluate on a single task.

        Args:
            task_id: Task to evaluate
            test_loader: DataLoader for test data
            target_size: Target size for pixel-level evaluation

        Returns:
            Dict containing img_auc, pixel_auc, img_ap, pixel_ap
        """
        self.ae_model.eval()

        all_anomaly_scores = []
        all_image_scores = []
        all_gt_labels = []
        all_gt_masks = []

        with torch.no_grad():
            for images, labels, masks, names, paths in test_loader:
                images = images.to(self.device)

                # Extract features
                patch_embeddings, spatial_shape = self.feature_extractor(
                    images, return_spatial_shape=True
                )
                B = patch_embeddings.shape[0]
                H, W = spatial_shape
                patch_embeddings = patch_embeddings.reshape(B, H, W, -1)

                # Compute reconstruction error
                anomaly_scores = self.ae_model.compute_reconstruction_error(patch_embeddings)

                # Resize to target size
                anomaly_scores_resized = F.interpolate(
                    anomaly_scores.unsqueeze(1),
                    size=(target_size, target_size),
                    mode='bilinear',
                    align_corners=True
                ).squeeze(1)

                # Image-level score: max of patch scores
                image_scores = anomaly_scores.reshape(B, -1).max(dim=1)[0]

                all_anomaly_scores.append(anomaly_scores_resized.cpu().numpy())
                all_image_scores.append(image_scores.cpu().numpy())
                all_gt_labels.extend(labels.numpy())
                all_gt_masks.append(masks.numpy())

        # Concatenate results
        anomaly_scores_all = np.concatenate(all_anomaly_scores, axis=0)
        image_scores_all = np.concatenate(all_image_scores, axis=0)
        gt_labels = np.array(all_gt_labels, dtype=bool)
        gt_masks = np.concatenate(all_gt_masks, axis=0)
        gt_masks = np.squeeze(gt_masks, axis=1).astype(bool)

        # Apply Gaussian smoothing
        for i in range(anomaly_scores_all.shape[0]):
            anomaly_scores_all[i] = gaussian_filter(anomaly_scores_all[i], sigma=4)

        # Sanitize scores
        image_scores_all = np.nan_to_num(image_scores_all, nan=0.0, posinf=1.0, neginf=0.0)
        anomaly_scores_all = np.nan_to_num(anomaly_scores_all, nan=0.0, posinf=1.0, neginf=0.0)

        # Compute metrics
        try:
            img_auc = roc_auc_score(gt_labels, image_scores_all)
        except ValueError:
            img_auc = 0.5  # Only one class present

        try:
            pixel_auc = roc_auc_score(gt_masks.flatten(), anomaly_scores_all.flatten())
        except ValueError:
            pixel_auc = 0.5

        try:
            img_ap = average_precision_score(gt_labels, image_scores_all)
        except ValueError:
            img_ap = 0.0

        try:
            pixel_ap = average_precision_score(gt_masks.flatten(), anomaly_scores_all.flatten())
        except ValueError:
            pixel_ap = 0.0

        return {
            'img_auc': img_auc,
            'pixel_auc': pixel_auc,
            'img_ap': img_ap,
            'pixel_ap': pixel_ap
        }

    def evaluate_all_tasks(self, args, target_size: int = 224) -> Dict:
        """
        Evaluate on all learned tasks.

        Also computes forgetting measures for previous tasks.

        Args:
            args: Configuration with data_path, img_size, etc.
            target_size: Target size for pixel-level evaluation

        Returns:
            Dict with per-task and overall metrics
        """
        results = {
            'classes': [],
            'task_ids': [],
            'img_aucs': [],
            'pixel_aucs': [],
            'forgetting': {},
        }

        print(f"\n{'='*70}")
        print("Evaluating All Tasks")
        print(f"{'='*70}")

        for task_id, task_classes in self.task_classes.items():
            print(f"\nTask {task_id}: {task_classes}")

            for class_name in task_classes:
                # Create test dataset
                test_dataset = MVTEC(
                    args.data_path,
                    class_name=class_name,
                    train=False,
                    img_size=args.img_size,
                    crp_size=args.img_size,
                    msk_size=target_size
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=8,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )

                # Evaluate
                metrics = self.evaluate(task_id, test_loader, target_size)

                results['classes'].append(class_name)
                results['task_ids'].append(task_id)
                results['img_aucs'].append(metrics['img_auc'])
                results['pixel_aucs'].append(metrics['pixel_auc'])

                print(f"  {class_name}: I-AUC={metrics['img_auc']:.4f}, P-AUC={metrics['pixel_auc']:.4f}")

                # Store initial performance for forgetting calculation
                if task_id not in self.initial_performance:
                    self.initial_performance[task_id] = metrics['img_auc']

        # Compute averages
        results['mean_img_auc'] = np.mean(results['img_aucs'])
        results['mean_pixel_auc'] = np.mean(results['pixel_aucs'])

        # Compute forgetting for previous tasks
        if self.current_task_id > 0:
            for prev_task_id in range(self.current_task_id):
                if prev_task_id in self.initial_performance:
                    # Get current performance on previous task
                    prev_task_idx = [i for i, t in enumerate(results['task_ids']) if t == prev_task_id]
                    if prev_task_idx:
                        current_perf = np.mean([results['img_aucs'][i] for i in prev_task_idx])
                        initial_perf = self.initial_performance[prev_task_id]
                        forgetting = initial_perf - current_perf
                        self.forgetting_measures[self.current_task_id][prev_task_id] = forgetting
                        results['forgetting'][f"T{prev_task_id}"] = forgetting

        print(f"\n{'─'*70}")
        print(f"Overall: I-AUC={results['mean_img_auc']:.4f}, P-AUC={results['mean_pixel_auc']:.4f}")
        if results['forgetting']:
            print(f"Forgetting: {results['forgetting']}")
        print(f"{'='*70}")

        return results

    def save_results(self, results: Dict, filename: str = 'results.json'):
        """Save results to JSON file."""
        filepath = os.path.join(self.log_dir, filename)

        # Convert numpy types for JSON serialization
        serializable = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            elif isinstance(v, (np.float32, np.float64)):
                serializable[k] = float(v)
            else:
                serializable[k] = v

        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to: {filepath}")

    def save_training_curves(self, filename: str = 'training_curves.json'):
        """Save training loss curves."""
        filepath = os.path.join(self.log_dir, filename)

        curves = {}
        for task_id, losses in self.training_history.items():
            curves[f"task_{task_id}"] = losses

        with open(filepath, 'w') as f:
            json.dump(curves, f, indent=2)
        print(f"Training curves saved to: {filepath}")


# =============================================================================
# Main Function
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AutoEncoder Baseline for Continual Anomaly Detection"
    )

    # Dataset arguments
    parser.add_argument('--data_path', type=str, default='/Data/MVTecAD',
                        help='Path to MVTec AD dataset')
    parser.add_argument('--task_classes', nargs='+', type=str,
                        default=['leather', 'grid', 'transistor'],
                        help='Classes for each task (space-separated)')
    parser.add_argument('--classes_per_task', type=int, default=1,
                        help='Number of classes per task (for grouping)')

    # Model arguments
    parser.add_argument('--backbone_name', type=str, default='wide_resnet50_2',
                        help='Backbone for feature extraction')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--target_embed_dimension', type=int, default=768,
                        help='Feature embedding dimension')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent space dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='Number of training epochs per task')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Batch interval for logging')

    # Output arguments
    parser.add_argument('--log_dir', type=str,
                        default='./logs/PilotExperiment/AE_baseline',
                        help='Directory for logs and results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for computation')

    return parser.parse_args()


def create_task_loader(args, task_classes: List[str]) -> DataLoader:
    """Create DataLoader for a task."""
    from torch.utils.data import ConcatDataset

    datasets = []
    for class_name in task_classes:
        dataset = MVTEC(
            args.data_path,
            class_name=class_name,
            train=True,
            img_size=args.img_size,
            crp_size=args.img_size,
            msk_size=args.img_size
        )
        datasets.append(dataset)

    if len(datasets) == 1:
        combined = datasets[0]
    else:
        combined = ConcatDataset(datasets)

    return DataLoader(
        combined,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )


def main():
    """Main function for running the AE baseline experiment."""
    args = parse_args()

    # Create experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"AE_baseline_{timestamp}"

    # Update log directory
    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    os.makedirs(args.log_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(args.log_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("="*70)
    print("AutoEncoder Baseline for Continual Anomaly Detection")
    print("="*70)
    print(f"Experiment: {args.experiment_name}")
    print(f"Log Directory: {args.log_dir}")
    print(f"Task Classes: {args.task_classes}")
    print(f"Device: {args.device}")
    print("="*70)

    # Create feature extractor (frozen backbone)
    print("\nInitializing feature extractor...")
    feature_extractor = create_feature_extractor(
        backbone_name=args.backbone_name,
        input_shape=(3, args.img_size, args.img_size),
        target_embed_dimension=args.target_embed_dimension,
        device=args.device
    )
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False
    print(f"  Backbone: {args.backbone_name}")
    print(f"  Embedding Dimension: {args.target_embed_dimension}")

    # Create AutoEncoder model
    print("\nInitializing AutoEncoder model...")
    ae_model = AEModel(
        input_dim=args.target_embed_dimension,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout
    )
    total_params = sum(p.numel() for p in ae_model.parameters())
    print(f"  Latent Dimension: {args.latent_dim}")
    print(f"  Hidden Dimension: {args.hidden_dim}")
    print(f"  Total Parameters: {total_params:,}")

    # Create trainer
    trainer = AEContinualTrainer(
        feature_extractor=feature_extractor,
        ae_model=ae_model,
        device=args.device,
        log_dir=args.log_dir
    )

    # Group task classes
    task_class_groups = []
    for i in range(0, len(args.task_classes), args.classes_per_task):
        group = args.task_classes[i:i + args.classes_per_task]
        task_class_groups.append(group)

    print(f"\nTask Schedule ({len(task_class_groups)} tasks):")
    for i, group in enumerate(task_class_groups):
        freeze_status = "E frozen" if i > 0 else "E+D trainable"
        print(f"  Task {i}: {group} [{freeze_status}]")

    # Training loop
    all_results = {}

    for task_id, task_classes in enumerate(task_class_groups):
        # Create data loader for this task
        train_loader = create_task_loader(args, task_classes)
        print(f"\n  Training samples: {len(train_loader.dataset)}")

        # Train on this task
        trainer.train_task(
            task_id=task_id,
            task_classes=task_classes,
            train_loader=train_loader,
            num_epochs=args.num_epochs,
            lr=args.lr,
            log_interval=args.log_interval
        )

        # Evaluate on all tasks
        results = trainer.evaluate_all_tasks(args, target_size=args.img_size)
        all_results[f"after_task_{task_id}"] = results

        # Save intermediate results
        trainer.save_results(results, f'results_after_task_{task_id}.json')

    # Save final results
    final_results = {
        'task_classes': task_class_groups,
        'training_history': dict(trainer.training_history),
        'forgetting_measures': dict(trainer.forgetting_measures),
        'final_performance': all_results[f"after_task_{len(task_class_groups)-1}"],
        'all_results': all_results
    }
    trainer.save_results(final_results, 'final_results.json')
    trainer.save_training_curves()

    # Print final summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nFinal Performance (after {len(task_class_groups)} tasks):")
    final_perf = final_results['final_performance']
    print(f"  Mean Image AUC: {final_perf['mean_img_auc']:.4f}")
    print(f"  Mean Pixel AUC: {final_perf['mean_pixel_auc']:.4f}")

    if trainer.forgetting_measures:
        print(f"\nForgetting Measures:")
        for after_task, measures in trainer.forgetting_measures.items():
            for prev_task, forgetting in measures.items():
                print(f"  After Task {after_task}, Task {prev_task} forgetting: {forgetting:.4f}")

    print(f"\nResults saved to: {args.log_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
