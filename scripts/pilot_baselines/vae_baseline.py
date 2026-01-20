#!/usr/bin/env python
"""
VAE Baseline for Continual Anomaly Detection - Pilot Experiment

This script implements a Variational AutoEncoder (VAE) baseline to demonstrate that
parameter decomposition (Base frozen + Task-specific trainable) works for Normalizing
Flows but NOT for reconstruction-based methods.

Architecture:
    Input: Patch embeddings from frozen WideResNet-50 backbone (768-dim)
    Encoder: FC layers -> mu, log_var (latent_dim)
    Decoder: FC layers -> Reconstructed embedding (768-dim)
    Loss: Reconstruction MSE + beta * KL divergence
    Anomaly Score: Reconstruction error ||x - x_hat||^2

Continual Learning Setup (Base + Task-specific decomposition):
    Task 0: Train Encoder + Decoder (full model)
    Task 0 complete: Freeze Encoder (Base)
    Task 1+: Train Decoder only (Task-specific)

Expected Behavior:
    - This decomposition should FAIL for VAE because:
      1. Decoder alone cannot adapt to new input distributions
      2. Frozen encoder maps new tasks to suboptimal latent space
      3. Reconstruction quality degrades for new tasks

Usage:
    python vae_baseline.py --task_classes leather grid transistor --num_epochs 40

Author: MoLE-Flow Team
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score

# Add parent path for moleflow imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moleflow.extractors.cnn_extractor import CNNPatchCoreExtractor
from moleflow.data.mvtec import MVTEC, MVTEC_CLASS_NAMES


# =============================================================================
# VAE Model Components
# =============================================================================

class VAEEncoder(nn.Module):
    """
    VAE Encoder: Maps patch embeddings to latent distribution parameters.

    Architecture: FC -> ReLU -> FC -> ReLU -> (FC_mu, FC_logvar)

    Args:
        input_dim: Input embedding dimension (e.g., 768)
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
        dropout: Dropout probability
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512,
                 latent_dim: int = 256, dropout: float = 0.1):
        super(VAEEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)

        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.

        Args:
            x: Input embeddings (B, D) or (B*H*W, D)

        Returns:
            mu: Mean of latent distribution (B, latent_dim)
            log_var: Log variance of latent distribution (B, latent_dim)
        """
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.dropout(h)

        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)

        return mu, log_var


class VAEDecoder(nn.Module):
    """
    VAE Decoder: Reconstructs patch embeddings from latent samples.

    Architecture: FC -> ReLU -> FC -> ReLU -> FC

    Args:
        latent_dim: Latent space dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension (e.g., 768)
        dropout: Dropout probability
    """

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 512,
                 output_dim: int = 768, dropout: float = 0.1):
        super(VAEDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Decoder network
        self.fc1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            z: Latent samples (B, latent_dim)

        Returns:
            x_recon: Reconstructed embeddings (B, output_dim)
        """
        h = F.relu(self.bn1(self.fc1(z)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.dropout(h)
        x_recon = self.fc3(h)

        return x_recon


class VAEModel(nn.Module):
    """
    Complete VAE Model for Anomaly Detection.

    Combines encoder and decoder with reparameterization trick.

    Args:
        input_dim: Input/output embedding dimension (e.g., 768)
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
        dropout: Dropout probability
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512,
                 latent_dim: int = 256, dropout: float = 0.1):
        super(VAEModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Build encoder and decoder
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim, dropout)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim, dropout)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon

        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution

        Returns:
            z: Sampled latent vector
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, use mean for deterministic output
            return mu

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode -> reparameterize -> decode

        Args:
            x: Input embeddings (B, D)

        Returns:
            x_recon: Reconstructed embeddings
            mu: Latent mean
            log_var: Latent log variance
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)

        return x_recon, mu, log_var

    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor,
                     mu: torch.Tensor, log_var: torch.Tensor,
                     beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss: Reconstruction + beta * KL divergence

        Args:
            x: Original input
            x_recon: Reconstructed input
            mu: Latent mean
            log_var: Latent log variance
            beta: Weight for KL term (beta-VAE)

        Returns:
            Dict containing 'total', 'recon', 'kl' losses
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - var)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        # Total loss
        total_loss = recon_loss + beta * kl_loss

        return {
            'total': total_loss,
            'recon': recon_loss,
            'kl': kl_loss
        }

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly score as reconstruction error.

        Args:
            x: Input embeddings (B, D) or (B, H, W, D)

        Returns:
            scores: Reconstruction error per sample
        """
        original_shape = x.shape

        # Flatten spatial dimensions if present
        if len(x.shape) == 4:
            B, H, W, D = x.shape
            x_flat = x.reshape(B * H * W, D)
        else:
            x_flat = x
            B = x.shape[0]
            H, W = 1, 1

        # Forward pass
        with torch.no_grad():
            x_recon, _, _ = self.forward(x_flat)

        # Compute per-patch reconstruction error
        recon_error = (x_flat - x_recon).pow(2).sum(dim=-1)  # (B*H*W,)

        # Reshape back to spatial format if needed
        if len(original_shape) == 4:
            recon_error = recon_error.reshape(B, H, W)

        return recon_error

    def freeze_encoder(self):
        """Freeze encoder parameters (for Task 1+)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("   Encoder frozen (Base parameters)")

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("   Encoder unfrozen")

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder = sum(p.numel() for p in self.encoder.parameters())
        decoder = sum(p.numel() for p in self.decoder.parameters())
        return {
            'total': total,
            'trainable': trainable,
            'encoder': encoder,
            'decoder': decoder
        }


# =============================================================================
# Continual Learning Trainer
# =============================================================================

class VAEContinualTrainer:
    """
    Continual Learning Trainer for VAE Baseline.

    Implements parameter decomposition strategy:
    - Task 0: Train full model (Encoder + Decoder)
    - Task 1+: Freeze Encoder, train Decoder only

    This is expected to FAIL because:
    1. Frozen encoder cannot adapt to new input distributions
    2. Decoder alone cannot compensate for encoder's fixed representations
    """

    def __init__(self, vae_model: VAEModel, feature_extractor: nn.Module,
                 device: str = 'cuda', beta: float = 1.0):
        """
        Args:
            vae_model: VAE model instance
            feature_extractor: Frozen CNN feature extractor
            device: Device to use
            beta: KL divergence weight
        """
        self.vae = vae_model.to(device)
        self.extractor = feature_extractor
        self.extractor.eval()  # Always frozen
        self.device = device
        self.beta = beta

        # Task management
        self.task_classes: Dict[int, List[str]] = {}
        self.task_results: Dict[int, Dict] = {}

        # Training history
        self.history = {
            'task_id': [],
            'epoch': [],
            'train_loss': [],
            'recon_loss': [],
            'kl_loss': [],
        }

    def train_task(self, task_id: int, task_classes: List[str],
                   train_loader: DataLoader, num_epochs: int = 40,
                   lr: float = 1e-3, log_interval: int = 10):
        """
        Train VAE on a task.

        Args:
            task_id: Task identifier
            task_classes: List of class names in this task
            train_loader: DataLoader for training
            num_epochs: Number of epochs
            lr: Learning rate
            log_interval: Logging interval
        """
        print(f"\n{'='*70}")
        print(f"Task {task_id}: {task_classes}")
        print(f"{'='*70}")

        self.task_classes[task_id] = task_classes

        # Parameter management based on task
        if task_id == 0:
            print("   Mode: Full VAE Training (Encoder + Decoder)")
            self.vae.unfreeze_encoder()
        else:
            print("   Mode: Decoder Only (Encoder Frozen)")
            self.vae.freeze_encoder()

        # Count parameters
        param_counts = self.vae.count_parameters()
        print(f"   Total Parameters: {param_counts['total']:,}")
        print(f"   Trainable Parameters: {param_counts['trainable']:,}")
        print(f"   Encoder: {param_counts['encoder']:,} ({'trainable' if task_id == 0 else 'frozen'})")
        print(f"   Decoder: {param_counts['decoder']:,} (trainable)")

        # Setup optimizer (only for trainable params)
        optimizer = torch.optim.AdamW(
            self.vae.get_trainable_params(),
            lr=lr,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=lr * 0.01
        )

        # Training loop
        self.vae.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                images = batch[0].to(self.device)

                # Extract features (frozen extractor)
                with torch.no_grad():
                    patch_embeddings = self.extractor(images)  # (B, H, W, D)

                # Flatten spatial dimensions
                B, H, W, D = patch_embeddings.shape
                x = patch_embeddings.reshape(B * H * W, D)

                # Forward pass
                x_recon, mu, log_var = self.vae(x)

                # Compute loss
                losses = self.vae.compute_loss(x, x_recon, mu, log_var, self.beta)

                # Backward pass
                optimizer.zero_grad()
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.vae.get_trainable_params(), max_norm=1.0)
                optimizer.step()

                # Accumulate
                epoch_loss += losses['total'].item()
                epoch_recon += losses['recon'].item()
                epoch_kl += losses['kl'].item()
                n_batches += 1

                # Logging
                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = epoch_loss / n_batches
                    avg_recon = epoch_recon / n_batches
                    avg_kl = epoch_kl / n_batches
                    print(f"   Epoch [{epoch+1:3d}/{num_epochs}] "
                          f"Batch [{batch_idx+1:3d}/{len(train_loader)}] "
                          f"Loss: {losses['total'].item():.4f} "
                          f"(Recon: {losses['recon'].item():.4f}, KL: {losses['kl'].item():.4f})")

            scheduler.step()

            # Record history
            avg_loss = epoch_loss / max(n_batches, 1)
            avg_recon = epoch_recon / max(n_batches, 1)
            avg_kl = epoch_kl / max(n_batches, 1)

            self.history['task_id'].append(task_id)
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(avg_loss)
            self.history['recon_loss'].append(avg_recon)
            self.history['kl_loss'].append(avg_kl)

            # Epoch summary
            print(f"   Epoch [{epoch+1:3d}/{num_epochs}] "
                  f"Avg Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})")

        print(f"\n   Task {task_id} training completed!")

    def evaluate(self, test_loader: DataLoader, class_name: str,
                 return_scores: bool = False) -> Dict:
        """
        Evaluate VAE on test data.

        Args:
            test_loader: DataLoader for test data
            class_name: Class name being evaluated
            return_scores: If True, return raw scores

        Returns:
            Dict with evaluation metrics
        """
        self.vae.eval()

        all_scores = []
        all_labels = []
        all_masks = []

        with torch.no_grad():
            for batch in test_loader:
                images, labels, masks, _, _ = batch
                images = images.to(self.device)

                # Extract features
                patch_embeddings = self.extractor(images)  # (B, H, W, D)

                # Compute anomaly scores
                scores = self.vae.compute_anomaly_score(patch_embeddings)  # (B, H, W)

                all_scores.append(scores.cpu())
                all_labels.extend(labels.numpy())
                all_masks.append(masks.cpu())

        # Concatenate
        all_scores = torch.cat(all_scores, dim=0).numpy()  # (N, H, W)
        all_labels = np.array(all_labels, dtype=bool)
        all_masks = torch.cat(all_masks, dim=0).squeeze(1).numpy()  # (N, msk_H, msk_W)

        # Resize scores to mask size for pixel-level evaluation
        from scipy.ndimage import zoom
        H_score, W_score = all_scores.shape[1], all_scores.shape[2]
        H_mask, W_mask = all_masks.shape[1], all_masks.shape[2]

        if H_score != H_mask or W_score != W_mask:
            # Resize each score map
            resized_scores = np.zeros((all_scores.shape[0], H_mask, W_mask))
            for i in range(all_scores.shape[0]):
                resized_scores[i] = zoom(all_scores[i],
                                         (H_mask / H_score, W_mask / W_score),
                                         order=1)
            all_scores = resized_scores

        # Image-level scores (max of patch scores)
        image_scores = all_scores.max(axis=(1, 2))

        # Compute metrics
        img_auc = roc_auc_score(all_labels, image_scores)

        # Pixel-level (only for anomalous images)
        pixel_scores_flat = all_scores.flatten()
        pixel_labels_flat = all_masks.flatten().astype(bool)

        pixel_auc = roc_auc_score(pixel_labels_flat, pixel_scores_flat)
        pixel_ap = average_precision_score(pixel_labels_flat, pixel_scores_flat)

        results = {
            'class_name': class_name,
            'img_auc': img_auc,
            'pixel_auc': pixel_auc,
            'pixel_ap': pixel_ap,
            'n_samples': len(all_labels),
            'n_anomalies': all_labels.sum()
        }

        if return_scores:
            results['scores'] = all_scores
            results['labels'] = all_labels

        return results

    def evaluate_all_tasks(self, data_path: str, args) -> Dict:
        """
        Evaluate on all learned tasks.

        Args:
            data_path: Path to dataset
            args: Configuration arguments

        Returns:
            Dict with per-task and overall metrics
        """
        print(f"\n{'='*70}")
        print("Evaluating All Tasks")
        print(f"{'='*70}")

        all_results = {
            'classes': [],
            'task_ids': [],
            'img_aucs': [],
            'pixel_aucs': [],
            'pixel_aps': []
        }

        for task_id, task_classes in self.task_classes.items():
            print(f"\nTask {task_id}: {task_classes}")

            for class_name in task_classes:
                # Create test dataset
                test_dataset = MVTEC(
                    root=data_path,
                    class_name=class_name,
                    train=False,
                    img_size=args.img_size,
                    crp_size=args.img_size,
                    msk_size=args.msk_size
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=8, shuffle=False,
                    num_workers=4, pin_memory=True
                )

                # Evaluate
                results = self.evaluate(test_loader, class_name)

                all_results['classes'].append(class_name)
                all_results['task_ids'].append(task_id)
                all_results['img_aucs'].append(results['img_auc'])
                all_results['pixel_aucs'].append(results['pixel_auc'])
                all_results['pixel_aps'].append(results['pixel_ap'])

                print(f"   {class_name}: I-AUC={results['img_auc']:.4f}, "
                      f"P-AUC={results['pixel_auc']:.4f}, P-AP={results['pixel_ap']:.4f}")

        # Compute averages
        all_results['mean_img_auc'] = np.mean(all_results['img_aucs'])
        all_results['mean_pixel_auc'] = np.mean(all_results['pixel_aucs'])
        all_results['mean_pixel_ap'] = np.mean(all_results['pixel_aps'])

        # Per-task averages
        all_results['task_avg_img_aucs'] = {}
        all_results['task_avg_pixel_aucs'] = {}

        for task_id in self.task_classes.keys():
            task_indices = [i for i, t in enumerate(all_results['task_ids']) if t == task_id]
            all_results['task_avg_img_aucs'][task_id] = np.mean(
                [all_results['img_aucs'][i] for i in task_indices]
            )
            all_results['task_avg_pixel_aucs'][task_id] = np.mean(
                [all_results['pixel_aucs'][i] for i in task_indices]
            )

        # Summary
        print(f"\n{'-'*70}")
        print("Summary:")
        for task_id in self.task_classes.keys():
            print(f"   Task {task_id}: I-AUC={all_results['task_avg_img_aucs'][task_id]:.4f}, "
                  f"P-AUC={all_results['task_avg_pixel_aucs'][task_id]:.4f}")
        print(f"\n   Overall: I-AUC={all_results['mean_img_auc']:.4f}, "
              f"P-AUC={all_results['mean_pixel_auc']:.4f}, "
              f"P-AP={all_results['mean_pixel_ap']:.4f}")
        print(f"{'='*70}")

        return all_results

    def compute_forgetting(self, initial_results: Dict, final_results: Dict) -> Dict:
        """
        Compute forgetting measure for continual learning.

        Forgetting = initial_performance - final_performance

        Args:
            initial_results: Results right after training each task
            final_results: Results after all tasks are trained

        Returns:
            Dict with forgetting measures
        """
        forgetting = {}

        for task_id in initial_results.keys():
            initial_auc = initial_results[task_id]['mean_img_auc']

            # Find final AUC for this task
            task_classes = self.task_classes[task_id]
            final_aucs = []
            for i, cls in enumerate(final_results['classes']):
                if cls in task_classes:
                    final_aucs.append(final_results['img_aucs'][i])

            if final_aucs:
                final_auc = np.mean(final_aucs)
                forgetting[task_id] = {
                    'initial_auc': initial_auc,
                    'final_auc': final_auc,
                    'forgetting': initial_auc - final_auc,
                    'classes': task_classes
                }

        # Average forgetting (excluding last task)
        if len(forgetting) > 1:
            task_ids = sorted(forgetting.keys())[:-1]  # Exclude last task
            avg_forgetting = np.mean([forgetting[t]['forgetting'] for t in task_ids])
            forgetting['average'] = avg_forgetting

        return forgetting

    def save_results(self, results: Dict, save_dir: str, prefix: str = 'vae'):
        """Save results to files."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        json_path = save_path / f'{prefix}_results.json'
        with open(json_path, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable = {}
            for k, v in results.items():
                if isinstance(v, np.ndarray):
                    serializable[k] = v.tolist()
                elif isinstance(v, (np.float32, np.float64)):
                    serializable[k] = float(v)
                elif isinstance(v, dict):
                    serializable[k] = {
                        str(kk): float(vv) if isinstance(vv, (np.float32, np.float64)) else vv
                        for kk, vv in v.items()
                    }
                else:
                    serializable[k] = v
            json.dump(serializable, f, indent=2)

        print(f"Results saved to: {json_path}")

        # Save training history
        history_path = save_path / f'{prefix}_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"Training history saved to: {history_path}")


# =============================================================================
# Dataset Utilities
# =============================================================================

class TaskDataset(Dataset):
    """Simple dataset wrapper for continual learning tasks."""

    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_task_dataset(data_path: str, task_classes: List[str],
                        img_size: int, msk_size: int, train: bool = True) -> TaskDataset:
    """Create dataset for specific task classes."""
    all_data = []

    for class_name in task_classes:
        dataset = MVTEC(
            root=data_path,
            class_name=class_name,
            train=train,
            img_size=img_size,
            crp_size=img_size,
            msk_size=msk_size
        )

        for i in range(len(dataset)):
            all_data.append(dataset[i])

    return TaskDataset(all_data)


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main entry point for VAE baseline experiment."""
    parser = argparse.ArgumentParser(
        description='VAE Baseline for Continual Anomaly Detection'
    )

    # Task configuration
    parser.add_argument('--task_classes', type=str, nargs='+',
                        default=['leather', 'grid', 'transistor'],
                        help='Classes to learn sequentially (one class per task)')

    # Model configuration
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent space dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer dimension')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='KL divergence weight (beta-VAE)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')

    # Training configuration
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Data configuration
    parser.add_argument('--data_path', type=str, default='/Data/MVTecAD',
                        help='Path to MVTec AD dataset')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--msk_size', type=int, default=256,
                        help='Mask size for evaluation')

    # Feature extractor configuration
    parser.add_argument('--backbone_name', type=str, default='wide_resnet50_2',
                        help='Backbone model name (must match MoLE-Flow)')
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='Embedding dimension from backbone')

    # Output configuration
    parser.add_argument('--log_dir', type=str,
                        default='/Volume/MoLeFlow/logs/PilotExperiment/VAE_baseline',
                        help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (auto-generated if not specified)')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Generate experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_str = "_".join(args.task_classes[:3])
        args.experiment_name = f"VAE_{task_str}_{timestamp}"

    # Setup logging directory
    log_dir = Path(args.log_dir) / args.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("VAE Baseline for Continual Anomaly Detection")
    print("='*70")
    print(f"   Experiment: {args.experiment_name}")
    print(f"   Task Classes: {args.task_classes}")
    print(f"   Latent Dim: {args.latent_dim}")
    print(f"   Hidden Dim: {args.hidden_dim}")
    print(f"   Beta: {args.beta}")
    print(f"   Epochs/Task: {args.num_epochs}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Backbone: {args.backbone_name}")
    print(f"   Embed Dim: {args.embed_dim}")
    print(f"{'='*70}")

    # Initialize feature extractor (frozen CNN backbone)
    print("\nInitializing feature extractor...")
    feature_extractor = CNNPatchCoreExtractor(
        backbone_name=args.backbone_name,
        input_shape=(3, args.img_size, args.img_size),
        target_embed_dimension=args.embed_dim,
        device=device
    )
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False
    print(f"   Backbone: {args.backbone_name} (frozen)")
    print(f"   Output dimension: {args.embed_dim}")

    # Initialize VAE model
    print("\nInitializing VAE model...")
    vae_model = VAEModel(
        input_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout
    )
    print(f"   Input/Output dim: {args.embed_dim}")
    print(f"   Latent dim: {args.latent_dim}")
    print(f"   Total parameters: {sum(p.numel() for p in vae_model.parameters()):,}")

    # Initialize trainer
    trainer = VAEContinualTrainer(
        vae_model=vae_model,
        feature_extractor=feature_extractor,
        device=device,
        beta=args.beta
    )

    # Store results after each task for forgetting computation
    task_results = {}

    # Continual learning loop
    for task_id, class_name in enumerate(args.task_classes):
        task_classes = [class_name]

        print(f"\n{'#'*70}")
        print(f"# Task {task_id}: {task_classes}")
        print(f"{'#'*70}")

        # Create training dataset
        train_dataset = create_task_dataset(
            args.data_path, task_classes,
            args.img_size, args.msk_size, train=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True
        )

        print(f"   Training samples: {len(train_dataset)}")

        # Train on this task
        trainer.train_task(
            task_id=task_id,
            task_classes=task_classes,
            train_loader=train_loader,
            num_epochs=args.num_epochs,
            lr=args.lr,
            log_interval=10
        )

        # Evaluate all tasks seen so far
        print(f"\nEvaluation after Task {task_id}")
        results = trainer.evaluate_all_tasks(args.data_path, args)

        # Store for forgetting computation
        task_results[task_id] = {
            'mean_img_auc': results['mean_img_auc'],
            'mean_pixel_auc': results['mean_pixel_auc'],
            'classes': list(trainer.task_classes[task_id])
        }

    # Final evaluation
    print(f"\n{'='*70}")
    print("Final Evaluation (All Tasks)")
    print(f"{'='*70}")
    final_results = trainer.evaluate_all_tasks(args.data_path, args)

    # Compute forgetting
    print(f"\n{'='*70}")
    print("Forgetting Analysis")
    print(f"{'='*70}")
    forgetting = trainer.compute_forgetting(task_results, final_results)

    for task_id, metrics in forgetting.items():
        if task_id == 'average':
            print(f"\n   Average Forgetting: {metrics:.4f}")
        else:
            print(f"   Task {task_id} ({metrics['classes']}): "
                  f"Initial={metrics['initial_auc']:.4f}, "
                  f"Final={metrics['final_auc']:.4f}, "
                  f"Forgetting={metrics['forgetting']:.4f}")

    # Save all results
    all_results = {
        'experiment_name': args.experiment_name,
        'config': vars(args),
        'final_results': final_results,
        'task_results': task_results,
        'forgetting': forgetting,
        'timestamp': datetime.now().isoformat()
    }

    trainer.save_results(all_results, str(log_dir), prefix='vae')

    # Save configuration
    config_path = log_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nConfiguration saved to: {config_path}")

    # Print final summary
    print(f"\n{'='*70}")
    print("VAE Baseline Experiment Completed")
    print(f"{'='*70}")
    print(f"   Final Mean Image AUC: {final_results['mean_img_auc']:.4f}")
    print(f"   Final Mean Pixel AUC: {final_results['mean_pixel_auc']:.4f}")
    if 'average' in forgetting:
        print(f"   Average Forgetting: {forgetting['average']:.4f}")
    print(f"   Results saved to: {log_dir}")
    print(f"{'='*70}")

    return trainer, final_results, forgetting


if __name__ == '__main__':
    main()
