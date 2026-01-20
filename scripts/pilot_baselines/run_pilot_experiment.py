#!/usr/bin/env python3
"""
Unified Pilot Experiment Runner for Continual Anomaly Detection Baselines.

This script runs a comparative pilot experiment between different AD architectures:
- NF (MoLE-Flow reference)
- AE (Autoencoder)
- VAE (Variational Autoencoder)
- TS (Teacher-Student)
- Memory (Memory Bank / PatchCore-style)

Goal: Demonstrate that "Base frozen + Task-specific trainable" decomposition
works effectively for Normalizing Flows but fails for other architectures.

Usage:
    python run_pilot_experiment.py --models ae vae ts memory nf
    python run_pilot_experiment.py --models ae --num_epochs 50
    python run_pilot_experiment.py --task_classes leather grid transistor carpet

Author: MoLE-Flow Team
"""

import argparse
import sys
import os
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pilot_baselines.shared_utils import (
    ExperimentConfig,
    FeatureExtractorWrapper,
    ExperimentLogger,
    BaselineModel,
    create_task_dataloaders,
    evaluate_all_tasks,
    set_seed,
    generate_comparison_table,
    generate_forgetting_report,
)


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_CONFIG = {
    'task_classes': ['leather', 'grid', 'transistor'],
    'num_epochs': 30,
    'batch_size': 16,
    'backbone': 'wide_resnet50_2',
    'data_path': '/Data/MVTecAD',
    'log_base_dir': '/Volume/MoLeFlow/logs/PilotExperiment',
    'seeds': [42],
    'lr': 1e-4,
    'img_size': 256,
    'msk_size': 256,
    'target_embed_dim': 1024,
}

AVAILABLE_MODELS = ['ae', 'vae', 'ts', 'memory', 'nf']


# =============================================================================
# Baseline Model Implementations
# =============================================================================

class AutoencoderBaseline(BaselineModel):
    """
    Autoencoder baseline for anomaly detection.

    Architecture:
    - Base encoder/decoder shared across tasks (frozen after Task 0)
    - Task-specific adapter layers (trainable per task)

    Anomaly score: Reconstruction error (MSE)
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256, device: str = 'cuda'):
        super().__init__(feature_dim, device)
        self.hidden_dim = hidden_dim

        # Shared encoder (will be frozen after Task 0)
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        ).to(device)

        # Shared decoder (will be frozen after Task 0)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, feature_dim),
        ).to(device)

        # Task-specific adapters (trainable per task)
        self.task_adapters: Dict[int, nn.Module] = {}

    def add_task(self, task_id: int):
        """Add task-specific adapter."""
        # Simple FiLM-style adapter
        adapter = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        ).to(self.device)
        self.task_adapters[task_id] = adapter
        self.current_task = task_id

        # Freeze base encoder/decoder after Task 0
        if task_id > 0:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through AE.

        Args:
            features: (B, H, W, D) patch features

        Returns:
            reconstructed: (B, H, W, D) reconstructed features
        """
        B, H, W, D = features.shape
        x = features.reshape(B * H * W, D)

        # Encode
        z = self.encoder(x)

        # Apply task adapter if available
        if self.current_task in self.task_adapters:
            z = z + self.task_adapters[self.current_task](z)

        # Decode
        x_rec = self.decoder(z)

        return x_rec.reshape(B, H, W, D)

    def compute_anomaly_score(self, features: torch.Tensor):
        """Compute reconstruction-based anomaly score."""
        reconstructed = self.forward(features)

        # Per-patch MSE
        patch_scores = ((features - reconstructed) ** 2).mean(dim=-1)  # (B, H, W)

        # Image score: max patch score
        image_scores = patch_scores.reshape(patch_scores.shape[0], -1).max(dim=1)[0]

        return image_scores, patch_scores

    def get_trainable_params(self, task_id: int = None):
        """Get trainable parameters for current task."""
        if task_id is None:
            task_id = self.current_task

        params = []
        if task_id == 0:
            # Train everything for Task 0
            params.extend(self.encoder.parameters())
            params.extend(self.decoder.parameters())
        if task_id in self.task_adapters:
            params.extend(self.task_adapters[task_id].parameters())

        return params

    def train_task(
        self,
        task_id: int,
        train_loader,
        feature_extractor: FeatureExtractorWrapper,
        num_epochs: int,
        lr: float,
        logger: ExperimentLogger = None
    ) -> List[float]:
        """Train AE on a task."""
        self.add_task(task_id)
        self.train()

        params = self.get_trainable_params(task_id)
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            n_batches = 0

            for batch in train_loader:
                images = batch[0].to(self.device)

                # Get features (frozen backbone)
                features = feature_extractor(images)

                # Forward and compute loss
                reconstructed = self.forward(features)
                loss = F.mse_loss(reconstructed, features)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

            if logger:
                logger.log_training_step(task_id, epoch, avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  [AE] Task {task_id} Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

        return losses


class VAEBaseline(BaselineModel):
    """
    Variational Autoencoder baseline for anomaly detection.

    Similar to AE but with:
    - Latent space regularization (KL divergence)
    - Task-specific prior adaptation

    Anomaly score: Reconstruction error + KL divergence
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256, latent_dim: int = 64, device: str = 'cuda'):
        super().__init__(feature_dim, device)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder to mean and log-variance
        self.encoder_shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        ).to(device)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim).to(device)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim).to(device)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, feature_dim),
        ).to(device)

        # Task-specific prior adapters
        self.task_adapters: Dict[int, nn.Module] = {}

    def add_task(self, task_id: int):
        """Add task-specific adapter."""
        # Prior adapter: shifts the prior mean
        adapter = nn.Linear(self.latent_dim, self.latent_dim).to(self.device)
        nn.init.zeros_(adapter.weight)
        nn.init.zeros_(adapter.bias)
        self.task_adapters[task_id] = adapter
        self.current_task = task_id

        # Freeze shared components after Task 0
        if task_id > 0:
            for param in self.encoder_shared.parameters():
                param.requires_grad = False
            for param in self.fc_mu.parameters():
                param.requires_grad = False
            for param in self.fc_logvar.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, features: torch.Tensor, return_kl: bool = False):
        """Forward pass through VAE."""
        B, H, W, D = features.shape
        x = features.reshape(B * H * W, D)

        # Encode
        h = self.encoder_shared(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Apply task adapter to prior
        if self.current_task in self.task_adapters:
            mu = mu + self.task_adapters[self.current_task](mu)

        # Sample
        z = self.reparameterize(mu, logvar)

        # Decode
        x_rec = self.decoder(z)
        x_rec = x_rec.reshape(B, H, W, D)

        if return_kl:
            # KL divergence
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            kl = kl.reshape(B, H, W)
            return x_rec, kl

        return x_rec

    def compute_anomaly_score(self, features: torch.Tensor):
        """Compute VAE anomaly score = reconstruction + KL."""
        reconstructed, kl = self.forward(features, return_kl=True)

        # Reconstruction error per patch
        recon_error = ((features - reconstructed) ** 2).mean(dim=-1)  # (B, H, W)

        # Combined score
        patch_scores = recon_error + 0.1 * kl  # Weight KL lower

        # Image score
        image_scores = patch_scores.reshape(patch_scores.shape[0], -1).max(dim=1)[0]

        return image_scores, patch_scores

    def get_trainable_params(self, task_id: int = None):
        """Get trainable parameters."""
        if task_id is None:
            task_id = self.current_task

        params = []
        if task_id == 0:
            params.extend(self.encoder_shared.parameters())
            params.extend(self.fc_mu.parameters())
            params.extend(self.fc_logvar.parameters())
            params.extend(self.decoder.parameters())
        if task_id in self.task_adapters:
            params.extend(self.task_adapters[task_id].parameters())

        return params

    def train_task(
        self,
        task_id: int,
        train_loader,
        feature_extractor: FeatureExtractorWrapper,
        num_epochs: int,
        lr: float,
        logger: ExperimentLogger = None
    ) -> List[float]:
        """Train VAE on a task."""
        self.add_task(task_id)
        self.train()

        params = self.get_trainable_params(task_id)
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            n_batches = 0

            for batch in train_loader:
                images = batch[0].to(self.device)
                features = feature_extractor(images)

                # Forward with KL
                reconstructed, kl = self.forward(features, return_kl=True)

                # ELBO loss
                recon_loss = F.mse_loss(reconstructed, features)
                kl_loss = kl.mean()
                loss = recon_loss + 0.001 * kl_loss  # Beta-VAE style

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

            if logger:
                logger.log_training_step(task_id, epoch, avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  [VAE] Task {task_id} Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

        return losses


class TeacherStudentBaseline(BaselineModel):
    """
    Teacher-Student baseline for anomaly detection.

    Architecture:
    - Teacher network: Frozen after Task 0
    - Student network: Trained to match teacher on normal data
    - Task-specific adaptation layers

    Anomaly score: Discrepancy between teacher and student outputs
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256, device: str = 'cuda'):
        super().__init__(feature_dim, device)
        self.hidden_dim = hidden_dim

        # Teacher network (frozen after Task 0)
        self.teacher = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(device)

        # Student network (always trained)
        self.student_base = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        ).to(device)

        # Task-specific student heads
        self.task_heads: Dict[int, nn.Module] = {}

    def add_task(self, task_id: int):
        """Add task-specific student head."""
        head = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
        self.task_heads[task_id] = head
        self.current_task = task_id

        # Freeze teacher and student base after Task 0
        if task_id > 0:
            for param in self.teacher.parameters():
                param.requires_grad = False
            for param in self.student_base.parameters():
                param.requires_grad = False

    def forward(self, features: torch.Tensor):
        """Forward pass returning teacher and student outputs."""
        B, H, W, D = features.shape
        x = features.reshape(B * H * W, D)

        # Teacher output (no grad)
        with torch.no_grad():
            t_out = self.teacher(x)

        # Student output
        s_base = self.student_base(x)
        if self.current_task in self.task_heads:
            s_out = self.task_heads[self.current_task](s_base)
        else:
            s_out = s_base

        return t_out.reshape(B, H, W, -1), s_out.reshape(B, H, W, -1)

    def compute_anomaly_score(self, features: torch.Tensor):
        """Compute discrepancy-based anomaly score."""
        t_out, s_out = self.forward(features)

        # Per-patch discrepancy
        patch_scores = ((t_out - s_out) ** 2).mean(dim=-1)  # (B, H, W)

        # Image score
        image_scores = patch_scores.reshape(patch_scores.shape[0], -1).max(dim=1)[0]

        return image_scores, patch_scores

    def get_trainable_params(self, task_id: int = None):
        """Get trainable parameters."""
        if task_id is None:
            task_id = self.current_task

        params = []
        if task_id == 0:
            params.extend(self.teacher.parameters())
            params.extend(self.student_base.parameters())
        if task_id in self.task_heads:
            params.extend(self.task_heads[task_id].parameters())

        return params

    def train_task(
        self,
        task_id: int,
        train_loader,
        feature_extractor: FeatureExtractorWrapper,
        num_epochs: int,
        lr: float,
        logger: ExperimentLogger = None
    ) -> List[float]:
        """Train Teacher-Student on a task."""
        self.add_task(task_id)
        self.train()

        params = self.get_trainable_params(task_id)
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            n_batches = 0

            for batch in train_loader:
                images = batch[0].to(self.device)
                features = feature_extractor(images)

                # Get teacher and student outputs
                t_out, s_out = self.forward(features)

                # Knowledge distillation loss
                loss = F.mse_loss(s_out, t_out.detach())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

            if logger:
                logger.log_training_step(task_id, epoch, avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  [TS] Task {task_id} Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

        return losses


class MemoryBankBaseline(BaselineModel):
    """
    Memory Bank (PatchCore-style) baseline for anomaly detection.

    Architecture:
    - Memory bank of normal feature vectors per task
    - Nearest neighbor distance for anomaly scoring
    - Task-specific memory banks

    Anomaly score: Distance to nearest normal neighbor

    Note: This is a non-parametric method - no backprop training.
    """

    def __init__(self, feature_dim: int, memory_size: int = 500, device: str = 'cuda'):
        super().__init__(feature_dim, device)
        self.memory_size = memory_size

        # Task-specific memory banks
        self.memory_banks: Dict[int, torch.Tensor] = {}

    def add_task(self, task_id: int):
        """Initialize memory bank for task."""
        self.current_task = task_id
        # Memory will be built during training

    def _subsample_memory(self, features: torch.Tensor) -> torch.Tensor:
        """Subsample features for memory bank."""
        if features.shape[0] <= self.memory_size:
            return features

        # Random subsampling (could use coreset selection)
        indices = torch.randperm(features.shape[0])[:self.memory_size]
        return features[indices]

    def compute_anomaly_score(self, features: torch.Tensor):
        """Compute nearest neighbor distance-based anomaly score."""
        B, H, W, D = features.shape

        if self.current_task not in self.memory_banks:
            # No memory: return zeros
            return torch.zeros(B, device=self.device), torch.zeros(B, H, W, device=self.device)

        memory = self.memory_banks[self.current_task]  # (M, D)

        # Flatten features
        x = features.reshape(B * H * W, D)

        # Compute pairwise distances (batch for memory)
        # x: (N, D), memory: (M, D) -> distances: (N, M)
        distances = torch.cdist(x, memory)  # (N, M)

        # Nearest neighbor distance
        nn_dist, _ = distances.min(dim=1)  # (N,)

        patch_scores = nn_dist.reshape(B, H, W)

        # Image score: max patch distance
        image_scores = patch_scores.reshape(B, -1).max(dim=1)[0]

        return image_scores, patch_scores

    def get_trainable_params(self, task_id: int = None):
        """Memory bank has no trainable params."""
        return []

    def train_task(
        self,
        task_id: int,
        train_loader,
        feature_extractor: FeatureExtractorWrapper,
        num_epochs: int,  # Ignored for memory bank
        lr: float,  # Ignored
        logger: ExperimentLogger = None
    ) -> List[float]:
        """Build memory bank from training data."""
        self.add_task(task_id)
        self.eval()

        all_features = []

        print(f"  [Memory] Building memory bank for Task {task_id}...")

        with torch.no_grad():
            for batch in train_loader:
                images = batch[0].to(self.device)
                features = feature_extractor(images)

                # Flatten to patches
                B, H, W, D = features.shape
                patches = features.reshape(B * H * W, D)
                all_features.append(patches.cpu())

        # Concatenate and subsample
        all_features = torch.cat(all_features, dim=0)
        memory = self._subsample_memory(all_features)
        self.memory_banks[task_id] = memory.to(self.device)

        print(f"  [Memory] Memory bank size: {memory.shape[0]} vectors")

        if logger:
            logger.log_training_step(task_id, 0, 0.0)

        return [0.0]  # No loss for memory bank


class NFReferenceBaseline(BaselineModel):
    """
    Normalizing Flow reference (runs actual MoLE-Flow).

    This calls the actual MoLE-Flow training script as a subprocess
    to ensure we're comparing against the real implementation.
    """

    def __init__(self, feature_dim: int, device: str = 'cuda'):
        super().__init__(feature_dim, device)
        self._results_cache = {}

    def add_task(self, task_id: int):
        """NF reference doesn't need this - it runs as a whole."""
        self.current_task = task_id

    def compute_anomaly_score(self, features: torch.Tensor):
        """NF reference doesn't use this - results come from subprocess."""
        raise NotImplementedError("NF reference runs via subprocess")

    def get_trainable_params(self, task_id: int = None):
        """NF reference doesn't use this."""
        return []

    def train_task(self, *args, **kwargs):
        """NF reference doesn't use this - it runs as a whole."""
        raise NotImplementedError("Use run_nf_reference() instead")


# =============================================================================
# Model Factory
# =============================================================================

def create_model(model_name: str, feature_dim: int, device: str = 'cuda') -> BaselineModel:
    """
    Create a baseline model by name.

    Args:
        model_name: One of 'ae', 'vae', 'ts', 'memory', 'nf'
        feature_dim: Feature dimension from backbone
        device: Device to use

    Returns:
        BaselineModel instance
    """
    model_name = model_name.lower()

    if model_name == 'ae':
        return AutoencoderBaseline(feature_dim, device=device)
    elif model_name == 'vae':
        return VAEBaseline(feature_dim, device=device)
    elif model_name == 'ts':
        return TeacherStudentBaseline(feature_dim, device=device)
    elif model_name == 'memory':
        return MemoryBankBaseline(feature_dim, device=device)
    elif model_name == 'nf':
        return NFReferenceBaseline(feature_dim, device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {AVAILABLE_MODELS}")


# =============================================================================
# NF Reference Runner
# =============================================================================

def run_nf_reference(config: ExperimentConfig, log_dir: Path) -> Dict:
    """
    Run MoLE-Flow as reference using the actual implementation.

    Args:
        config: Experiment configuration
        log_dir: Directory for logs

    Returns:
        Dict with results
    """
    nf_log_dir = log_dir / 'NF_reference'
    nf_log_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    task_classes_str = ' '.join(config.task_classes)
    cmd = [
        'python', str(PROJECT_ROOT / 'run_moleflow.py'),
        '--dataset', 'mvtec',
        '--data_path', config.data_path,
        '--task_classes', *config.task_classes,
        '--backbone_name', config.backbone,
        '--num_epochs', str(config.num_epochs),
        '--lr', str(config.lr),
        '--batch_size', str(config.batch_size),
        '--img_size', str(config.img_size),
        '--log_dir', str(nf_log_dir),
        '--experiment_name', 'NF_pilot',
        '--seed', str(config.seeds[0]),
        # Use default MoLE-Flow settings
        '--num_coupling_layers', '6',
        '--dia_n_blocks', '2',
        '--lora_rank', '64',
        '--use_whitening_adapter',
        '--use_tail_aware_loss',
        '--tail_weight', '0.7',
    ]

    print(f"\n{'='*60}")
    print("Running NF Reference (MoLE-Flow)")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=3600  # 1 hour timeout
        )

        if result.returncode != 0:
            print(f"NF reference failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return {'error': result.stderr}

        print("NF reference completed successfully")

        # Parse results from log directory
        # Look for final_results.csv or similar
        results = {'model_name': 'NF', 'task_metrics': {}}

        # Try to load results
        results_file = nf_log_dir / 'NF_pilot' / 'final_results.csv'
        if results_file.exists():
            import csv
            with open(results_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    task_id = row.get('task_id', 0)
                    results['task_metrics'][task_id] = {
                        'image_auc': float(row.get('image_auc', 0)),
                        'pixel_auc': float(row.get('pixel_auc', 0)),
                        'pixel_ap': float(row.get('pixel_ap', 0)),
                    }

        return results

    except subprocess.TimeoutExpired:
        print("NF reference timed out")
        return {'error': 'timeout'}
    except Exception as e:
        print(f"NF reference failed with exception: {e}")
        traceback.print_exc()
        return {'error': str(e)}


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_single_model_experiment(
    model_name: str,
    config: ExperimentConfig,
    feature_extractor: FeatureExtractorWrapper,
    log_dir: Path
) -> Dict:
    """
    Run experiment for a single baseline model.

    Args:
        model_name: Model name ('ae', 'vae', 'ts', 'memory')
        config: Experiment configuration
        feature_extractor: Shared feature extractor
        log_dir: Base log directory

    Returns:
        Dict with results
    """
    print(f"\n{'='*60}")
    print(f"Running {model_name.upper()} Baseline")
    print(f"{'='*60}")

    # Create logger
    logger = ExperimentLogger(log_dir, model_name.upper(), config)

    # Create model
    model = create_model(model_name, config.target_embed_dim, config.device)

    # Track initial AUCs for forgetting analysis
    initial_aucs = {}

    # Train on each task sequentially
    for task_id in range(len(config.task_classes)):
        task_class = config.task_classes[task_id]
        print(f"\n--- Task {task_id}: {task_class} ---")

        # Get data loaders
        train_loader, test_loader = create_task_dataloaders(config, task_id)
        print(f"  Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

        # Train on this task
        start_time = time.time()
        losses = model.train_task(
            task_id=task_id,
            train_loader=train_loader,
            feature_extractor=feature_extractor,
            num_epochs=config.num_epochs,
            lr=config.lr,
            logger=logger
        )
        train_time = time.time() - start_time
        logger.log_training_time(task_id, train_time)

        print(f"  Training time: {train_time:.1f}s")

        # Evaluate on all tasks seen so far
        print(f"  Evaluating on tasks 0-{task_id}...")

        results = evaluate_all_tasks(
            model=model,
            feature_extractor=feature_extractor,
            config=config,
            current_task=task_id
        )

        # Log metrics
        logger.log_task_metrics(task_id, results, after_task=task_id)

        # Print current performance
        for tid, metrics in sorted(results.items()):
            print(f"    Task {tid} ({metrics.task_class}): I-AUC={metrics.image_auc:.4f}")

            # Track forgetting
            if tid == task_id:
                # Just trained this task - record initial AUC
                initial_aucs[tid] = metrics.image_auc
            elif tid in initial_aucs:
                # Previously trained task - compute forgetting
                logger.log_forgetting(
                    tid,
                    initial_aucs[tid],
                    metrics.image_auc,
                    after_task=task_id
                )

    # Save results
    logger.save_results()
    logger.print_summary()

    return logger.results


def run_experiment(
    models: List[str],
    config: ExperimentConfig
) -> Dict[str, Dict]:
    """
    Run the full pilot experiment.

    Args:
        models: List of model names to run
        config: Experiment configuration

    Returns:
        Dict mapping model names to their results
    """
    print("\n" + "="*70)
    print("PILOT EXPERIMENT: Continual Anomaly Detection Baselines")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Tasks: {config.task_classes}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Backbone: {config.backbone}")
    print(f"  Models: {models}")
    print(f"  Log dir: {config.log_base_dir}")

    # Create log directory
    log_dir = Path(config.log_base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.experiment_name:
        log_dir = log_dir / f"{config.experiment_name}_{timestamp}"
    else:
        log_dir = log_dir / f"pilot_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    with open(log_dir / 'experiment_config.json', 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    # Set seed
    set_seed(config.seeds[0])

    # Create shared feature extractor
    print("\nInitializing feature extractor...")
    feature_extractor = FeatureExtractorWrapper(
        backbone_name=config.backbone,
        input_shape=(3, config.img_size, config.img_size),
        target_embed_dim=config.target_embed_dim,
        device=config.device
    )

    # Run each model
    all_results = {}

    for model_name in models:
        if model_name.lower() == 'nf':
            # Run NF reference separately
            results = run_nf_reference(config, log_dir)
            all_results['NF'] = results
        else:
            results = run_single_model_experiment(
                model_name=model_name,
                config=config,
                feature_extractor=feature_extractor,
                log_dir=log_dir
            )
            all_results[model_name.upper()] = results

    # Generate summary report
    generate_summary_report(log_dir, list(all_results.keys()), config)

    return all_results


def generate_summary_report(log_dir: Path, model_names: List[str], config: ExperimentConfig):
    """Generate summary report with comparison table and visualizations."""
    summary_dir = log_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("Generating Summary Report")
    print("="*60)

    # Generate comparison table
    table = generate_comparison_table(str(log_dir), model_names)
    forgetting = generate_forgetting_report(str(log_dir), model_names)

    # Write markdown report
    report_path = summary_dir / 'pilot_report.md'
    with open(report_path, 'w') as f:
        f.write("# Pilot Experiment Report\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- Tasks: {config.task_classes}\n")
        f.write(f"- Epochs: {config.num_epochs}\n")
        f.write(f"- Backbone: {config.backbone}\n")
        f.write(f"- Models: {model_names}\n\n")
        f.write(table)
        f.write("\n")
        f.write(forgetting)

    print(f"Report saved to: {report_path}")

    # Generate comparison plots
    try:
        generate_comparison_plots(log_dir, model_names, summary_dir)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    # Save comparison table as CSV
    csv_path = summary_dir / 'comparison_table.csv'
    with open(csv_path, 'w') as f:
        f.write(table.replace('|', ',').replace('-', ''))

    print(f"CSV saved to: {csv_path}")


def generate_comparison_plots(log_dir: Path, model_names: List[str], summary_dir: Path):
    """Generate comparison visualizations."""

    # Collect results
    all_results = {}
    for name in model_names:
        result_path = log_dir / f"{name}_baseline" / 'results.json'
        if result_path.exists():
            with open(result_path) as f:
                all_results[name] = json.load(f)

    if not all_results:
        print("No results found for plotting")
        return

    # Plot 1: Final AUC comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get final metrics for each model
    model_aucs = {}
    for name, results in all_results.items():
        if 'task_metrics' not in results:
            continue
        final_key = max(results['task_metrics'].keys())
        metrics = results['task_metrics'][final_key]
        model_aucs[name] = [metrics[str(i)]['image_auc'] for i in range(len(metrics))]

    if model_aucs:
        # Bar chart of final AUCs
        x = np.arange(len(model_aucs))
        width = 0.2
        task_names = [f"Task {i}" for i in range(3)]

        for i, task in enumerate(task_names):
            values = [model_aucs[m][i] if i < len(model_aucs[m]) else 0 for m in model_aucs]
            axes[0].bar(x + i*width, values, width, label=task)

        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('Image AUC')
        axes[0].set_title('Final Task Performance')
        axes[0].set_xticks(x + width)
        axes[0].set_xticklabels(list(model_aucs.keys()))
        axes[0].legend()
        axes[0].set_ylim(0.5, 1.0)

    # Plot 2: Forgetting analysis
    forgetting_data = {}
    for name, results in all_results.items():
        if 'forgetting' not in results:
            continue
        fm_values = [f.get('forgetting_measure', 0) for f in results['forgetting'].values()]
        if fm_values:
            forgetting_data[name] = np.mean(fm_values)

    if forgetting_data:
        names = list(forgetting_data.keys())
        values = list(forgetting_data.values())
        colors = ['green' if v <= 0 else 'red' for v in values]

        axes[1].bar(names, values, color=colors)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Average Forgetting Measure')
        axes[1].set_title('Forgetting Analysis (lower is better)')

    plt.tight_layout()
    plt.savefig(summary_dir / 'comparison_overview.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to: {summary_dir / 'comparison_overview.png'}")


# =============================================================================
# CLI Argument Parser
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run pilot experiment comparing AD architectures for continual learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all models
  python run_pilot_experiment.py --models ae vae ts memory nf

  # Run specific models
  python run_pilot_experiment.py --models ae vae

  # Custom task sequence
  python run_pilot_experiment.py --task_classes leather grid transistor carpet

  # More epochs
  python run_pilot_experiment.py --models ae --num_epochs 50
        """
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=['ae', 'vae', 'ts', 'memory'],
        choices=AVAILABLE_MODELS,
        help=f'Models to run (default: ae vae ts memory). Available: {AVAILABLE_MODELS}'
    )

    parser.add_argument(
        '--task_classes',
        nargs='+',
        default=DEFAULT_CONFIG['task_classes'],
        help='Task classes in order (default: leather grid transistor)'
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=DEFAULT_CONFIG['num_epochs'],
        help=f'Number of training epochs (default: {DEFAULT_CONFIG["num_epochs"]})'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=DEFAULT_CONFIG['batch_size'],
        help=f'Batch size (default: {DEFAULT_CONFIG["batch_size"]})'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=DEFAULT_CONFIG['lr'],
        help=f'Learning rate (default: {DEFAULT_CONFIG["lr"]})'
    )

    parser.add_argument(
        '--backbone',
        type=str,
        default=DEFAULT_CONFIG['backbone'],
        help=f'Feature backbone (default: {DEFAULT_CONFIG["backbone"]})'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default=DEFAULT_CONFIG['data_path'],
        help=f'Path to MVTec AD dataset (default: {DEFAULT_CONFIG["data_path"]})'
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default=DEFAULT_CONFIG['log_base_dir'],
        help=f'Base log directory (default: {DEFAULT_CONFIG["log_base_dir"]})'
    )

    parser.add_argument(
        '--experiment_name',
        type=str,
        default='',
        help='Optional experiment name for log directory'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--img_size',
        type=int,
        default=DEFAULT_CONFIG['img_size'],
        help=f'Image size (default: {DEFAULT_CONFIG["img_size"]})'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available)'
    )

    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    args = parse_args()

    # Build config from args
    config = ExperimentConfig(
        task_classes=args.task_classes,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone=args.backbone,
        data_path=args.data_path,
        log_base_dir=args.log_dir,
        experiment_name=args.experiment_name,
        seeds=[args.seed],
        img_size=args.img_size,
        msk_size=args.img_size,
        target_embed_dim=1024,
        device=args.device,
    )

    # Run experiment
    results = run_experiment(args.models, config)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {config.log_base_dir}")


if __name__ == '__main__':
    main()
