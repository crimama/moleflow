#!/usr/bin/env python3
"""
Feature-level vs Coupling-level Adaptation Comparison Experiment.

This script runs a controlled experiment comparing:
1. Coupling-level (DeCoFlow): LoRA adapters within NF coupling subnets
2. Feature-level: Adapters applied before NF input

Both configurations use matched parameter counts for fair comparison.
"""

import argparse
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from moleflow import (
    MoLESpatialAwareNF,
    PositionalEmbeddingGenerator,
    create_feature_extractor,
    init_seeds,
)
from moleflow.data.datasets import create_task_dataset
from moleflow.config import AblationConfig
from moleflow.models.adapters import FeatureLevelPromptAdapter, FeatureLevelMLPAdapter


class FeatureLevelTrainer:
    """
    Custom trainer for Feature-level adaptation baseline.

    Key differences from Coupling-level (DeCoFlow):
    - NF is completely frozen after Task 0
    - Feature-level adapter is trained instead of LoRA
    - Adapter is applied before NF forward pass
    """

    def __init__(self, nf_model, feature_extractor, pos_embedding_gen,
                 embed_dim, device, log_dir, adapter_type='prompt', target_params=1_930_000):
        self.nf = nf_model
        self.extractor = feature_extractor
        self.pos_embedding_gen = pos_embedding_gen
        self.embed_dim = embed_dim
        self.device = device
        self.log_dir = log_dir
        self.adapter_type = adapter_type
        self.target_params = target_params

        # Task-specific adapters
        self.task_adapters = nn.ModuleDict()
        self.current_task_id = -1

        # Feature statistics for routing
        self.task_prototypes = {}

    def _create_adapter(self, task_id: int):
        """Create a feature-level adapter for new task."""
        if self.adapter_type == 'prompt':
            adapter = FeatureLevelPromptAdapter(
                channels=self.embed_dim,
                spatial_size=14,  # For 224x224 input
                target_params=self.target_params
            )
        else:
            adapter = FeatureLevelMLPAdapter(
                channels=self.embed_dim,
                target_params=self.target_params
            )
        return adapter.to(self.device)

    def train_task(self, task_id: int, train_loader: DataLoader,
                   num_epochs: int, lr: float):
        """Train on a new task."""
        self.current_task_id = task_id

        # Create new adapter for this task
        adapter_key = f"task_{task_id}"
        self.task_adapters[adapter_key] = self._create_adapter(task_id)

        if task_id == 0:
            # Task 0: Train NF base + adapter
            params = list(self.nf.parameters()) + list(self.task_adapters[adapter_key].parameters())
        else:
            # Task > 0: Freeze NF, train only adapter
            for param in self.nf.parameters():
                param.requires_grad = False
            params = list(self.task_adapters[adapter_key].parameters())

        optimizer = torch.optim.AdamW(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        self.nf.train()
        adapter = self.task_adapters[adapter_key]
        adapter.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                # Unpack batch: (images, labels, masks, class_labels, paths)
                images = batch[0].to(self.device)

                # Extract features
                with torch.no_grad():
                    features, spatial_shape = self.extractor(images, return_spatial_shape=True)
                    # features: (B, H*W, D)
                    B = features.shape[0]
                    H, W = spatial_shape
                    D = features.shape[-1]
                    features = features.reshape(B, H, W, D)  # (B, H, W, D)

                    # Add positional embedding
                    if self.pos_embedding_gen is not None:
                        features = self.pos_embedding_gen((H, W), features)

                # Apply feature-level adapter
                features = adapter(features)  # (B, H, W, D)

                # NF forward - expects (B, H, W, D)
                optimizer.zero_grad()
                z, logdet_patch = self.nf(features)  # z: (B, H, W, D), logdet_patch: (B, H, W)

                # NLL loss per patch: -log p(z) = 0.5 * ||z||^2 - log|det J|
                # z: (B, H, W, D), sum over D -> (B, H, W)
                nll_patch = 0.5 * torch.sum(z ** 2, dim=-1) - logdet_patch  # (B, H, W)
                loss = nll_patch.mean()  # Average over all patches

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / n_batches
            if (epoch + 1) % 10 == 0:
                print(f"  [Task {task_id}] Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

        # Collect feature statistics for routing
        self._collect_task_statistics(train_loader, adapter)

        return avg_loss

    def _collect_task_statistics(self, loader, adapter):
        """Collect feature statistics for prototype-based routing."""
        features_list = []

        self.nf.eval()
        adapter.eval()

        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(self.device)
                features, spatial_shape = self.extractor(images, return_spatial_shape=True)
                B = features.shape[0]
                H, W = spatial_shape
                D = features.shape[-1]
                features = features.reshape(B, H, W, D)

                if self.pos_embedding_gen is not None:
                    features = self.pos_embedding_gen((H, W), features)

                features = adapter(features)
                # Global average pooling
                features = features.mean(dim=[1, 2])  # (B, D)
                features_list.append(features.cpu())

        all_features = torch.cat(features_list, dim=0)
        mean = all_features.mean(dim=0)
        cov = torch.cov(all_features.T)

        self.task_prototypes[self.current_task_id] = {
            'mean': mean,
            'cov': cov + 1e-4 * torch.eye(cov.shape[0])  # Regularization
        }

    def evaluate(self, test_loaders: dict):
        """Evaluate on all tasks."""
        results = {}

        self.nf.eval()

        for task_id, loader in test_loaders.items():
            adapter_key = f"task_{task_id}"
            if adapter_key not in self.task_adapters:
                continue

            adapter = self.task_adapters[adapter_key]
            adapter.eval()

            all_scores = []
            all_labels = []

            with torch.no_grad():
                for batch in loader:
                    images = batch[0].to(self.device)
                    labels = batch[1]

                    features, spatial_shape = self.extractor(images, return_spatial_shape=True)
                    B = features.shape[0]
                    H, W = spatial_shape
                    D = features.shape[-1]
                    features = features.reshape(B, H, W, D)

                    if self.pos_embedding_gen is not None:
                        features = self.pos_embedding_gen((H, W), features)

                    features = adapter(features)  # (B, H, W, D)

                    z, logdet_patch = self.nf(features)  # z: (B, H, W, D), logdet_patch: (B, H, W)

                    # Anomaly score = NLL per patch
                    nll_patch = 0.5 * torch.sum(z ** 2, dim=-1) - logdet_patch  # (B, H, W)
                    scores = nll_patch.reshape(B, -1).mean(dim=1)  # (B,) - mean over patches

                    all_scores.append(scores.cpu())
                    all_labels.append(labels)

            all_scores = torch.cat(all_scores).numpy()
            all_labels = torch.cat(all_labels).numpy()

            # Compute AUC
            if len(np.unique(all_labels)) > 1:
                auc = roc_auc_score(all_labels, all_scores)
            else:
                auc = 0.5

            results[task_id] = {'img_auc': auc}

        return results


def create_args(data_path, dataset, img_size=224, msk_size=256):
    """Create args namespace for dataset creation."""
    args = SimpleNamespace()
    args.data_path = data_path
    args.dataset = dataset
    args.img_size = img_size
    args.msk_size = msk_size
    return args


def main():
    parser = argparse.ArgumentParser(description='Feature-level Adaptation Baseline')
    parser.add_argument('--task_classes', type=str, nargs='+',
                        default=['bottle', 'cable', 'capsule', 'carpet', 'grid',
                                 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                                 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'])
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_path', type=str, default='/Data/MVTecAD')
    parser.add_argument('--dataset', type=str, default='mvtec')
    parser.add_argument('--backbone_name', type=str, default='wide_resnet50_2')
    parser.add_argument('--num_coupling_layers', type=int, default=8)
    parser.add_argument('--adapter_type', type=str, default='prompt',
                        choices=['prompt', 'mlp'])
    parser.add_argument('--experiment_name', type=str, default='FeatureLevel')
    parser.add_argument('--log_dir', type=str, default='logs/FeatureLevel_Baseline')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--target_params', type=int, default=1_930_000,
                        help='Target parameter count for adapter (should match LoRA params)')

    args = parser.parse_args()

    # Setup
    init_seeds(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.experiment_name}_{args.adapter_type}_{timestamp}"
    log_dir = Path(args.log_dir) / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 70)
    print(f"Feature-Level Adaptation Experiment")
    print(f"=" * 70)
    print(f"Adapter Type: {args.adapter_type}")
    print(f"Tasks: {args.task_classes}")
    print(f"Target Params: {args.target_params:,}")
    print(f"Log Directory: {log_dir}")

    # Save config
    with open(log_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Create feature extractor
    extractor = create_feature_extractor(
        backbone_name=args.backbone_name,
        device=device
    )
    embed_dim = extractor.target_embed_dimension

    # Create NF model (base only, no LoRA)
    ablation_config = AblationConfig()
    ablation_config.use_lora = False  # Disable LoRA for feature-level baseline

    nf = MoLESpatialAwareNF(
        embed_dim=embed_dim,
        coupling_layers=args.num_coupling_layers,
        lora_rank=1,  # Minimal rank (LoRA disabled via config)
        device=device,
        ablation_config=ablation_config
    ).to(device)

    # Positional embedding
    pos_gen = PositionalEmbeddingGenerator(device=device)

    # Create trainer
    trainer = FeatureLevelTrainer(
        nf_model=nf,
        feature_extractor=extractor,
        pos_embedding_gen=pos_gen,
        embed_dim=embed_dim,
        device=device,
        log_dir=log_dir,
        adapter_type=args.adapter_type,
        target_params=args.target_params
    )

    # Setup data args
    data_args = create_args(args.data_path, args.dataset)

    # Global class to idx mapping
    GLOBAL_CLASS_TO_IDX = {cls: i for i, cls in enumerate(args.task_classes)}

    # Train on each task
    all_results = []
    test_loaders = {}

    for task_id, task_class in enumerate(args.task_classes):
        print(f"\n{'='*70}")
        print(f"Training Task {task_id}: {task_class}")
        print(f"{'='*70}")

        # Setup args for this task
        data_args.class_to_idx = {task_class: GLOBAL_CLASS_TO_IDX[task_class]}
        data_args.n_classes = 1

        # Create data loaders
        train_dataset = create_task_dataset(
            data_args, [task_class], GLOBAL_CLASS_TO_IDX, train=True
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=4, drop_last=True)

        test_dataset = create_task_dataset(
            data_args, [task_class], GLOBAL_CLASS_TO_IDX, train=False
        )
        test_loaders[task_id] = DataLoader(test_dataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=4)

        # Train
        final_loss = trainer.train_task(task_id, train_loader, args.num_epochs, args.lr)

        # Evaluate all tasks
        print(f"\nEvaluating after Task {task_id}...")
        results = trainer.evaluate(test_loaders)

        print(f"\nResults after Task {task_id}:")
        for tid, res in results.items():
            print(f"  Task {tid} ({args.task_classes[tid]}): I-AUC = {res['img_auc']:.4f}")

        all_results.append(results)

        # Save intermediate results
        with open(log_dir / f'results_after_task_{task_id}.json', 'w') as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    # Compute FM and BWT
    print(f"\n{'='*70}")
    print(f"Final Analysis")
    print(f"{'='*70}")

    # Final results
    final_results = all_results[-1]
    avg_auc = np.mean([r['img_auc'] for r in final_results.values()])
    print(f"Final Average I-AUC: {avg_auc:.4f}")

    # Compute BWT
    bwt_sum = 0
    n_prev_tasks = len(args.task_classes) - 1

    for task_id in range(n_prev_tasks):
        # Performance after training task_id
        perf_after_train = all_results[task_id].get(task_id, {}).get('img_auc', 0)
        # Final performance
        perf_final = final_results.get(task_id, {}).get('img_auc', 0)
        bwt_sum += perf_final - perf_after_train

    bwt = bwt_sum / n_prev_tasks if n_prev_tasks > 0 else 0
    print(f"BWT (I-AUC): {bwt:+.4f}")

    # Save final summary
    summary = {
        'final_avg_auc': avg_auc,
        'bwt': bwt,
        'adapter_type': args.adapter_type,
        'num_tasks': len(args.task_classes),
        'per_task_final': {str(k): v for k, v in final_results.items()}
    }
    with open(log_dir / 'final_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nExperiment completed! Results saved to: {log_dir}")


if __name__ == '__main__':
    main()
