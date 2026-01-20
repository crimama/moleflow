#!/usr/bin/env python3
"""
SVD Analysis for Full Fine-tuning Weight Changes

This script validates the theoretical claim that "Low-rank adaptation is sufficient"
by analyzing the intrinsic dimensionality of weight changes during full fine-tuning.

Key Analysis:
1. Train a base NF model on Task 0 (e.g., leather)
2. Full fine-tune the model on Task 1 (e.g., grid) - NO LoRA, NO freezing
3. Compute Delta W = W_task1 - W_base for each layer
4. Perform SVD on Delta W and analyze singular value spectrum
5. Compute effective rank and energy concentration

Expected Result:
- If top-k singular values capture 95%+ of energy, this proves that
  task-specific adaptation is intrinsically low-rank
- This justifies using LoRA with rank=k instead of full fine-tuning

Author: MoLE-Flow Team
Date: 2026-01-16
"""

import os
import sys
import json
import argparse
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from moleflow.models.mole_nf import MoLESpatialAwareNF
from moleflow.models.lora import LoRALinear, MoLESubnet, MoLEContextSubnet
from moleflow.extractors.vit_extractor import ViTPatchCoreExtractor
from moleflow.extractors.cnn_extractor import CNNPatchCoreExtractor
from moleflow.models.position_embedding import PositionalEmbeddingGenerator
from moleflow.data.mvtec import MVTEC
from moleflow.config.ablation import AblationConfig
from torch.utils.data import DataLoader


# CNN backbone names
CNN_BACKBONES = ['resnet18', 'resnet50', 'resnet101', 'wide_resnet50_2', 'wide_resnet101_2',
                 'efficientnet_b0', 'efficientnet_b4', 'convnext_tiny', 'convnext_small']


def get_mvtec_train_loader(data_path: str, class_name: str, batch_size: int, img_size: int) -> DataLoader:
    """Create a training data loader for MVTec AD dataset."""
    dataset = MVTEC(
        root=data_path,
        class_name=class_name,
        train=True,
        img_size=img_size,
        crp_size=img_size
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


def compute_effective_rank(singular_values: torch.Tensor, threshold: float = 0.95) -> int:
    """Compute effective rank based on energy threshold."""
    S_squared = singular_values ** 2
    total_energy = S_squared.sum()

    if total_energy == 0:
        return 0

    cumulative_energy = S_squared.cumsum(0) / total_energy
    effective_rank = (cumulative_energy < threshold).sum().item() + 1

    return min(effective_rank, len(singular_values))


def compute_energy_at_rank(singular_values: torch.Tensor, rank: int) -> float:
    """Compute fraction of energy captured by top-k singular values."""
    if len(singular_values) == 0:
        return 0.0

    S_squared = singular_values ** 2
    total_energy = S_squared.sum()

    if total_energy == 0:
        return 0.0

    k = min(rank, len(singular_values))
    top_k_energy = S_squared[:k].sum()

    return (top_k_energy / total_energy).item()


def extract_linear_weights(model: MoLESpatialAwareNF) -> Dict[str, torch.Tensor]:
    """Extract all linear layer weights from NF model.

    Returns CPU tensors for reliable comparison across models.
    """
    weights = {}

    for subnet_idx, subnet in enumerate(model.subnets):
        if hasattr(subnet, 's_layer1'):
            # MoLEContextSubnet: 4 layers
            layers = [
                ('s_layer1', subnet.s_layer1),
                ('s_layer2', subnet.s_layer2),
                ('t_layer1', subnet.t_layer1),
                ('t_layer2', subnet.t_layer2)
            ]
        elif hasattr(subnet, 'layer1'):
            # MoLESubnet: 2 layers
            if hasattr(subnet.layer1, 'base_linear'):
                # LoRALinear
                layers = [
                    ('layer1', subnet.layer1),
                    ('layer2', subnet.layer2)
                ]
            else:
                # Regular Linear
                layers = [
                    ('layer1_weight', subnet.layer1),
                    ('layer2_weight', subnet.layer2)
                ]
        else:
            continue

        for layer_name, layer in layers:
            key = f"subnet{subnet_idx}_{layer_name}"

            if hasattr(layer, 'base_linear'):
                # LoRALinear - extract base weight (move to CPU for reliable comparison)
                weights[key] = layer.base_linear.weight.data.detach().cpu().clone()
            elif hasattr(layer, 'weight'):
                # Regular Linear (move to CPU for reliable comparison)
                weights[key] = layer.weight.data.detach().cpu().clone()

    return weights


def analyze_weight_change(
    W_base: torch.Tensor,
    W_task: torch.Tensor,
    layer_name: str
) -> Dict:
    """
    Analyze the SVD of weight change Delta W = W_task - W_base.

    Args:
        W_base: Base weight matrix from Task 0
        W_task: Weight matrix after fine-tuning on Task 1
        layer_name: Name of the layer for identification

    Returns:
        Dictionary with SVD analysis results
    """
    # Compute weight change
    Delta_W = W_task - W_base

    # Check if there's any change
    delta_norm = torch.norm(Delta_W).item()
    base_norm = torch.norm(W_base).item()

    if delta_norm < 1e-10:
        return {
            'layer_name': layer_name,
            'delta_norm': delta_norm,
            'base_norm': base_norm,
            'relative_change': 0.0,
            'no_change': True
        }

    # SVD decomposition
    U, S, Vh = torch.linalg.svd(Delta_W, full_matrices=False)

    # Compute effective ranks at different thresholds
    effective_rank_90 = compute_effective_rank(S, 0.90)
    effective_rank_95 = compute_effective_rank(S, 0.95)
    effective_rank_99 = compute_effective_rank(S, 0.99)
    effective_rank_999 = compute_effective_rank(S, 0.999)

    # Compute energy at common LoRA ranks
    energy_at_rank_8 = compute_energy_at_rank(S, 8)
    energy_at_rank_16 = compute_energy_at_rank(S, 16)
    energy_at_rank_32 = compute_energy_at_rank(S, 32)
    energy_at_rank_64 = compute_energy_at_rank(S, 64)
    energy_at_rank_128 = compute_energy_at_rank(S, 128)

    # Spectral properties
    spectral_norm = S[0].item() if len(S) > 0 else 0
    frobenius_norm = torch.sqrt((S ** 2).sum()).item()
    nuclear_norm = S.sum().item()

    # Matrix dimensions
    out_features, in_features = Delta_W.shape
    max_rank = min(out_features, in_features)

    return {
        'layer_name': layer_name,
        'shape': (out_features, in_features),
        'max_rank': max_rank,
        'delta_norm': delta_norm,
        'base_norm': base_norm,
        'relative_change': delta_norm / base_norm if base_norm > 0 else 0,
        'spectral_norm': spectral_norm,
        'frobenius_norm': frobenius_norm,
        'nuclear_norm': nuclear_norm,
        'singular_values': S.cpu().numpy(),
        'effective_rank_90': effective_rank_90,
        'effective_rank_95': effective_rank_95,
        'effective_rank_99': effective_rank_99,
        'effective_rank_999': effective_rank_999,
        'energy_at_rank_8': energy_at_rank_8,
        'energy_at_rank_16': energy_at_rank_16,
        'energy_at_rank_32': energy_at_rank_32,
        'energy_at_rank_64': energy_at_rank_64,
        'energy_at_rank_128': energy_at_rank_128,
        'no_change': False
    }


def train_nf_epoch(
    model: nn.Module,
    extractor: nn.Module,
    pos_generator: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    skip_high_loss: bool = False
) -> float:
    """Train NF model for one epoch.

    Args:
        skip_high_loss: If True, skip batches with very high loss (default for normal training).
                       If False, train on all batches (needed for fine-tuning across domains).
    """
    import math

    model.train()
    extractor.eval()

    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        images = batch[0].to(device)

        with torch.no_grad():
            patch_embeddings, spatial_shape = extractor(images, return_spatial_shape=True)
            B = patch_embeddings.shape[0]
            H, W = spatial_shape

            # Apply positional embedding
            patch_embeddings_with_pos = pos_generator(spatial_shape, patch_embeddings)

        # Forward through NF
        z, logdet_patch = model.forward(patch_embeddings_with_pos, reverse=False)

        # Compute NLL loss
        B, H, W, D = z.shape
        log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
        log_px_patch = log_pz_patch + logdet_patch
        log_px_image = log_px_patch.sum(dim=(1, 2))
        nll_loss = -log_px_image.mean()

        # Skip NaN losses always, but high losses only when skip_high_loss is True
        if torch.isnan(nll_loss):
            continue
        if skip_high_loss and nll_loss.item() > 1e8:
            continue

        # Clamp loss for stability in cross-domain fine-tuning
        if nll_loss.item() > 1e8:
            nll_loss = nll_loss.clamp(max=1e8)

        optimizer.zero_grad()
        nll_loss.backward()
        # More aggressive gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += nll_loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def create_full_finetune_model(embed_dim: int, coupling_layers: int, device: str) -> MoLESpatialAwareNF:
    """
    Create NF model with regular Linear layers (no LoRA).
    This is for full fine-tuning experiment.
    """
    # Create ablation config that disables LoRA
    ablation_config = AblationConfig()
    ablation_config.use_lora = False  # No LoRA - use regular Linear
    ablation_config.use_task_adapter = False  # No TaskInputAdapter
    ablation_config.use_task_bias = False  # No task-specific bias
    ablation_config.use_spatial_context = False  # Simpler model
    ablation_config.use_scale_context = False

    model = MoLESpatialAwareNF(
        embed_dim=embed_dim,
        coupling_layers=coupling_layers,
        lora_rank=64,  # Not used since use_lora=False
        device=device,
        ablation_config=ablation_config
    )

    return model


def plot_singular_value_spectrum(
    results: List[Dict],
    output_dir: Path,
    title_suffix: str = ""
):
    """Plot singular value spectrum for all layers."""
    # Filter results with actual changes
    valid_results = [r for r in results if not r.get('no_change', True)]

    if not valid_results:
        print("No layers with weight changes to plot.")
        return

    # Select representative layers (first 8)
    selected = valid_results[:min(8, len(valid_results))]

    n_plots = len(selected)
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, r in enumerate(selected):
        ax = axes[idx]
        S = r['singular_values']

        # Plot singular values (log scale)
        ax.semilogy(S, 'b-', linewidth=2, label='Singular Values')

        # Mark effective ranks
        ax.axvline(r['effective_rank_95'], color='r', linestyle='--',
                   label=f"95% rank: {r['effective_rank_95']}")
        ax.axvline(r['effective_rank_99'], color='g', linestyle=':',
                   label=f"99% rank: {r['effective_rank_99']}")

        # Mark common LoRA ranks
        for rank, color in [(8, 'orange'), (32, 'purple'), (64, 'cyan')]:
            if rank < len(S):
                ax.axvline(rank, color=color, linestyle='-.', alpha=0.5,
                          label=f"r={rank}: {r[f'energy_at_rank_{rank}']:.1%}")

        ax.set_xlabel('Index')
        ax.set_ylabel('Singular Value')
        ax.set_title(f"{r['layer_name']}\nShape: {r['shape']}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Delta W Singular Value Spectrum {title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'svd_spectrum{title_suffix.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_energy_at_ranks(results: List[Dict], output_dir: Path):
    """Plot energy captured at different LoRA ranks."""
    valid_results = [r for r in results if not r.get('no_change', True)]

    if not valid_results:
        return

    ranks = [8, 16, 32, 64, 128]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Energy at different ranks (per layer)
    ax1 = axes[0]

    layer_names = [r['layer_name'][:15] for r in valid_results]
    x = np.arange(len(layer_names))
    width = 0.15

    for i, rank in enumerate(ranks):
        energies = [r.get(f'energy_at_rank_{rank}', 0) for r in valid_results]
        ax1.bar(x + i * width, energies, width, label=f'r={rank}')

    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Energy Captured')
    ax1.set_title('Energy Captured by Top-k Singular Values')
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    ax1.axhline(0.95, color='r', linestyle='--', label='95% threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Average energy across all layers
    ax2 = axes[1]

    avg_energies = []
    std_energies = []
    for rank in ranks:
        energies = [r.get(f'energy_at_rank_{rank}', 0) for r in valid_results]
        avg_energies.append(np.mean(energies))
        std_energies.append(np.std(energies))

    ax2.bar(range(len(ranks)), avg_energies, yerr=std_energies, capsize=5, color='steelblue')
    ax2.set_xticks(range(len(ranks)))
    ax2.set_xticklabels([f'r={r}' for r in ranks])
    ax2.set_xlabel('LoRA Rank')
    ax2.set_ylabel('Average Energy Captured')
    ax2.set_title('Average Energy vs LoRA Rank (All Layers)')
    ax2.axhline(0.95, color='r', linestyle='--', label='95% threshold')
    ax2.axhline(0.99, color='g', linestyle=':', label='99% threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / 'energy_at_ranks.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_effective_rank_histogram(results: List[Dict], output_dir: Path):
    """Plot histogram of effective ranks."""
    valid_results = [r for r in results if not r.get('no_change', True)]

    if not valid_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Effective rank histogram
    ax1 = axes[0]
    ranks_95 = [r['effective_rank_95'] for r in valid_results]
    ranks_99 = [r['effective_rank_99'] for r in valid_results]

    ax1.hist(ranks_95, bins=20, alpha=0.7, label='95% energy', color='blue')
    ax1.hist(ranks_99, bins=20, alpha=0.7, label='99% energy', color='green')
    ax1.axvline(np.mean(ranks_95), color='blue', linestyle='--',
                label=f'Mean 95%: {np.mean(ranks_95):.1f}')
    ax1.axvline(np.mean(ranks_99), color='green', linestyle='--',
                label=f'Mean 99%: {np.mean(ranks_99):.1f}')
    ax1.set_xlabel('Effective Rank')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Effective Ranks')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Effective rank vs max rank
    ax2 = axes[1]
    max_ranks = [r['max_rank'] for r in valid_results]

    ax2.scatter(max_ranks, ranks_95, alpha=0.7, label='95% energy')
    ax2.scatter(max_ranks, ranks_99, alpha=0.7, label='99% energy')

    # Add reference lines
    max_val = max(max_ranks)
    ax2.plot([0, max_val], [0, max_val], 'r--', label='y=x (full rank)')
    ax2.plot([0, max_val], [0, max_val * 0.1], 'g:', label='10% of max')

    ax2.set_xlabel('Max Possible Rank (min(out, in))')
    ax2.set_ylabel('Effective Rank')
    ax2.set_title('Effective Rank vs Matrix Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'effective_rank_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_summary_report(results: List[Dict], output_dir: Path, args) -> Dict:
    """Generate comprehensive summary report."""
    valid_results = [r for r in results if not r.get('no_change', True)]

    if not valid_results:
        return {'error': 'No layers with weight changes'}

    # Aggregate statistics
    ranks_95 = [r['effective_rank_95'] for r in valid_results]
    ranks_99 = [r['effective_rank_99'] for r in valid_results]

    energy_8 = [r['energy_at_rank_8'] for r in valid_results]
    energy_16 = [r['energy_at_rank_16'] for r in valid_results]
    energy_32 = [r['energy_at_rank_32'] for r in valid_results]
    energy_64 = [r['energy_at_rank_64'] for r in valid_results]

    relative_changes = [r['relative_change'] for r in valid_results]

    summary = {
        'experiment_config': {
            'task0_class': args.task0_class,
            'task1_class': args.task1_class,
            'num_epochs': args.num_epochs,
            'coupling_layers': args.coupling_layers,
            'backbone': args.backbone
        },
        'effective_rank': {
            'mean_95': np.mean(ranks_95),
            'std_95': np.std(ranks_95),
            'min_95': np.min(ranks_95),
            'max_95': np.max(ranks_95),
            'mean_99': np.mean(ranks_99),
            'std_99': np.std(ranks_99)
        },
        'energy_at_common_ranks': {
            'rank_8': {'mean': np.mean(energy_8), 'std': np.std(energy_8)},
            'rank_16': {'mean': np.mean(energy_16), 'std': np.std(energy_16)},
            'rank_32': {'mean': np.mean(energy_32), 'std': np.std(energy_32)},
            'rank_64': {'mean': np.mean(energy_64), 'std': np.std(energy_64)}
        },
        'relative_weight_change': {
            'mean': np.mean(relative_changes),
            'std': np.std(relative_changes),
            'max': np.max(relative_changes)
        },
        'num_layers_analyzed': len(valid_results),
        'timestamp': datetime.now().isoformat()
    }

    # Determine optimal LoRA rank
    for rank, energy_list in [(8, energy_8), (16, energy_16), (32, energy_32), (64, energy_64)]:
        if np.mean(energy_list) >= 0.95:
            summary['recommended_lora_rank'] = rank
            break
    else:
        summary['recommended_lora_rank'] = 128

    return summary


def main():
    parser = argparse.ArgumentParser(description='SVD Analysis of Full Fine-tuning Weight Changes')
    parser.add_argument('--data_path', type=str, default='/Data/MVTecAD',
                        help='Path to MVTec AD dataset')
    parser.add_argument('--task0_class', type=str, default='leather',
                        help='Class for Task 0 (base training)')
    parser.add_argument('--task1_class', type=str, default='grid',
                        help='Class for Task 1 (fine-tuning)')
    parser.add_argument('--backbone', type=str, default='wide_resnet50_2',
                        help='Backbone model name')
    parser.add_argument('--coupling_layers', type=int, default=8,
                        help='Number of coupling layers in NF')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of training epochs per task')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./analysis_results/svd_full_finetune',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--mode', type=str, default='independent',
                        choices=['independent', 'finetune'],
                        help='Analysis mode: independent (train two models from scratch) '
                             'or finetune (fine-tune task0 model on task1)')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"SVD Analysis: {'Independent Training' if args.mode == 'independent' else 'Fine-tuning'} Mode")
    print("=" * 70)
    print(f"Task 0: {args.task0_class}")
    print(f"Task 1: {args.task1_class}")
    print(f"Mode: {args.mode}")
    print(f"Backbone: {args.backbone}")
    print(f"Coupling Layers: {args.coupling_layers}")
    print(f"Epochs per task: {args.num_epochs}")
    print(f"Output: {output_dir}")
    print()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create feature extractor
    print("\nInitializing feature extractor...")
    if any(cnn_name in args.backbone.lower() for cnn_name in ['resnet', 'wide_resnet', 'efficientnet', 'convnext']):
        extractor = CNNPatchCoreExtractor(
            backbone_name=args.backbone,
            device=device,
            input_shape=(3, args.img_size, args.img_size)
        )
    else:
        extractor = ViTPatchCoreExtractor(
            backbone_name=args.backbone,
            device=device,
            input_shape=(3, args.img_size, args.img_size)
        )
    embed_dim = getattr(extractor, 'embed_dim', None) or extractor.target_embed_dimension
    print(f"Embedding dimension: {embed_dim}")

    # Create positional embedding generator
    pos_generator = PositionalEmbeddingGenerator(device=device)

    if args.mode == 'independent':
        # ============================================================
        # INDEPENDENT MODE: Train two models from same initialization
        # This shows intrinsic rank of task-specific differences
        # ============================================================

        # Create model and save initial (random) weights
        print(f"\n{'='*70}")
        print("Creating model and saving initial weights...")
        print(f"{'='*70}")

        model_init = create_full_finetune_model(embed_dim, args.coupling_layers, device)
        model_init.add_task(0)

        # Save initial (random) weights
        init_state_dict = {k: v.detach().cpu().clone() for k, v in model_init.state_dict().items()}
        print(f"Saved initial weights ({len(init_state_dict)} tensors)")

        # ================================================================
        # Step 1: Train model on Task 0 from initial weights
        # ================================================================
        print(f"\n{'='*70}")
        print(f"Step 1: Training model on {args.task0_class}")
        print(f"{'='*70}")

        model_task0 = create_full_finetune_model(embed_dim, args.coupling_layers, device)
        model_task0.add_task(0)
        model_task0.load_state_dict({k: v.to(device) for k, v in init_state_dict.items()})

        train_loader_task0 = get_mvtec_train_loader(
            data_path=args.data_path,
            class_name=args.task0_class,
            batch_size=args.batch_size,
            img_size=args.img_size
        )

        optimizer_task0 = torch.optim.AdamW(model_task0.parameters(), lr=args.lr, weight_decay=1e-4)

        for epoch in range(args.num_epochs):
            loss = train_nf_epoch(model_task0, extractor, pos_generator, train_loader_task0, optimizer_task0, device)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{args.num_epochs}] Loss: {loss:.4f}")

        weights_task0 = extract_linear_weights(model_task0)
        print(f"\nExtracted {len(weights_task0)} layers from {args.task0_class} model")

        # ================================================================
        # Step 2: Train model on Task 1 from SAME initial weights
        # ================================================================
        print(f"\n{'='*70}")
        print(f"Step 2: Training model on {args.task1_class}")
        print(f"{'='*70}")

        model_task1 = create_full_finetune_model(embed_dim, args.coupling_layers, device)
        model_task1.add_task(0)
        model_task1.load_state_dict({k: v.to(device) for k, v in init_state_dict.items()})

        train_loader_task1 = get_mvtec_train_loader(
            data_path=args.data_path,
            class_name=args.task1_class,
            batch_size=args.batch_size,
            img_size=args.img_size
        )

        optimizer_task1 = torch.optim.AdamW(model_task1.parameters(), lr=args.lr, weight_decay=1e-4)

        for epoch in range(args.num_epochs):
            loss = train_nf_epoch(model_task1, extractor, pos_generator, train_loader_task1, optimizer_task1, device)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{args.num_epochs}] Loss: {loss:.4f}")

        weights_task1 = extract_linear_weights(model_task1)
        print(f"\nExtracted {len(weights_task1)} layers from {args.task1_class} model")

    else:
        # ============================================================
        # FINETUNE MODE: Fine-tune Task 0 model on Task 1
        # ============================================================

        # Step 1: Train base model on Task 0
        print(f"\n{'='*70}")
        print(f"Step 1: Training base model on {args.task0_class}")
        print(f"{'='*70}")

        model_task0 = create_full_finetune_model(embed_dim, args.coupling_layers, device)
        model_task0.add_task(0)

        train_loader_task0 = get_mvtec_train_loader(
            data_path=args.data_path,
            class_name=args.task0_class,
            batch_size=args.batch_size,
            img_size=args.img_size
        )

        optimizer_task0 = torch.optim.AdamW(model_task0.parameters(), lr=args.lr, weight_decay=1e-4)

        for epoch in range(args.num_epochs):
            loss = train_nf_epoch(model_task0, extractor, pos_generator, train_loader_task0, optimizer_task0, device)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{args.num_epochs}] Loss: {loss:.4f}")

        base_state_dict = {k: v.detach().cpu().clone() for k, v in model_task0.state_dict().items()}
        weights_task0 = extract_linear_weights(model_task0)
        print(f"\nExtracted {len(weights_task0)} layers")

        # Step 2: Fine-tune on Task 1
        print(f"\n{'='*70}")
        print(f"Step 2: Fine-tuning on {args.task1_class}")
        print(f"{'='*70}")

        model_task1 = create_full_finetune_model(embed_dim, args.coupling_layers, device)
        model_task1.add_task(0)
        model_task1.load_state_dict({k: v.to(device) for k, v in base_state_dict.items()})

        train_loader_task1 = get_mvtec_train_loader(
            data_path=args.data_path,
            class_name=args.task1_class,
            batch_size=args.batch_size,
            img_size=args.img_size
        )

        optimizer_task1 = torch.optim.AdamW(model_task1.parameters(), lr=args.lr, weight_decay=1e-4)

        for epoch in range(args.num_epochs):
            loss = train_nf_epoch(model_task1, extractor, pos_generator, train_loader_task1,
                                  optimizer_task1, device, skip_high_loss=False)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{args.num_epochs}] Loss: {loss:.4f}")

        weights_task1 = extract_linear_weights(model_task1)
        print(f"\nExtracted {len(weights_task1)} layers")

    # ================================================================
    # Step 3: SVD Analysis of Weight Changes
    # ================================================================
    print(f"\n{'='*70}")
    print("Step 3: SVD Analysis of Weight Changes (Delta W = W_task1 - W_base)")
    print(f"{'='*70}")

    results = []

    for layer_name in weights_task0.keys():
        if layer_name not in weights_task1:
            continue

        W_base = weights_task0[layer_name]
        W_task = weights_task1[layer_name]

        analysis = analyze_weight_change(W_base, W_task, layer_name)
        results.append(analysis)

        if not analysis.get('no_change', False):
            print(f"\n{layer_name}:")
            print(f"  Shape: {analysis['shape']}")
            print(f"  Relative change: {analysis['relative_change']:.4f}")
            print(f"  Effective rank (95%): {analysis['effective_rank_95']}")
            print(f"  Effective rank (99%): {analysis['effective_rank_99']}")
            print(f"  Energy at r=64: {analysis['energy_at_rank_64']:.3f}")

    # ================================================================
    # Step 4: Generate Plots and Report
    # ================================================================
    print(f"\n{'='*70}")
    print("Step 4: Generating Plots and Report")
    print(f"{'='*70}")

    # Plot singular value spectrum
    plot_singular_value_spectrum(results, output_dir, f" ({args.task0_class} -> {args.task1_class})")
    print("  - Saved: svd_spectrum.png")

    # Plot energy at different ranks
    plot_energy_at_ranks(results, output_dir)
    print("  - Saved: energy_at_ranks.png")

    # Plot effective rank histogram
    plot_effective_rank_histogram(results, output_dir)
    print("  - Saved: effective_rank_histogram.png")

    # Generate summary
    summary = generate_summary_report(results, output_dir, args)

    # Save results
    with open(output_dir / 'analysis_results.json', 'w') as f:
        # Convert numpy arrays for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump({
            'summary': convert(summary),
            'per_layer_results': convert([{k: v for k, v in r.items() if k != 'singular_values'} for r in results])
        }, f, indent=2)
    print("  - Saved: analysis_results.json")

    # ================================================================
    # Final Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    valid_results = [r for r in results if not r.get('no_change', True)]

    if valid_results:
        ranks_95 = [r['effective_rank_95'] for r in valid_results]
        energy_64 = [r['energy_at_rank_64'] for r in valid_results]

        print(f"\nEffective Rank (95% energy):")
        print(f"  Mean: {np.mean(ranks_95):.1f}")
        print(f"  Std:  {np.std(ranks_95):.1f}")
        print(f"  Range: [{np.min(ranks_95)}, {np.max(ranks_95)}]")

        print(f"\nEnergy captured at rank=64:")
        print(f"  Mean: {np.mean(energy_64):.3f} ({np.mean(energy_64)*100:.1f}%)")
        print(f"  Min:  {np.min(energy_64):.3f}")

        print(f"\nRecommended LoRA rank: {summary.get('recommended_lora_rank', 'N/A')}")

        # Interpretation
        print(f"\n{'='*70}")
        print("INTERPRETATION")
        print(f"{'='*70}")

        mean_rank_95 = np.mean(ranks_95)
        mean_energy_64 = np.mean(energy_64)

        if mean_rank_95 < 32:
            print("[STRONG EVIDENCE] Task adaptation is intrinsically very low-rank.")
            print(f"  - Mean effective rank ({mean_rank_95:.1f}) << typical LoRA rank (64)")
            print("  - LoRA with rank=32 should be sufficient for this task pair.")
        elif mean_rank_95 < 64:
            print("[MODERATE EVIDENCE] Task adaptation has moderate intrinsic dimensionality.")
            print(f"  - Mean effective rank ({mean_rank_95:.1f}) < LoRA rank (64)")
            print("  - LoRA with rank=64 is appropriate.")
        else:
            print("[WEAK EVIDENCE] Task adaptation requires higher rank.")
            print(f"  - Mean effective rank ({mean_rank_95:.1f}) >= LoRA rank (64)")
            print("  - Consider using rank=128 or higher.")

        if mean_energy_64 > 0.99:
            print(f"\n[VERY STRONG] r=64 captures {mean_energy_64*100:.1f}% energy on average.")
            print("  -> Low-rank adaptation is highly justified!")
        elif mean_energy_64 > 0.95:
            print(f"\n[STRONG] r=64 captures {mean_energy_64*100:.1f}% energy on average.")
            print("  -> Low-rank adaptation is well justified.")
        elif mean_energy_64 > 0.90:
            print(f"\n[MODERATE] r=64 captures {mean_energy_64*100:.1f}% energy on average.")
            print("  -> Low-rank adaptation is reasonable but some information is lost.")

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
