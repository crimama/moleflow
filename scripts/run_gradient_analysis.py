#!/usr/bin/env python
"""
Experiment 3: Gradient Dynamics Analysis for Tail-Aware Loss.

This experiment analyzes HOW Tail-Aware Loss changes gradient distribution during training.

Hypotheses:
- H3: Mean-only training dilutes gradients from tail patches.
- H6: Tail training improves gradient focus on hard patches.

Key Measurements:
1. Per-patch gradient magnitude distribution
2. Gradient concentration (Gini coefficient)
3. Layer-wise gradient statistics
4. Which patches receive the strongest gradients

Author: MoLE-Flow Research Team
"""

import os
import sys
import math
import json
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy import stats

# Add project root to path
sys.path.insert(0, '/Volume/MoLeFlow')

from moleflow import (
    MoLESpatialAwareNF,
    MoLEContinualTrainer,
    PositionalEmbeddingGenerator,
    create_feature_extractor,
    get_backbone_type,
    init_seeds,
    create_task_dataset,
)
from moleflow.config import AblationConfig


class GradientDynamicsAnalyzer:
    """
    Analyzer for gradient dynamics in Tail-Aware Loss training.

    Key analyses:
    1. Per-patch gradient magnitude distribution
    2. Layer-wise gradient statistics
    3. Gradient concentration (Gini coefficient)
    4. Gradient-NLL correlation
    """

    def __init__(self,
                 nf_model: nn.Module,
                 vit_extractor,
                 pos_embed_generator,
                 device: str = 'cuda'):
        self.nf_model = nf_model
        self.vit_extractor = vit_extractor
        self.pos_embed_generator = pos_embed_generator
        self.device = device

    def analyze_gradient_for_configuration(self,
                                           train_loader: DataLoader,
                                           task_id: int,
                                           tail_weight: float,
                                           tail_ratio: float = 0.02,
                                           num_batches: int = 20) -> dict:
        """
        Analyze gradients for a specific loss configuration.

        Args:
            train_loader: Training data loader
            task_id: Task ID
            tail_weight: Weight for tail loss (0=mean only, 1=tail only, 0.7=balanced)
            tail_ratio: Ratio of patches considered as tail
            num_batches: Number of batches to analyze

        Returns:
            Dict with gradient statistics
        """
        self.nf_model.train()
        self.vit_extractor.eval()

        # Storage for analysis
        all_patch_gradients = []  # Per-patch gradient magnitudes
        all_nll_values = []       # Corresponding NLL values per patch
        all_tail_masks = []       # Binary mask for tail patches
        layer_gradients = defaultdict(list)  # Layer-wise gradients

        with torch.enable_grad():
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= num_batches:
                    break

                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.device)
                B = images.shape[0]

                # Extract features (detached, we want gradients w.r.t. model params)
                with torch.no_grad():
                    patch_embeddings, spatial_shape = self.vit_extractor(
                        images, return_spatial_shape=True
                    )
                    H, W = spatial_shape
                    if self.pos_embed_generator is not None:
                        features = self.pos_embed_generator(spatial_shape, patch_embeddings)
                    else:
                        features = patch_embeddings.reshape(B, H, W, -1)

                # Enable gradients on features for per-patch analysis
                features = features.clone().detach().requires_grad_(True)

                # Zero model gradients
                self.nf_model.zero_grad()

                # Forward pass
                self.nf_model.set_active_task(task_id)
                z, logdet_patch = self.nf_model(features)
                _, H, W, D = z.shape

                # Compute patch-wise NLL
                log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
                nll_patch = -(log_pz_patch + logdet_patch)  # (B, H, W)

                # Compute loss based on configuration
                flat_nll = nll_patch.reshape(B, -1)
                num_patches = H * W

                # Mean loss
                mean_loss = flat_nll.mean()

                # Tail loss
                k = max(1, int(num_patches * tail_ratio))
                top_k_nll, top_k_indices = torch.topk(flat_nll, k, dim=1)
                tail_loss = top_k_nll.mean()

                # Combined loss
                loss = (1 - tail_weight) * mean_loss + tail_weight * tail_loss

                # Backward pass
                loss.backward()

                # Collect per-patch gradient magnitudes from feature gradients
                if features.grad is not None:
                    grad_magnitude = features.grad.norm(dim=-1)  # (B, H, W)
                    all_patch_gradients.append(grad_magnitude.detach().cpu())
                    all_nll_values.append(nll_patch.detach().cpu())

                    # Create tail mask
                    tail_mask = torch.zeros(B, num_patches, device=self.device)
                    for b in range(B):
                        tail_mask[b, top_k_indices[b]] = 1.0
                    tail_mask = tail_mask.reshape(B, H, W)
                    all_tail_masks.append(tail_mask.cpu())

                # Collect layer-wise gradient statistics
                for name, param in self.nf_model.named_parameters():
                    if param.grad is not None:
                        grad_mag = param.grad.abs().mean().item()
                        layer_gradients[name].append(grad_mag)

                if (batch_idx + 1) % 5 == 0:
                    print(f"    Processed batch {batch_idx + 1}/{num_batches}")

        # Aggregate results
        all_patch_gradients = torch.cat(all_patch_gradients, dim=0)  # (N, H, W)
        all_nll_values = torch.cat(all_nll_values, dim=0)
        all_tail_masks = torch.cat(all_tail_masks, dim=0)

        # Compute statistics
        results = self._compute_gradient_statistics(
            all_patch_gradients,
            all_nll_values,
            all_tail_masks,
            layer_gradients,
            tail_weight
        )

        return results

    def _compute_gradient_statistics(self,
                                     patch_gradients: torch.Tensor,
                                     nll_values: torch.Tensor,
                                     tail_masks: torch.Tensor,
                                     layer_gradients: dict,
                                     tail_weight: float) -> dict:
        """Compute comprehensive gradient statistics."""

        # Flatten for analysis
        flat_grads = patch_gradients.reshape(-1).numpy()
        flat_nll = nll_values.reshape(-1).numpy()
        flat_tail = tail_masks.reshape(-1).numpy()

        # 1. Basic statistics
        results = {
            'tail_weight': tail_weight,
            'grad_mean': float(np.mean(flat_grads)),
            'grad_std': float(np.std(flat_grads)),
            'grad_median': float(np.median(flat_grads)),
            'grad_max': float(np.max(flat_grads)),
            'grad_min': float(np.min(flat_grads)),
            'grad_skewness': float(stats.skew(flat_grads)),
            'grad_kurtosis': float(stats.kurtosis(flat_grads)),
        }

        # 2. Gini coefficient (gradient concentration)
        gini = self._compute_gini(flat_grads)
        results['gini_coefficient'] = float(gini)

        # 3. Effective gradient rank
        eff_rank = self._compute_effective_rank(flat_grads)
        results['effective_rank'] = float(eff_rank)
        results['total_patches'] = len(flat_grads)
        results['rank_ratio'] = float(eff_rank / len(flat_grads))

        # 4. Gradient at tail vs non-tail patches
        tail_mask = flat_tail > 0.5
        grad_at_tail = flat_grads[tail_mask]
        grad_at_nontail = flat_grads[~tail_mask]

        results['grad_tail_mean'] = float(np.mean(grad_at_tail))
        results['grad_tail_std'] = float(np.std(grad_at_tail))
        results['grad_nontail_mean'] = float(np.mean(grad_at_nontail))
        results['grad_nontail_std'] = float(np.std(grad_at_nontail))
        results['grad_tail_nontail_ratio'] = float(
            np.mean(grad_at_tail) / (np.mean(grad_at_nontail) + 1e-10)
        )

        # 5. Gradient-NLL correlation
        correlation, p_value = stats.spearmanr(flat_grads, flat_nll)
        results['grad_nll_correlation'] = float(correlation)
        results['grad_nll_pvalue'] = float(p_value)

        # 6. Layer-wise gradient statistics
        layer_stats = {}
        for name, grads in layer_gradients.items():
            if len(grads) > 0:
                layer_stats[name] = {
                    'mean': float(np.mean(grads)),
                    'std': float(np.std(grads)),
                    'max': float(np.max(grads)),
                }
        results['layer_gradients'] = layer_stats

        # 7. Gradient distribution (histogram data)
        hist, bin_edges = np.histogram(flat_grads, bins=50, density=True)
        results['grad_histogram'] = {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
        }

        return results

    def _compute_gini(self, values: np.ndarray) -> float:
        """
        Compute Gini coefficient for gradient concentration.

        Gini = 0: Perfect equality (all gradients equal)
        Gini = 1: Perfect inequality (all gradient in one patch)
        """
        values = np.abs(values)
        values = values / (values.sum() + 1e-10)  # Normalize
        n = len(values)
        sorted_values = np.sort(values)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_values) / (n * np.sum(sorted_values) + 1e-10)) - (n + 1) / n
        return gini

    def _compute_effective_rank(self, values: np.ndarray) -> float:
        """
        Compute effective rank (entropy-based).

        Higher = more distributed gradients
        Lower = more concentrated gradients
        """
        values = np.abs(values)
        values = values / (values.sum() + 1e-10)
        values = np.clip(values, 1e-10, 1.0)
        entropy = -np.sum(values * np.log(values))
        effective_rank = np.exp(entropy)
        return effective_rank


def run_gradient_analysis(args):
    """Run gradient dynamics analysis experiment."""

    print("=" * 70)
    print("Experiment 3: Gradient Dynamics Analysis for Tail-Aware Loss")
    print("=" * 70)
    print(f"Class: {args.target_class}")
    print(f"Tail ratio: {args.tail_ratio}")
    print(f"Number of batches: {args.num_batches}")
    print()

    # Set seed for reproducibility
    init_seeds(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Create data loader for target class
    print(f"Loading data for class: {args.target_class}...")

    # Create args-like object for create_task_dataset
    class DataArgs:
        def __init__(self, data_path, img_size, dataset='mvtec'):
            self.data_path = data_path
            self.img_size = img_size
            self.msk_size = img_size  # Usually same as img_size
            self.dataset = dataset

    data_args = DataArgs(args.data_path, args.img_size, args.dataset)
    global_class_to_idx = {args.target_class: 0}

    train_dataset = create_task_dataset(
        args=data_args,
        task_classes=[args.target_class],
        global_class_to_idx=global_class_to_idx,
        train=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    print(f"  Loaded {len(train_dataset)} samples")

    # Create feature extractor
    print(f"Creating feature extractor: {args.backbone_name}...")
    backbone_type = get_backbone_type(args.backbone_name)
    input_shape = (3, args.img_size, args.img_size)
    vit_extractor = create_feature_extractor(
        backbone_name=args.backbone_name,
        input_shape=input_shape,
        device=device
    )
    vit_extractor.eval()

    # Get feature dimension
    with torch.no_grad():
        sample_images = next(iter(train_loader))[0][:1].to(device)
        sample_features, spatial_shape = vit_extractor(sample_images, return_spatial_shape=True)
        embed_dim = sample_features.shape[-1]
        print(f"  Feature dimension: {embed_dim}")
        print(f"  Spatial shape: {spatial_shape}")

    # Create positional embedding generator
    pos_embed_generator = PositionalEmbeddingGenerator(device=device)

    # Create ablation config (minimal setup for gradient analysis)
    ablation_config = AblationConfig(
        use_lora=True,
        use_router=False,  # Not needed for gradient analysis
        use_pos_embedding=True,
        use_task_adapter=True,
        use_dia=False,
        use_tail_aware_loss=False,  # We'll manually compute losses
    )

    # Create NF model
    print("Creating NF model...")
    nf_model = MoLESpatialAwareNF(
        embed_dim=embed_dim,
        coupling_layers=args.num_coupling_layers,
        lora_rank=args.lora_rank,
        device=device,
        ablation_config=ablation_config,
    )
    nf_model.add_task(0)  # Add task 0
    nf_model.to(device)

    # Quick pre-training to get non-trivial gradients
    print("Quick pre-training (5 epochs) to get non-trivial gradients...")
    nf_model.train()
    optimizer = torch.optim.AdamW(nf_model.parameters(), lr=1e-3)

    for epoch in range(5):
        epoch_loss = 0
        n_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 10:  # Only 10 batches per epoch
                break
            images = batch[0].to(device)
            with torch.no_grad():
                patch_embeddings, spatial_shape = vit_extractor(images, return_spatial_shape=True)
                B = patch_embeddings.shape[0]
                H, W = spatial_shape
                features = pos_embed_generator(spatial_shape, patch_embeddings)

            nf_model.set_active_task(0)
            z, logdet_patch = nf_model(features)
            D = z.shape[-1]
            log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
            log_px = log_pz + logdet_patch
            loss = -log_px.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f"  Epoch {epoch+1}/5: Loss = {epoch_loss / n_batches:.4f}")

    # Create analyzer
    analyzer = GradientDynamicsAnalyzer(
        nf_model=nf_model,
        vit_extractor=vit_extractor,
        pos_embed_generator=pos_embed_generator,
        device=device,
    )

    # Define configurations to compare
    configurations = [
        {'name': 'mean_only', 'tail_weight': 0.0, 'description': 'Mean-only training (standard NLL)'},
        {'name': 'tail_only', 'tail_weight': 1.0, 'description': 'Tail-only training (focus on hard patches)'},
        {'name': 'balanced', 'tail_weight': 0.7, 'description': 'Balanced training (70% tail weight)'},
    ]

    # Run analysis for each configuration
    all_results = {}
    for config in configurations:
        print(f"\n--- Analyzing: {config['name']} (tail_weight={config['tail_weight']}) ---")
        print(f"    {config['description']}")

        results = analyzer.analyze_gradient_for_configuration(
            train_loader=train_loader,
            task_id=0,
            tail_weight=config['tail_weight'],
            tail_ratio=args.tail_ratio,
            num_batches=args.num_batches,
        )
        all_results[config['name']] = results

    # Print comparison summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n--- Gradient Distribution Statistics ---")
    print(f"{'Configuration':<15} {'Mean':<12} {'Std':<12} {'Skewness':<12} {'Gini':<12}")
    print("-" * 65)
    for name, res in all_results.items():
        print(f"{name:<15} {res['grad_mean']:<12.6f} {res['grad_std']:<12.6f} "
              f"{res['grad_skewness']:<12.4f} {res['gini_coefficient']:<12.4f}")

    print("\n--- Gradient Focus on Tail Patches ---")
    print(f"{'Configuration':<15} {'Grad@Tail':<12} {'Grad@NonTail':<12} {'Ratio':<12} {'Corr(NLL)':<12}")
    print("-" * 65)
    for name, res in all_results.items():
        print(f"{name:<15} {res['grad_tail_mean']:<12.6f} {res['grad_nontail_mean']:<12.6f} "
              f"{res['grad_tail_nontail_ratio']:<12.4f} {res['grad_nll_correlation']:<12.4f}")

    print("\n--- Gradient Concentration (Effective Rank) ---")
    print(f"{'Configuration':<15} {'Eff. Rank':<15} {'Total Patches':<15} {'Rank Ratio':<15}")
    print("-" * 65)
    for name, res in all_results.items():
        print(f"{name:<15} {res['effective_rank']:<15.2f} {res['total_patches']:<15} "
              f"{res['rank_ratio']:<15.4f}")

    # Hypothesis evaluation
    print("\n" + "=" * 70)
    print("HYPOTHESIS EVALUATION")
    print("=" * 70)

    mean_results = all_results['mean_only']
    tail_results = all_results['tail_only']
    balanced_results = all_results['balanced']

    # H3: Mean-only training dilutes gradients from tail patches
    print("\n[H3] Mean-only training dilutes gradients from tail patches:")
    mean_ratio = mean_results['grad_tail_nontail_ratio']
    tail_ratio_val = tail_results['grad_tail_nontail_ratio']

    if mean_ratio < tail_ratio_val:
        print(f"  SUPPORTED: Mean-only gradient ratio ({mean_ratio:.4f}) < Tail gradient ratio ({tail_ratio_val:.4f})")
        print(f"  Interpretation: Mean loss spreads gradients more uniformly, while tail loss concentrates on hard patches.")
    else:
        print(f"  NOT SUPPORTED: Mean-only ratio ({mean_ratio:.4f}) >= Tail ratio ({tail_ratio_val:.4f})")

    # H6: Tail training improves gradient focus
    print("\n[H6] Tail training improves gradient focus on hard patches:")
    mean_corr = mean_results['grad_nll_correlation']
    tail_corr = tail_results['grad_nll_correlation']

    if tail_corr > mean_corr:
        print(f"  SUPPORTED: Tail NLL-gradient correlation ({tail_corr:.4f}) > Mean ({mean_corr:.4f})")
        print(f"  Interpretation: Tail loss creates stronger gradient-difficulty alignment.")
    else:
        print(f"  PARTIALLY SUPPORTED: Correlations similar (Mean: {mean_corr:.4f}, Tail: {tail_corr:.4f})")

    # Additional insight: Gini coefficient
    print("\n[Additional] Gradient Concentration Analysis:")
    mean_gini = mean_results['gini_coefficient']
    tail_gini = tail_results['gini_coefficient']

    if tail_gini > mean_gini:
        print(f"  Tail loss concentrates gradients more (Gini: {tail_gini:.4f}) vs Mean (Gini: {mean_gini:.4f})")
        print(f"  This confirms H3: tail-aware training focuses learning on specific patches.")
    else:
        print(f"  Unexpected: Mean Gini ({mean_gini:.4f}) >= Tail Gini ({tail_gini:.4f})")

    # Save results
    results_file = os.path.join(output_dir, 'gradient_analysis_results.json')
    with open(results_file, 'w') as f:
        # Remove non-serializable layer gradients for JSON
        save_results = {}
        for name, res in all_results.items():
            save_res = {k: v for k, v in res.items() if k != 'layer_gradients' and k != 'grad_histogram'}
            save_results[name] = save_res
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Generate visualizations
    generate_visualizations(all_results, output_dir)

    return all_results


def generate_visualizations(all_results: dict, output_dir: str):
    """Generate comparison visualizations."""

    print("\nGenerating visualizations...")

    # Figure 1: Gradient Distribution Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    configs = ['mean_only', 'tail_only', 'balanced']
    titles = ['Mean-Only (w=0)', 'Tail-Only (w=1)', 'Balanced (w=0.7)']
    colors = ['blue', 'red', 'green']

    for idx, (config, title, color) in enumerate(zip(configs, titles, colors)):
        ax = axes[idx]
        hist_data = all_results[config]['grad_histogram']
        bin_edges = np.array(hist_data['bin_edges'])
        counts = np.array(hist_data['counts'])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax.bar(bin_centers, counts, width=bin_edges[1]-bin_edges[0],
               color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel('Gradient Magnitude')
        ax.set_ylabel('Density')
        ax.set_xlim(0, np.percentile(bin_edges, 99))

        # Add statistics annotation
        stats_text = (f"Mean: {all_results[config]['grad_mean']:.4f}\n"
                     f"Gini: {all_results[config]['gini_coefficient']:.3f}")
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_distribution_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: gradient_distribution_comparison.png")

    # Figure 2: Gradient Focus Comparison (Bar Chart)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 2a: Gradient at Tail vs Non-Tail
    ax = axes[0]
    x = np.arange(len(configs))
    width = 0.35

    tail_grads = [all_results[c]['grad_tail_mean'] for c in configs]
    nontail_grads = [all_results[c]['grad_nontail_mean'] for c in configs]

    bars1 = ax.bar(x - width/2, tail_grads, width, label='Tail Patches', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, nontail_grads, width, label='Non-Tail Patches', color='blue', alpha=0.7)

    ax.set_ylabel('Mean Gradient Magnitude')
    ax.set_title('Gradient Magnitude: Tail vs Non-Tail Patches')
    ax.set_xticks(x)
    ax.set_xticklabels(['Mean-Only', 'Tail-Only', 'Balanced'])
    ax.legend()

    # Add ratio annotations
    for i, config in enumerate(configs):
        ratio = all_results[config]['grad_tail_nontail_ratio']
        ax.annotate(f'Ratio: {ratio:.2f}x', xy=(i, max(tail_grads[i], nontail_grads[i])),
                   xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)

    # 2b: Gini Coefficient and Effective Rank Ratio
    ax = axes[1]

    gini_values = [all_results[c]['gini_coefficient'] for c in configs]
    rank_ratios = [all_results[c]['rank_ratio'] for c in configs]

    bars1 = ax.bar(x - width/2, gini_values, width, label='Gini Coefficient', color='orange', alpha=0.7)
    bars2 = ax.bar(x + width/2, rank_ratios, width, label='Rank Ratio', color='purple', alpha=0.7)

    ax.set_ylabel('Value')
    ax.set_title('Gradient Concentration Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(['Mean-Only', 'Tail-Only', 'Balanced'])
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_focus_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: gradient_focus_comparison.png")

    # Figure 3: NLL-Gradient Correlation
    fig, ax = plt.subplots(figsize=(8, 5))

    correlations = [all_results[c]['grad_nll_correlation'] for c in configs]
    colors = ['blue', 'red', 'green']

    bars = ax.bar(configs, correlations, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Spearman Correlation')
    ax.set_title('Gradient-NLL Correlation\n(Higher = Gradients align with patch difficulty)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, correlations):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_nll_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: gradient_nll_correlation.png")

    # Figure 4: Summary Comparison Table as Image
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Create table data
    columns = ['Metric', 'Mean-Only (w=0)', 'Tail-Only (w=1)', 'Balanced (w=0.7)', 'Winner']
    rows = [
        ['Grad Mean', f"{all_results['mean_only']['grad_mean']:.6f}",
         f"{all_results['tail_only']['grad_mean']:.6f}",
         f"{all_results['balanced']['grad_mean']:.6f}", '-'],
        ['Grad @ Tail', f"{all_results['mean_only']['grad_tail_mean']:.6f}",
         f"{all_results['tail_only']['grad_tail_mean']:.6f}",
         f"{all_results['balanced']['grad_tail_mean']:.6f}",
         'Tail' if all_results['tail_only']['grad_tail_mean'] > all_results['mean_only']['grad_tail_mean'] else 'Mean'],
        ['Tail/NonTail Ratio', f"{all_results['mean_only']['grad_tail_nontail_ratio']:.3f}",
         f"{all_results['tail_only']['grad_tail_nontail_ratio']:.3f}",
         f"{all_results['balanced']['grad_tail_nontail_ratio']:.3f}",
         'Tail' if all_results['tail_only']['grad_tail_nontail_ratio'] > all_results['mean_only']['grad_tail_nontail_ratio'] else 'Mean'],
        ['Gini Coeff', f"{all_results['mean_only']['gini_coefficient']:.4f}",
         f"{all_results['tail_only']['gini_coefficient']:.4f}",
         f"{all_results['balanced']['gini_coefficient']:.4f}",
         'Tail' if all_results['tail_only']['gini_coefficient'] > all_results['mean_only']['gini_coefficient'] else 'Mean'],
        ['NLL Correlation', f"{all_results['mean_only']['grad_nll_correlation']:.4f}",
         f"{all_results['tail_only']['grad_nll_correlation']:.4f}",
         f"{all_results['balanced']['grad_nll_correlation']:.4f}",
         'Tail' if all_results['tail_only']['grad_nll_correlation'] > all_results['mean_only']['grad_nll_correlation'] else 'Mean'],
    ]

    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Color header
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Color winner column
    for i in range(1, len(rows) + 1):
        if rows[i-1][-1] == 'Tail':
            table[(i, 4)].set_facecolor('#FFD7D7')
        elif rows[i-1][-1] == 'Mean':
            table[(i, 4)].set_facecolor('#D7E8FF')

    ax.set_title('Gradient Dynamics: Summary Comparison', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: gradient_summary_table.png")

    print("All visualizations generated!")


def main():
    parser = argparse.ArgumentParser(description='Gradient Dynamics Analysis for Tail-Aware Loss')

    # Data settings
    parser.add_argument('--dataset', type=str, default='mvtec', help='Dataset name')
    parser.add_argument('--data_path', type=str, default='/Data/MVTecAD', help='Dataset path')
    parser.add_argument('--target_class', type=str, default='leather', help='Target class to analyze')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')

    # Model settings
    parser.add_argument('--backbone_name', type=str, default='wide_resnet50_2', help='Backbone model')
    parser.add_argument('--num_coupling_layers', type=int, default=6, help='Number of coupling layers')
    parser.add_argument('--lora_rank', type=int, default=64, help='LoRA rank')

    # Analysis settings
    parser.add_argument('--tail_ratio', type=float, default=0.02, help='Tail patch ratio')
    parser.add_argument('--num_batches', type=int, default=20, help='Number of batches to analyze')
    parser.add_argument('--output_dir', type=str,
                       default='/Volume/MoLeFlow/analysis_results/gradient_dynamics',
                       help='Output directory')

    args = parser.parse_args()

    # Run analysis
    results = run_gradient_analysis(args)

    return results


if __name__ == '__main__':
    main()
