#!/usr/bin/env python
"""
Experiment 1: Tail Patch Spatial Distribution Analysis

This script analyzes WHERE tail patches (top 2% NLL) are located spatially
and what characteristics they have.

Hypotheses:
- H1a: Tail patches are located near image edges
- H1b: Tail patches correspond to high gradient (texture transition) regions
- H1c: Tail patch distribution is non-uniform spatially

Measurements:
1. Spatial heatmap of tail patch frequency
2. Edge distance statistics (KS test vs uniform)
3. Correlation with image gradient magnitude (Mann-Whitney U test)

Usage:
    python scripts/run_spatial_analysis.py --task_class leather --num_epochs 20
"""

import argparse
import math
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.ndimage import sobel
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moleflow import (
    MoLESpatialAwareNF,
    MoLEContinualTrainer,
    PositionalEmbeddingGenerator,
    create_feature_extractor,
    get_backbone_type,
    init_seeds,
    get_config,
    create_task_dataset,
)
from moleflow.config import AblationConfig, parse_ablation_args, add_ablation_args


def analyze_spatial_distribution(
    trainer,
    train_loader,
    tail_ratio: float = 0.02,
    num_batches: int = 50,
    device: str = 'cuda'
):
    """
    Analyze where tail patches are located spatially.

    Returns:
        Dict with spatial distribution analysis results
    """
    print("\n" + "=" * 70)
    print("Experiment 1: Tail Patch Spatial Distribution Analysis")
    print("=" * 70)
    print(f"  Tail ratio: {tail_ratio} (top {tail_ratio*100:.1f}% NLL patches)")

    trainer.nf_model.eval()
    trainer.vit_extractor.eval()

    # Accumulators
    spatial_heatmap = None
    edge_distances = []
    gradient_at_tail = []
    gradient_at_nontail = []
    nll_values_all = []
    position_nll_map = None  # (H, W) accumulator for average NLL at each position
    position_count = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= num_batches:
                break

            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)
            B = images.shape[0]

            # Forward pass
            features, spatial_shape = trainer.vit_extractor(images, return_spatial_shape=True)
            H, W = spatial_shape

            if trainer.use_pos_embedding and trainer.pos_embed_generator is not None:
                features = trainer.pos_embed_generator(spatial_shape, features)
            else:
                features = features.reshape(B, H, W, -1)

            z, logdet_patch = trainer.nf_model(features, task_id=0)
            _, H, W, D = z.shape

            # Compute patch-wise NLL
            log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
            nll_patch = -(log_pz_patch + logdet_patch)  # (B, H, W)

            # Initialize accumulators
            if spatial_heatmap is None:
                spatial_heatmap = torch.zeros(H, W, device=device)
                position_nll_map = torch.zeros(H, W, device=device)
                position_count = torch.zeros(H, W, device=device)

            # Find tail patches
            flat_nll = nll_patch.reshape(B, -1)
            num_patches = H * W
            k = max(1, int(num_patches * tail_ratio))

            for b in range(B):
                _, tail_indices = torch.topk(flat_nll[b], k)
                tail_rows = tail_indices // W
                tail_cols = tail_indices % W

                # Update heatmap
                for r, c in zip(tail_rows.tolist(), tail_cols.tolist()):
                    spatial_heatmap[r, c] += 1

                # Compute edge distance
                for r, c in zip(tail_rows.tolist(), tail_cols.tolist()):
                    dist = min(r, c, H - 1 - r, W - 1 - c)
                    edge_distances.append(dist)

                # Accumulate position-wise NLL
                position_nll_map += nll_patch[b]
                position_count += 1

            # Compute image gradient magnitude
            img_np = images.cpu().numpy()
            for b in range(B):
                # Convert to grayscale
                if img_np.shape[1] == 3:
                    gray = 0.299 * img_np[b, 0] + 0.587 * img_np[b, 1] + 0.114 * img_np[b, 2]
                else:
                    gray = img_np[b, 0]

                # Compute gradient magnitude
                gx = sobel(gray, axis=0)
                gy = sobel(gray, axis=1)
                grad_mag = np.sqrt(gx ** 2 + gy ** 2)

                # Resize to feature map size
                grad_mag_resized = F.interpolate(
                    torch.from_numpy(grad_mag).unsqueeze(0).unsqueeze(0).float(),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=True
                ).squeeze().numpy()

                # Get gradient at tail vs non-tail locations
                _, tail_indices = torch.topk(flat_nll[b], k)
                all_indices = set(range(num_patches))
                tail_set = set(tail_indices.tolist())
                nontail_set = all_indices - tail_set

                for idx in tail_set:
                    r, c = idx // W, idx % W
                    gradient_at_tail.append(grad_mag_resized[r, c])

                # Sample same number from non-tail
                for idx in list(nontail_set)[:k]:
                    r, c = idx // W, idx % W
                    gradient_at_nontail.append(grad_mag_resized[r, c])

                # Store NLL values
                nll_values_all.extend(flat_nll[b].cpu().numpy().tolist())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{num_batches} batches")

    # Normalize heatmap
    spatial_heatmap = spatial_heatmap.cpu().numpy()
    spatial_heatmap_normalized = spatial_heatmap / spatial_heatmap.sum()

    # Average NLL per position
    position_nll_avg = (position_nll_map / position_count).cpu().numpy()

    # Edge distance statistics
    edge_distances = np.array(edge_distances)
    max_dist = min(H, W) // 2

    # Expected uniform distribution distance
    uniform_distances = []
    for r in range(H):
        for c in range(W):
            uniform_distances.append(min(r, c, H - 1 - r, W - 1 - c))
    uniform_distances = np.array(uniform_distances)

    # KS test: tail vs uniform
    ks_stat, ks_pvalue = stats.ks_2samp(edge_distances, uniform_distances)

    # Gradient correlation
    gradient_at_tail = np.array(gradient_at_tail)
    gradient_at_nontail = np.array(gradient_at_nontail)

    # Mann-Whitney U test: tail gradient vs non-tail gradient
    mw_stat, mw_pvalue = stats.mannwhitneyu(
        gradient_at_tail, gradient_at_nontail, alternative='greater'
    )

    # Additional statistics
    # Chi-square test for spatial uniformity
    expected_uniform = np.ones_like(spatial_heatmap_normalized) / (H * W)
    # Normalize to have same total for chi-square
    observed_counts = (spatial_heatmap_normalized * 1000).astype(int)
    expected_counts = (expected_uniform * 1000).astype(int)
    # Flatten for chi-square
    chi2_stat, chi2_pvalue = stats.chisquare(
        observed_counts.flatten() + 1,  # Add 1 to avoid zeros
        expected_counts.flatten() + 1
    )

    # Corner vs center analysis
    center_h, center_w = H // 2, W // 2
    corner_radius = min(H, W) // 4

    corner_mask = np.zeros((H, W), dtype=bool)
    corner_mask[:corner_radius, :corner_radius] = True
    corner_mask[:corner_radius, -corner_radius:] = True
    corner_mask[-corner_radius:, :corner_radius] = True
    corner_mask[-corner_radius:, -corner_radius:] = True

    center_mask = np.zeros((H, W), dtype=bool)
    center_mask[
        center_h - corner_radius:center_h + corner_radius,
        center_w - corner_radius:center_w + corner_radius
    ] = True

    corner_freq = spatial_heatmap_normalized[corner_mask].mean()
    center_freq = spatial_heatmap_normalized[center_mask].mean()
    corner_center_ratio = corner_freq / (center_freq + 1e-8)

    results = {
        'spatial_heatmap': spatial_heatmap,
        'spatial_heatmap_normalized': spatial_heatmap_normalized,
        'position_nll_avg': position_nll_avg,
        'edge_distance_mean': edge_distances.mean(),
        'edge_distance_std': edge_distances.std(),
        'edge_distance_median': np.median(edge_distances),
        'uniform_distance_mean': uniform_distances.mean(),
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'gradient_tail_mean': gradient_at_tail.mean(),
        'gradient_tail_std': gradient_at_tail.std(),
        'gradient_nontail_mean': gradient_at_nontail.mean(),
        'gradient_nontail_std': gradient_at_nontail.std(),
        'gradient_ratio': gradient_at_tail.mean() / (gradient_at_nontail.mean() + 1e-8),
        'mann_whitney_stat': mw_stat,
        'mann_whitney_pvalue': mw_pvalue,
        'chi2_statistic': chi2_stat,
        'chi2_pvalue': chi2_pvalue,
        'corner_freq': corner_freq,
        'center_freq': center_freq,
        'corner_center_ratio': corner_center_ratio,
        'H': H,
        'W': W,
        'tail_ratio': tail_ratio,
        'num_tail_patches': len(edge_distances),
    }

    return results


def plot_spatial_analysis(results, save_path):
    """Generate comprehensive visualization of spatial analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    H, W = results['H'], results['W']

    # 1. Raw heatmap (frequency)
    ax = axes[0, 0]
    im = ax.imshow(results['spatial_heatmap'], cmap='hot', interpolation='nearest')
    ax.set_title(f'Tail Patch Frequency (raw counts)\n{results["num_tail_patches"]} tail patches')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='Count')

    # 2. Normalized heatmap (probability)
    ax = axes[0, 1]
    im = ax.imshow(results['spatial_heatmap_normalized'], cmap='hot', interpolation='nearest')
    ax.set_title('Tail Patch Probability Distribution')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='Probability')

    # 3. Average NLL per position
    ax = axes[0, 2]
    im = ax.imshow(results['position_nll_avg'], cmap='viridis', interpolation='nearest')
    ax.set_title('Average NLL per Spatial Position')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='Avg NLL')

    # 4. Edge distance comparison
    ax = axes[1, 0]
    categories = ['Tail\nPatches', 'Uniform\nExpected']
    means = [results['edge_distance_mean'], results['uniform_distance_mean']]
    stds = [results['edge_distance_std'], 0]
    colors = ['#e74c3c', '#3498db']

    bars = ax.bar(categories, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel('Mean Distance to Edge (patches)')
    ax.set_title(f'Edge Distance Analysis\nKS test: stat={results["ks_statistic"]:.3f}, p={results["ks_pvalue"]:.2e}')

    # Add significance annotation
    if results['ks_pvalue'] < 0.05:
        sig_text = "***" if results['ks_pvalue'] < 0.001 else "**" if results['ks_pvalue'] < 0.01 else "*"
        ax.annotate(sig_text, xy=(0.5, max(means) * 1.1), fontsize=16, ha='center')

    # 5. Gradient magnitude comparison
    ax = axes[1, 1]
    categories = ['Tail\nPatches', 'Non-Tail\nPatches']
    means = [results['gradient_tail_mean'], results['gradient_nontail_mean']]
    stds = [results['gradient_tail_std'], results['gradient_nontail_std']]
    colors = ['#e74c3c', '#2ecc71']

    bars = ax.bar(categories, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel('Image Gradient Magnitude')
    ax.set_title(f'Gradient Analysis\nMW test: stat={results["mann_whitney_stat"]:.0f}, p={results["mann_whitney_pvalue"]:.2e}')
    ax.text(0.5, max(means) * 0.9, f'Ratio: {results["gradient_ratio"]:.2f}x',
            ha='center', fontsize=10, transform=ax.get_xaxis_transform())

    # 6. Corner vs Center frequency
    ax = axes[1, 2]
    categories = ['Corner\nRegions', 'Center\nRegion']
    freqs = [results['corner_freq'] * 1000, results['center_freq'] * 1000]  # Scale for visibility
    colors = ['#9b59b6', '#f39c12']

    bars = ax.bar(categories, freqs, color=colors, alpha=0.8)
    ax.set_ylabel('Relative Frequency (x1000)')
    ax.set_title(f'Corner vs Center Analysis\nRatio: {results["corner_center_ratio"]:.2f}')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to {save_path}")


def print_results_summary(results):
    """Print formatted results summary."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n[Spatial Configuration]")
    print(f"  Feature map size: {results['H']} x {results['W']}")
    print(f"  Tail ratio: {results['tail_ratio']} ({results['tail_ratio']*100:.1f}%)")
    print(f"  Total tail patches analyzed: {results['num_tail_patches']}")

    print(f"\n[H1a: Edge Distance Analysis]")
    print(f"  Tail patches edge distance: {results['edge_distance_mean']:.2f} +/- {results['edge_distance_std']:.2f}")
    print(f"  Median edge distance: {results['edge_distance_median']:.2f}")
    print(f"  Uniform expected distance: {results['uniform_distance_mean']:.2f}")
    print(f"  KS test statistic: {results['ks_statistic']:.4f}")
    print(f"  KS test p-value: {results['ks_pvalue']:.4e}")

    if results['ks_pvalue'] < 0.05:
        if results['edge_distance_mean'] < results['uniform_distance_mean']:
            print(f"  --> VERDICT: H1a SUPPORTED (tail patches CLOSER to edges, p < 0.05)")
        else:
            print(f"  --> VERDICT: H1a REJECTED (tail patches FARTHER from edges)")
    else:
        print(f"  --> VERDICT: H1a INCONCLUSIVE (distribution similar to uniform)")

    print(f"\n[H1b: Gradient Correlation Analysis]")
    print(f"  Gradient at tail patches: {results['gradient_tail_mean']:.4f} +/- {results['gradient_tail_std']:.4f}")
    print(f"  Gradient at non-tail patches: {results['gradient_nontail_mean']:.4f} +/- {results['gradient_nontail_std']:.4f}")
    print(f"  Gradient ratio (tail/non-tail): {results['gradient_ratio']:.4f}")
    print(f"  Mann-Whitney U statistic: {results['mann_whitney_stat']:.0f}")
    print(f"  Mann-Whitney p-value: {results['mann_whitney_pvalue']:.4e}")

    if results['mann_whitney_pvalue'] < 0.05 and results['gradient_ratio'] > 1.0:
        print(f"  --> VERDICT: H1b SUPPORTED (tail patches have HIGHER gradient, p < 0.05)")
    elif results['mann_whitney_pvalue'] < 0.05:
        print(f"  --> VERDICT: H1b REJECTED (tail patches have LOWER gradient)")
    else:
        print(f"  --> VERDICT: H1b INCONCLUSIVE (no significant gradient difference)")

    print(f"\n[H1c: Spatial Uniformity Analysis]")
    print(f"  Chi-square statistic: {results['chi2_statistic']:.2f}")
    print(f"  Chi-square p-value: {results['chi2_pvalue']:.4e}")
    print(f"  Corner frequency (normalized): {results['corner_freq']:.6f}")
    print(f"  Center frequency (normalized): {results['center_freq']:.6f}")
    print(f"  Corner/Center ratio: {results['corner_center_ratio']:.4f}")

    if results['chi2_pvalue'] < 0.05:
        print(f"  --> VERDICT: H1c SUPPORTED (spatial distribution is NON-UNIFORM, p < 0.05)")
    else:
        print(f"  --> VERDICT: H1c REJECTED (spatial distribution similar to uniform)")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Overall interpretation
    edge_effect = results['ks_pvalue'] < 0.05 and results['edge_distance_mean'] < results['uniform_distance_mean']
    gradient_effect = results['mann_whitney_pvalue'] < 0.05 and results['gradient_ratio'] > 1.2
    non_uniform = results['chi2_pvalue'] < 0.05

    if edge_effect and gradient_effect:
        print("  STRONG EVIDENCE: Tail patches correspond to boundary/transition regions")
        print("  - They are located closer to image edges")
        print("  - They have higher gradient magnitude (texture transitions)")
        print("  - This suggests tail-aware loss helps model edge/boundary features")
    elif gradient_effect:
        print("  MODERATE EVIDENCE: Tail patches correspond to texture transition regions")
        print("  - Higher gradient magnitude in tail patches")
        print("  - May represent complex textures or boundaries in content")
    elif edge_effect:
        print("  MODERATE EVIDENCE: Tail patches are edge-biased")
        print("  - Located closer to image boundaries")
        print("  - May capture edge artifacts or boundary conditions")
    elif non_uniform:
        print("  WEAK EVIDENCE: Tail patches have non-uniform spatial distribution")
        print("  - Some spatial bias exists but mechanism unclear")
    else:
        print("  NO CLEAR PATTERN: Tail patches appear uniformly distributed")
        print("  - Tail-aware loss may work through non-spatial mechanisms")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Tail Patch Spatial Distribution Analysis')
    parser.add_argument('--task_class', type=str, default='leather',
                        help='Class to analyze')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs for quick training')
    parser.add_argument('--num_batches', type=int, default=50,
                        help='Number of batches to analyze')
    parser.add_argument('--tail_ratio', type=float, default=0.02,
                        help='Tail ratio (top K%% of NLL patches)')
    parser.add_argument('--data_path', type=str, default='/Data/MVTecAD',
                        help='Path to MVTec dataset')
    parser.add_argument('--backbone_name', type=str, default='wide_resnet50_2',
                        help='Backbone model name')
    parser.add_argument('--output_dir', type=str, default='/Volume/MoLeFlow/analysis_results/spatial_distribution',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training (use random weights for testing)')

    # Add ablation args for tail-aware loss settings
    add_ablation_args(parser)

    args = parser.parse_args()

    # Initialize
    init_seeds(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup configuration
    config = get_config(
        img_size=224,
        msk_size=256,
        data_path=args.data_path,
        batch_size=args.batch_size,
        seed=args.seed,
        lr=3e-4,
    )
    config.dataset = 'mvtec'

    # Parse ablation config - use tail-aware loss by default
    ablation_config = parse_ablation_args(args)
    ablation_config.use_tail_aware_loss = True
    ablation_config.tail_weight = 0.7
    ablation_config.tail_top_k_ratio = args.tail_ratio

    print(f"\n{'='*70}")
    print("Tail Patch Spatial Distribution Analysis")
    print(f"{'='*70}")
    print(f"  Class: {args.task_class}")
    print(f"  Tail ratio: {args.tail_ratio}")
    print(f"  Backbone: {args.backbone_name}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"{'='*70}")

    # Initialize feature extractor
    backbone_type = get_backbone_type(args.backbone_name)
    embed_dim = 768 if 'wide_resnet50' in args.backbone_name else 768

    feature_extractor = create_feature_extractor(
        backbone_name=args.backbone_name,
        input_shape=(3, 224, 224),
        target_embed_dimension=embed_dim,
        device=device,
        blocks_to_extract=[1, 3, 5, 11] if backbone_type == 'vit' else None,
        remove_cls_token=True,
        patch_size=3,
        patch_stride=1,
    )
    print(f"Feature extractor initialized: {args.backbone_name}")

    # Initialize position embedding
    pos_embed_generator = PositionalEmbeddingGenerator(device=device)

    # Initialize NF model
    nf_model = MoLESpatialAwareNF(
        embed_dim=embed_dim,
        coupling_layers=6,
        clamp_alpha=1.9,
        lora_rank=64,
        lora_alpha=1.0,
        device=device,
        ablation_config=ablation_config
    )
    print("NF model initialized")

    # Initialize trainer
    trainer = MoLEContinualTrainer(
        vit_extractor=feature_extractor,
        pos_embed_generator=pos_embed_generator,
        nf_model=nf_model,
        args=config,
        device=device,
        ablation_config=ablation_config
    )

    # Create dataset
    task_classes = [args.task_class]
    class_to_idx = {args.task_class: 0}

    config.class_to_idx = class_to_idx
    config.n_classes = 1

    train_dataset = create_task_dataset(config, task_classes, class_to_idx, train=True)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=False, drop_last=True
    )
    print(f"Dataset loaded: {len(train_dataset)} samples")

    # Quick training
    if not args.skip_training:
        print(f"\nTraining for {args.num_epochs} epochs...")
        trainer.train_task(
            task_id=0,
            task_classes=task_classes,
            train_loader=train_loader,
            num_epochs=args.num_epochs,
            lr=3e-4,
            log_interval=20
        )
    else:
        print("\nSkipping training (using random weights)")
        nf_model.add_task(0)

    # Run spatial distribution analysis
    print("\nRunning spatial distribution analysis...")
    results = analyze_spatial_distribution(
        trainer=trainer,
        train_loader=train_loader,
        tail_ratio=args.tail_ratio,
        num_batches=args.num_batches,
        device=device
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save plot
    plot_path = os.path.join(args.output_dir, f'spatial_analysis_{args.task_class}_{timestamp}.png')
    plot_spatial_analysis(results, plot_path)

    # Print summary
    print_results_summary(results)

    # Save numerical results to file
    results_file = os.path.join(args.output_dir, f'spatial_analysis_{args.task_class}_{timestamp}.txt')
    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TAIL PATCH SPATIAL DISTRIBUTION ANALYSIS RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Class: {args.task_class}\n")
        f.write(f"Tail ratio: {args.tail_ratio}\n")
        f.write(f"Training epochs: {args.num_epochs}\n\n")

        f.write("[Spatial Configuration]\n")
        f.write(f"  Feature map size: {results['H']} x {results['W']}\n")
        f.write(f"  Total tail patches: {results['num_tail_patches']}\n\n")

        f.write("[H1a: Edge Distance Analysis]\n")
        f.write(f"  Tail edge distance: {results['edge_distance_mean']:.4f} +/- {results['edge_distance_std']:.4f}\n")
        f.write(f"  Uniform expected: {results['uniform_distance_mean']:.4f}\n")
        f.write(f"  KS statistic: {results['ks_statistic']:.4f}\n")
        f.write(f"  KS p-value: {results['ks_pvalue']:.4e}\n\n")

        f.write("[H1b: Gradient Analysis]\n")
        f.write(f"  Gradient tail: {results['gradient_tail_mean']:.4f} +/- {results['gradient_tail_std']:.4f}\n")
        f.write(f"  Gradient non-tail: {results['gradient_nontail_mean']:.4f} +/- {results['gradient_nontail_std']:.4f}\n")
        f.write(f"  Gradient ratio: {results['gradient_ratio']:.4f}\n")
        f.write(f"  MW statistic: {results['mann_whitney_stat']:.0f}\n")
        f.write(f"  MW p-value: {results['mann_whitney_pvalue']:.4e}\n\n")

        f.write("[H1c: Spatial Uniformity]\n")
        f.write(f"  Chi2 statistic: {results['chi2_statistic']:.4f}\n")
        f.write(f"  Chi2 p-value: {results['chi2_pvalue']:.4e}\n")
        f.write(f"  Corner/Center ratio: {results['corner_center_ratio']:.4f}\n")

    print(f"\nResults saved to: {results_file}")
    print(f"Plot saved to: {plot_path}")

    # Return results for programmatic use
    return results


if __name__ == '__main__':
    main()
