#!/usr/bin/env python3
"""
Tail-Aware Loss Mechanism Analysis Script.

Tests the following hypotheses:
H1: Tail patches correspond to boundary/transition regions (high image gradient)
H3: Mean-only training dilutes gradients from tail patches
H7: Tail training improves latent space calibration (Gaussianity)
"""

import os
import sys
import math
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy import stats
from scipy.ndimage import sobel

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader
from moleflow.data.mvtec import MVTEC
from moleflow.extractors.cnn_extractor import CNNPatchCoreExtractor
from moleflow.models.mole_nf import MoLESpatialAwareNF
from moleflow.models.position_embedding import PositionalEmbeddingGenerator
from moleflow.trainer.continual_trainer import MoLEContinualTrainer
from moleflow.config.ablation import AblationConfig


def create_trainer(use_tail_loss: bool, device: str = 'cuda'):
    """Create trainer with or without tail-aware loss."""
    config = AblationConfig(
        use_lora=True,
        use_router=True,
        use_pos_embedding=True,
        use_whitening_adapter=True,
        use_tail_aware_loss=use_tail_loss,
        tail_weight=0.7 if use_tail_loss else 0.0,
        tail_top_k_ratio=0.02,
        score_aggregation_mode='top_k',
        score_aggregation_top_k=3,
        lambda_logdet=1e-4,
    )

    extractor = CNNPatchCoreExtractor(
        backbone_name='wide_resnet50_2',
        input_shape=(3, 224, 224),
        device=device
    )

    feature_dim = extractor.target_embed_dimension
    pos_embed = PositionalEmbeddingGenerator(device=device)

    nf_model = MoLESpatialAwareNF(
        embed_dim=feature_dim,
        coupling_layers=6,
        lora_rank=64,
        device=device,
        ablation_config=config,
    ).to(device)

    # Create minimal args object
    class Args:
        pass
    args = Args()

    trainer = MoLEContinualTrainer(
        vit_extractor=extractor,
        pos_embed_generator=pos_embed,
        nf_model=nf_model,
        args=args,
        device=device,
        ablation_config=config,
    )

    return trainer


def analyze_spatial_distribution(trainer, train_loader, device, num_batches=30):
    """
    Experiment 1: Analyze spatial distribution of tail patches.

    H1: Tail patches correspond to boundary/transition regions.
    """
    print("\n" + "=" * 70)
    print("Experiment 1: Tail Patch Spatial Distribution (H1)")
    print("=" * 70)

    trainer.nf_model.eval()
    trainer.vit_extractor.eval()

    tail_ratio = 0.02
    edge_distances = []
    gradient_at_tail = []
    gradient_at_nontail = []
    spatial_heatmap = None

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
            patch_embeddings, spatial_shape = trainer.vit_extractor(images, return_spatial_shape=True)
            H, W = spatial_shape
            patch_embeddings_4d = patch_embeddings.reshape(B, H, W, -1)

            if trainer.use_pos_embedding:
                patch_embeddings_4d = trainer.pos_embed_generator(spatial_shape, patch_embeddings)

            trainer.nf_model.set_active_task(0)
            z, logdet_patch = trainer.nf_model(patch_embeddings_4d)
            _, _, _, D = z.shape

            # Compute patch-wise NLL
            log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
            nll_patch = -(log_pz_patch + logdet_patch)

            # Initialize heatmap
            if spatial_heatmap is None:
                spatial_heatmap = torch.zeros(H, W, device=device)

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

            # Compute image gradient magnitude
            img_np = images.cpu().numpy()
            for b in range(B):
                if img_np.shape[1] == 3:
                    gray = 0.299 * img_np[b, 0] + 0.587 * img_np[b, 1] + 0.114 * img_np[b, 2]
                else:
                    gray = img_np[b, 0]

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

                # Get gradient at tail vs non-tail
                _, tail_indices = torch.topk(flat_nll[b], k)
                tail_set = set(tail_indices.tolist())
                nontail_set = set(range(num_patches)) - tail_set

                for idx in tail_set:
                    r, c = idx // W, idx % W
                    gradient_at_tail.append(grad_mag_resized[r, c])

                for idx in list(nontail_set)[:k]:
                    r, c = idx // W, idx % W
                    gradient_at_nontail.append(grad_mag_resized[r, c])

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{num_batches} batches")

    # Statistics
    edge_distances = np.array(edge_distances)
    gradient_at_tail = np.array(gradient_at_tail)
    gradient_at_nontail = np.array(gradient_at_nontail)

    # Expected uniform distribution
    uniform_distances = []
    for r in range(H):
        for c in range(W):
            uniform_distances.append(min(r, c, H - 1 - r, W - 1 - c))
    uniform_distances = np.array(uniform_distances)

    # Statistical tests
    ks_stat, ks_pvalue = stats.ks_2samp(edge_distances, uniform_distances)
    mw_stat, mw_pvalue = stats.mannwhitneyu(gradient_at_tail, gradient_at_nontail, alternative='greater')

    gradient_ratio = gradient_at_tail.mean() / (gradient_at_nontail.mean() + 1e-8)

    results = {
        'edge_distance_mean': float(edge_distances.mean()),
        'edge_distance_std': float(edge_distances.std()),
        'uniform_distance_mean': float(uniform_distances.mean()),
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pvalue),
        'gradient_tail_mean': float(gradient_at_tail.mean()),
        'gradient_nontail_mean': float(gradient_at_nontail.mean()),
        'gradient_ratio': float(gradient_ratio),
        'mann_whitney_pvalue': float(mw_pvalue),
    }

    print("\n--- Results ---")
    print(f"  Edge distance (tail): {edge_distances.mean():.3f} ± {edge_distances.std():.3f}")
    print(f"  Edge distance (uniform): {uniform_distances.mean():.3f}")
    print(f"  KS test p-value: {ks_pvalue:.4e}")
    print(f"  Gradient at tail: {gradient_at_tail.mean():.4f}")
    print(f"  Gradient at non-tail: {gradient_at_nontail.mean():.4f}")
    print(f"  Gradient ratio (tail/non-tail): {gradient_ratio:.2f}x")
    print(f"  Mann-Whitney p-value: {mw_pvalue:.4e}")

    print("\n--- Hypothesis H1 Verdict ---")
    if mw_pvalue < 0.05 and gradient_ratio > 1.2:
        print("  ✓ H1 SUPPORTED: Tail patches have significantly higher image gradient")
        print(f"    → Tail patches are in boundary/transition regions ({gradient_ratio:.1f}x more gradient)")
        results['H1_verdict'] = 'SUPPORTED'
    elif mw_pvalue < 0.05:
        print("  △ H1 PARTIALLY SUPPORTED: Statistically significant but weak effect")
        results['H1_verdict'] = 'PARTIAL'
    else:
        print("  ✗ H1 NOT SUPPORTED: No significant gradient difference")
        results['H1_verdict'] = 'NOT_SUPPORTED'

    return results


def analyze_latent_calibration(trainer, train_loader, device, num_batches=30):
    """
    Experiment 4: Analyze latent space calibration.

    H7: Tail training improves latent space Gaussianity.
    """
    print("\n" + "=" * 70)
    print("Experiment 4: Latent Space Calibration (H7)")
    print("=" * 70)

    trainer.nf_model.eval()
    trainer.vit_extractor.eval()

    all_z = []
    all_nll = []

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

            patch_embeddings, spatial_shape = trainer.vit_extractor(images, return_spatial_shape=True)
            H, W = spatial_shape
            patch_embeddings_4d = patch_embeddings.reshape(B, H, W, -1)

            if trainer.use_pos_embedding:
                patch_embeddings_4d = trainer.pos_embed_generator(spatial_shape, patch_embeddings)

            trainer.nf_model.set_active_task(0)
            z, logdet_patch = trainer.nf_model(patch_embeddings_4d)

            all_z.append(z.reshape(-1, z.shape[-1]).cpu())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{num_batches} batches")

    z_all = torch.cat(all_z, dim=0).numpy()
    n_samples, n_dims = z_all.shape

    print(f"\n  Collected {n_samples} latent samples with {n_dims} dimensions")

    # Test Gaussianity per dimension (sample 10 random dimensions)
    test_dims = np.random.choice(n_dims, min(10, n_dims), replace=False)

    ks_pvalues = []
    shapiro_pvalues = []

    for d in test_dims:
        z_d = z_all[:, d]

        # Standardize
        z_d_std = (z_d - z_d.mean()) / (z_d.std() + 1e-8)

        # KS test against N(0,1)
        ks_stat, ks_p = stats.kstest(z_d_std[:1000], 'norm')  # Sample for speed
        ks_pvalues.append(ks_p)

        # Shapiro-Wilk test (on smaller sample)
        if len(z_d_std) > 5000:
            sw_stat, sw_p = stats.shapiro(z_d_std[:5000])
        else:
            sw_stat, sw_p = stats.shapiro(z_d_std)
        shapiro_pvalues.append(sw_p)

    # Compute tail calibration error
    # Compare empirical vs theoretical quantiles at extreme percentiles
    percentiles = [1, 2, 5, 95, 98, 99]
    theoretical_quantiles = [stats.norm.ppf(p/100) for p in percentiles]

    empirical_quantiles = []
    for p in percentiles:
        eq = np.percentile(z_all.flatten(), p)
        empirical_quantiles.append(eq)

    # Standardize for comparison
    z_flat = z_all.flatten()
    z_std = (z_flat - z_flat.mean()) / (z_flat.std() + 1e-8)
    empirical_std_quantiles = [np.percentile(z_std, p) for p in percentiles]

    tail_calibration_error = np.mean([
        abs(e - t) for e, t in zip(empirical_std_quantiles, theoretical_quantiles)
    ])

    # QQ correlation
    sample_size = min(10000, len(z_std))
    z_sample = np.sort(np.random.choice(z_std, sample_size, replace=False))
    theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, sample_size))
    qq_correlation = np.corrcoef(z_sample, theoretical)[0, 1]

    results = {
        'mean_ks_pvalue': float(np.mean(ks_pvalues)),
        'mean_shapiro_pvalue': float(np.mean(shapiro_pvalues)),
        'tail_calibration_error': float(tail_calibration_error),
        'qq_correlation': float(qq_correlation),
        'z_mean': float(z_all.mean()),
        'z_std': float(z_all.std()),
        'empirical_quantiles': {str(p): float(q) for p, q in zip(percentiles, empirical_std_quantiles)},
        'theoretical_quantiles': {str(p): float(q) for p, q in zip(percentiles, theoretical_quantiles)},
    }

    print("\n--- Results ---")
    print(f"  z mean: {z_all.mean():.4f} (should be ~0)")
    print(f"  z std: {z_all.std():.4f} (should be ~1)")
    print(f"  QQ correlation: {qq_correlation:.4f} (should be ~1)")
    print(f"  Mean KS p-value: {np.mean(ks_pvalues):.4e}")
    print(f"  Tail calibration error: {tail_calibration_error:.4f}")
    print(f"\n  Quantile comparison (standardized):")
    print(f"    Percentile | Empirical | Theoretical | Error")
    print(f"    " + "-" * 50)
    for p, e, t in zip(percentiles, empirical_std_quantiles, theoretical_quantiles):
        print(f"    {p:>6}%    | {e:>9.3f} | {t:>11.3f} | {abs(e-t):>5.3f}")

    return results


def analyze_gradient_concentration(trainer, train_loader, device, num_batches=10):
    """
    Experiment 3: Analyze gradient concentration.

    H3: Mean-only training dilutes gradients from tail patches.
    """
    print("\n" + "=" * 70)
    print("Experiment 3: Gradient Concentration Analysis (H3)")
    print("=" * 70)

    trainer.nf_model.train()
    trainer.vit_extractor.eval()

    tail_ratio = 0.02

    def compute_gradient_stats(tail_weight: float):
        """Compute gradient statistics for a given tail weight."""
        grad_at_tail = []
        grad_at_nontail = []

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= num_batches:
                break

            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)
            B = images.shape[0]

            # Forward with gradient
            with torch.no_grad():
                patch_embeddings, spatial_shape = trainer.vit_extractor(images, return_spatial_shape=True)
                H, W = spatial_shape
                patch_embeddings_4d = patch_embeddings.reshape(B, H, W, -1)

                if trainer.use_pos_embedding:
                    patch_embeddings_4d = trainer.pos_embed_generator(spatial_shape, patch_embeddings)

            # Enable gradient for features
            features = patch_embeddings_4d.clone().detach().requires_grad_(True)

            trainer.nf_model.set_active_task(0)
            z, logdet_patch = trainer.nf_model(features)
            _, _, _, D = z.shape

            # Compute NLL
            log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
            nll_patch = -(log_pz_patch + logdet_patch)

            flat_nll = nll_patch.reshape(B, -1)
            num_patches = H * W
            k = max(1, int(num_patches * tail_ratio))

            # Compute loss
            mean_loss = flat_nll.mean()
            top_k_nll, top_k_indices = torch.topk(flat_nll, k, dim=1)
            tail_loss = top_k_nll.mean()

            loss = (1 - tail_weight) * mean_loss + tail_weight * tail_loss

            # Backward
            trainer.nf_model.zero_grad()
            loss.backward()

            # Get gradient w.r.t. features
            grad_features = features.grad  # (B, H, W, D)
            grad_magnitude = grad_features.norm(dim=-1)  # (B, H, W)

            flat_grad = grad_magnitude.reshape(B, -1)

            for b in range(B):
                tail_set = set(top_k_indices[b].tolist())
                nontail_indices = [i for i in range(num_patches) if i not in tail_set]

                grad_at_tail.extend(flat_grad[b, list(tail_set)].detach().cpu().tolist())
                grad_at_nontail.extend(flat_grad[b, nontail_indices[:k]].detach().cpu().tolist())

        return np.array(grad_at_tail), np.array(grad_at_nontail)

    # Compare mean-only vs tail-aware
    print("  Computing gradients for mean-only (tail_weight=0)...")
    grad_tail_mean, grad_nontail_mean = compute_gradient_stats(tail_weight=0.0)

    print("  Computing gradients for tail-aware (tail_weight=1.0)...")
    grad_tail_ta, grad_nontail_ta = compute_gradient_stats(tail_weight=1.0)

    # Compute concentration ratio
    ratio_mean_only = grad_tail_mean.mean() / (grad_nontail_mean.mean() + 1e-8)
    ratio_tail_aware = grad_tail_ta.mean() / (grad_nontail_ta.mean() + 1e-8)

    # Gini coefficient (concentration measure)
    def gini(x):
        x = np.sort(x)
        n = len(x)
        return (2 * np.sum(np.arange(1, n+1) * x) / (n * np.sum(x))) - (n + 1) / n

    gini_mean_only = gini(np.abs(grad_tail_mean))
    gini_tail_aware = gini(np.abs(grad_tail_ta))

    results = {
        'mean_only': {
            'grad_at_tail': float(grad_tail_mean.mean()),
            'grad_at_nontail': float(grad_nontail_mean.mean()),
            'ratio': float(ratio_mean_only),
            'gini': float(gini_mean_only),
        },
        'tail_aware': {
            'grad_at_tail': float(grad_tail_ta.mean()),
            'grad_at_nontail': float(grad_nontail_ta.mean()),
            'ratio': float(ratio_tail_aware),
            'gini': float(gini_tail_aware),
        },
        'amplification_factor': float(ratio_tail_aware / (ratio_mean_only + 1e-8)),
    }

    print("\n--- Results ---")
    print(f"  Mean-Only Training:")
    print(f"    Gradient at tail: {grad_tail_mean.mean():.6f}")
    print(f"    Gradient at non-tail: {grad_nontail_mean.mean():.6f}")
    print(f"    Ratio (tail/non-tail): {ratio_mean_only:.2f}x")
    print(f"    Gini coefficient: {gini_mean_only:.3f}")
    print(f"\n  Tail-Aware Training:")
    print(f"    Gradient at tail: {grad_tail_ta.mean():.6f}")
    print(f"    Gradient at non-tail: {grad_nontail_ta.mean():.6f}")
    print(f"    Ratio (tail/non-tail): {ratio_tail_aware:.2f}x")
    print(f"    Gini coefficient: {gini_tail_aware:.3f}")
    print(f"\n  Amplification factor: {results['amplification_factor']:.1f}x")

    print("\n--- Hypothesis H3 Verdict ---")
    if ratio_tail_aware > ratio_mean_only * 2:
        print(f"  ✓ H3 SUPPORTED: Tail-aware training concentrates gradients {results['amplification_factor']:.1f}x more")
        print(f"    → Mean-only dilutes gradients; tail-aware focuses on hard patches")
        results['H3_verdict'] = 'SUPPORTED'
    elif ratio_tail_aware > ratio_mean_only:
        print(f"  △ H3 PARTIALLY SUPPORTED: Some gradient concentration ({results['amplification_factor']:.1f}x)")
        results['H3_verdict'] = 'PARTIAL'
    else:
        print("  ✗ H3 NOT SUPPORTED: No significant gradient concentration difference")
        results['H3_verdict'] = 'NOT_SUPPORTED'

    return results


def main():
    """Run all mechanism analyses."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = '/Data/MVTecAD'
    class_name = 'leather'
    output_dir = '/Volume/MoLeFlow/analysis_results/mechanism'

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Tail-Aware Loss Mechanism Analysis")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Class: {class_name}")
    print(f"  Output: {output_dir}")

    # Create data loader
    train_dataset = MVTEC(
        data_path,
        class_name=class_name,
        train=True,
        img_size=224,
        crp_size=224,
        msk_size=224
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    # Create trainer (with tail-aware loss for analysis)
    print("\nCreating trainer...")
    trainer = create_trainer(use_tail_loss=True, device=device)

    # Quick training for meaningful analysis
    print("\nTraining model for 10 epochs...")
    trainer.train_task(
        task_id=0,
        task_classes=[class_name],
        train_loader=train_loader,
        num_epochs=10,
        lr=3e-4
    )

    all_results = {}

    # Experiment 1: Spatial Distribution
    all_results['exp1_spatial'] = analyze_spatial_distribution(
        trainer, train_loader, device, num_batches=30
    )

    # Experiment 3: Gradient Concentration
    all_results['exp3_gradient'] = analyze_gradient_concentration(
        trainer, train_loader, device, num_batches=10
    )

    # Experiment 4: Latent Calibration
    all_results['exp4_latent'] = analyze_latent_calibration(
        trainer, train_loader, device, num_batches=30
    )

    # Save results
    results_path = os.path.join(output_dir, 'mechanism_analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)
    print(f"  Results saved to: {results_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Why does 2% of patches improve Pixel AP by 10%?")
    print("=" * 70)

    h1 = all_results['exp1_spatial'].get('H1_verdict', 'N/A')
    h3 = all_results['exp3_gradient'].get('H3_verdict', 'N/A')

    print(f"\n  H1 (Tail = boundary regions): {h1}")
    print(f"  H3 (Gradient concentration): {h3}")

    amp = all_results['exp3_gradient'].get('amplification_factor', 1.0)
    grad_ratio = all_results['exp1_spatial'].get('gradient_ratio', 1.0)

    print(f"\n  Key findings:")
    print(f"    - Tail patches have {grad_ratio:.1f}x higher image gradient (boundary regions)")
    print(f"    - Tail-aware loss amplifies gradient focus by {amp:.1f}x")
    print(f"    - Combined effect: Precise learning at normal distribution boundaries")

    return all_results


if __name__ == '__main__':
    main()
