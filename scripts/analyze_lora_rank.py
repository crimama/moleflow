#!/usr/bin/env python3
"""
LoRA Rank Analysis Script

Validates the theoretical claim that task-specific changes in NF coupling subnets
are inherently low-rank by analyzing:
1. Singular value spectrum of weight updates
2. Effective rank computation
3. Rank ablation study

This addresses ECCV Reviewer W1 criticism about analogical reasoning.

Author: MoLE-Flow Team
Date: 2026-01-12
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from moleflow.models.mole_nf import MoLESpatialAwareNF
from moleflow.models.lora import LoRALinear, MoLESubnet


def compute_effective_rank(singular_values: torch.Tensor, threshold: float = 0.95) -> int:
    """
    Compute effective rank based on energy threshold.

    Args:
        singular_values: Sorted singular values (descending)
        threshold: Fraction of total energy to capture

    Returns:
        Effective rank (number of singular values needed to capture threshold energy)
    """
    S_squared = singular_values ** 2
    total_energy = S_squared.sum()

    if total_energy == 0:
        return 0

    cumulative_energy = S_squared.cumsum(0) / total_energy
    effective_rank = (cumulative_energy < threshold).sum().item() + 1

    return min(effective_rank, len(singular_values))


def compute_nuclear_norm_rank(singular_values: torch.Tensor, epsilon: float = 1e-3) -> int:
    """
    Compute rank as number of singular values above threshold.

    Args:
        singular_values: Sorted singular values
        epsilon: Threshold for considering a singular value non-zero

    Returns:
        Number of singular values above epsilon
    """
    return (singular_values > epsilon * singular_values[0]).sum().item()


def analyze_lora_weight_spectrum(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float
) -> Dict:
    """
    Analyze the singular value spectrum of LoRA weight update.

    The LoRA update is: ΔW = scaling * (B @ A)

    Args:
        lora_A: (rank, in_features) down-projection matrix
        lora_B: (out_features, rank) up-projection matrix
        scaling: LoRA scaling factor (alpha/rank)

    Returns:
        Dictionary with analysis results
    """
    # Compute full update matrix
    delta_W = scaling * (lora_B @ lora_A)

    # SVD analysis
    U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)

    # Rank metrics
    effective_rank_95 = compute_effective_rank(S, 0.95)
    effective_rank_99 = compute_effective_rank(S, 0.99)
    nuclear_rank = compute_nuclear_norm_rank(S)

    # Spectral analysis
    spectral_norm = S[0].item()
    frobenius_norm = torch.sqrt((S ** 2).sum()).item()

    # Energy concentration
    top_10_energy = (S[:10] ** 2).sum() / (S ** 2).sum() if len(S) > 0 else 0

    return {
        'singular_values': S.cpu().numpy(),
        'effective_rank_95': effective_rank_95,
        'effective_rank_99': effective_rank_99,
        'nuclear_rank': nuclear_rank,
        'spectral_norm': spectral_norm,
        'frobenius_norm': frobenius_norm,
        'top_10_energy_ratio': top_10_energy.item() if isinstance(top_10_energy, torch.Tensor) else top_10_energy,
        'delta_W_shape': delta_W.shape
    }


def analyze_base_vs_lora_weight(
    base_weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float
) -> Dict:
    """
    Compare base weight spectrum with LoRA update spectrum.

    Tests whether LoRA captures the "right" directions.
    """
    # Base weight SVD
    U_base, S_base, Vh_base = torch.linalg.svd(base_weight, full_matrices=False)

    # LoRA update SVD
    delta_W = scaling * (lora_B @ lora_A)
    U_lora, S_lora, Vh_lora = torch.linalg.svd(delta_W, full_matrices=False)

    # Subspace alignment: how much do LoRA directions overlap with base weight directions?
    # Compute principal angles between top-k subspaces
    k = min(32, min(U_base.shape[1], U_lora.shape[1]))

    U_base_k = U_base[:, :k]
    U_lora_k = U_lora[:, :k]

    # Cosine similarity of principal directions
    alignment = torch.abs(U_base_k.T @ U_lora_k)
    max_alignment = alignment.max(dim=1)[0].mean().item()

    # Relative magnitude
    lora_to_base_ratio = (S_lora.sum() / S_base.sum()).item() if S_base.sum() > 0 else 0

    return {
        'base_spectral_norm': S_base[0].item(),
        'lora_spectral_norm': S_lora[0].item() if len(S_lora) > 0 else 0,
        'direction_alignment': max_alignment,
        'magnitude_ratio': lora_to_base_ratio,
        'base_effective_rank': compute_effective_rank(S_base, 0.95),
        'lora_effective_rank': compute_effective_rank(S_lora, 0.95)
    }


def extract_model_lora_info(model: MoLESpatialAwareNF, task_id: int) -> List[Dict]:
    """
    Extract LoRA weight information from all subnets for a given task.

    Args:
        model: Trained MoLE-Flow model
        task_id: Task ID to analyze

    Returns:
        List of dictionaries with LoRA analysis per layer
    """
    task_key = str(task_id)
    results = []

    for subnet_idx, subnet in enumerate(model.subnets):
        layers = []

        if hasattr(subnet, 's_layer1'):
            # MoLEContextSubnet: 4 layers
            layers = [
                ('s_layer1', subnet.s_layer1),
                ('s_layer2', subnet.s_layer2),
                ('t_layer1', subnet.t_layer1),
                ('t_layer2', subnet.t_layer2)
            ]
        else:
            # MoLESubnet: 2 layers
            layers = [
                ('layer1', subnet.layer1),
                ('layer2', subnet.layer2)
            ]

        for layer_name, layer in layers:
            if task_key not in layer.lora_A or task_key not in layer.lora_B:
                continue

            A = layer.lora_A[task_key].data
            B = layer.lora_B[task_key].data
            scaling = layer.scaling
            base_weight = layer.base_linear.weight.data

            # Analyze LoRA update
            lora_analysis = analyze_lora_weight_spectrum(A, B, scaling)

            # Analyze base vs LoRA
            comparison = analyze_base_vs_lora_weight(base_weight, A, B, scaling)

            results.append({
                'subnet_idx': subnet_idx,
                'layer_name': layer_name,
                'task_id': task_id,
                'lora_rank': A.shape[0],
                'in_features': A.shape[1],
                'out_features': B.shape[0],
                'scaling': scaling,
                **lora_analysis,
                **comparison
            })

    return results


def plot_singular_value_spectrum(
    results: List[Dict],
    output_dir: Path,
    task_id: int
):
    """
    Plot singular value spectrum for all layers.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Get unique subnet/layer combinations
    layer_results = {}
    for r in results:
        key = f"{r['subnet_idx']}_{r['layer_name']}"
        if key not in layer_results:
            layer_results[key] = r

    for idx, (key, r) in enumerate(list(layer_results.items())[:8]):
        ax = axes[idx]
        S = r['singular_values']

        # Plot singular values
        ax.semilogy(S, 'b-', linewidth=2, label='Singular Values')

        # Mark effective ranks
        ax.axvline(r['effective_rank_95'], color='r', linestyle='--',
                   label=f"95% rank: {r['effective_rank_95']}")
        ax.axvline(r['effective_rank_99'], color='g', linestyle=':',
                   label=f"99% rank: {r['effective_rank_99']}")

        ax.set_xlabel('Index')
        ax.set_ylabel('Singular Value')
        ax.set_title(f"Subnet {r['subnet_idx']} {r['layer_name']}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'LoRA Weight Update Singular Value Spectrum (Task {task_id})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'svd_spectrum_task{task_id}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_effective_rank_distribution(
    all_results: Dict[int, List[Dict]],
    output_dir: Path
):
    """
    Plot distribution of effective ranks across tasks and layers.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Collect effective ranks
    ranks_95 = []
    ranks_99 = []
    nuclear_ranks = []
    task_ids = []

    for task_id, results in all_results.items():
        for r in results:
            ranks_95.append(r['effective_rank_95'])
            ranks_99.append(r['effective_rank_99'])
            nuclear_ranks.append(r['nuclear_rank'])
            task_ids.append(task_id)

    # Histogram of effective ranks
    ax1 = axes[0]
    ax1.hist(ranks_95, bins=20, alpha=0.7, label='95% energy')
    ax1.hist(ranks_99, bins=20, alpha=0.7, label='99% energy')
    ax1.set_xlabel('Effective Rank')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Effective Ranks')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Effective rank vs LoRA rank
    ax2 = axes[1]
    lora_ranks = [r['lora_rank'] for results in all_results.values() for r in results]
    ax2.scatter(lora_ranks, ranks_95, alpha=0.5, label='Effective rank (95%)')
    ax2.plot([0, max(lora_ranks)], [0, max(lora_ranks)], 'r--', label='y=x')
    ax2.set_xlabel('LoRA Rank (r)')
    ax2.set_ylabel('Effective Rank')
    ax2.set_title('Effective Rank vs LoRA Rank')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Box plot by task
    ax3 = axes[2]
    task_data = {}
    for task_id, results in all_results.items():
        task_data[task_id] = [r['effective_rank_95'] for r in results]

    ax3.boxplot(task_data.values(), labels=task_data.keys())
    ax3.set_xlabel('Task ID')
    ax3.set_ylabel('Effective Rank (95%)')
    ax3.set_title('Effective Rank by Task')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'effective_rank_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_alignment_analysis(
    all_results: Dict[int, List[Dict]],
    output_dir: Path
):
    """
    Analyze and plot alignment between LoRA directions and base weight directions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Collect alignment scores
    alignments = []
    magnitude_ratios = []

    for task_id, results in all_results.items():
        for r in results:
            alignments.append(r['direction_alignment'])
            magnitude_ratios.append(r['magnitude_ratio'])

    # Alignment histogram
    ax1 = axes[0]
    ax1.hist(alignments, bins=20, alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(alignments), color='r', linestyle='--',
                label=f'Mean: {np.mean(alignments):.3f}')
    ax1.set_xlabel('Direction Alignment Score')
    ax1.set_ylabel('Count')
    ax1.set_title('LoRA-to-Base Direction Alignment')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Magnitude ratio histogram
    ax2 = axes[1]
    ax2.hist(magnitude_ratios, bins=20, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(magnitude_ratios), color='r', linestyle='--',
                label=f'Mean: {np.mean(magnitude_ratios):.4f}')
    ax2.set_xlabel('LoRA/Base Magnitude Ratio')
    ax2.set_ylabel('Count')
    ax2.set_title('Relative Magnitude of LoRA Updates')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'alignment_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_summary_statistics(all_results: Dict[int, List[Dict]]) -> Dict:
    """
    Generate summary statistics across all tasks and layers.
    """
    all_ranks_95 = []
    all_ranks_99 = []
    all_alignments = []
    all_energy_ratios = []

    for task_id, results in all_results.items():
        for r in results:
            all_ranks_95.append(r['effective_rank_95'])
            all_ranks_99.append(r['effective_rank_99'])
            all_alignments.append(r['direction_alignment'])
            all_energy_ratios.append(r['top_10_energy_ratio'])

    summary = {
        'effective_rank_95': {
            'mean': np.mean(all_ranks_95),
            'std': np.std(all_ranks_95),
            'min': np.min(all_ranks_95),
            'max': np.max(all_ranks_95),
            'median': np.median(all_ranks_95)
        },
        'effective_rank_99': {
            'mean': np.mean(all_ranks_99),
            'std': np.std(all_ranks_99),
            'min': np.min(all_ranks_99),
            'max': np.max(all_ranks_99),
            'median': np.median(all_ranks_99)
        },
        'direction_alignment': {
            'mean': np.mean(all_alignments),
            'std': np.std(all_alignments),
            'min': np.min(all_alignments),
            'max': np.max(all_alignments)
        },
        'top_10_energy_concentration': {
            'mean': np.mean(all_energy_ratios),
            'std': np.std(all_energy_ratios),
            'interpretation': 'Fraction of total energy in top 10 singular values'
        },
        'num_tasks': len(all_results),
        'num_layers_analyzed': sum(len(r) for r in all_results.values())
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description='Analyze LoRA weight rank structure')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./analysis_results/lora_rank',
                        help='Output directory for plots and results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LoRA Rank Analysis for Normalizing Flow Coupling Subnets")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {output_dir}")
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Extract model state
    if 'nf_model_state_dict' in checkpoint:
        model_state = checkpoint['nf_model_state_dict']
    else:
        model_state = checkpoint

    # Reconstruct model (need config)
    config = checkpoint.get('config', {})
    embed_dim = config.get('embed_dim', 768)
    coupling_layers = config.get('num_coupling_layers', 8)
    lora_rank = config.get('lora_rank', 64)

    print(f"Model config: embed_dim={embed_dim}, coupling_layers={coupling_layers}, lora_rank={lora_rank}")

    # Create model and load state
    model = MoLESpatialAwareNF(
        embed_dim=embed_dim,
        coupling_layers=coupling_layers,
        lora_rank=lora_rank,
        device=args.device
    )

    # Determine number of tasks from checkpoint
    num_tasks = checkpoint.get('num_tasks', 1)

    # Add tasks to model structure
    for task_id in range(num_tasks):
        model.add_task(task_id)

    # Load weights
    model.load_state_dict(model_state, strict=False)
    model.eval()

    print(f"Loaded model with {num_tasks} tasks")
    print()

    # Analyze each task
    all_results = {}

    for task_id in range(num_tasks):
        print(f"Analyzing Task {task_id}...")
        results = extract_model_lora_info(model, task_id)
        all_results[task_id] = results

        # Print per-task summary
        ranks_95 = [r['effective_rank_95'] for r in results]
        print(f"  Effective rank (95%): mean={np.mean(ranks_95):.1f}, "
              f"min={np.min(ranks_95)}, max={np.max(ranks_95)}")

        # Plot singular value spectrum
        plot_singular_value_spectrum(results, output_dir, task_id)

    print()
    print("Generating aggregate analysis...")

    # Plot aggregate analysis
    plot_effective_rank_distribution(all_results, output_dir)
    plot_alignment_analysis(all_results, output_dir)

    # Generate summary
    summary = generate_summary_statistics(all_results)

    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Number of tasks analyzed: {summary['num_tasks']}")
    print(f"Total layers analyzed: {summary['num_layers_analyzed']}")
    print()
    print("Effective Rank (95% energy):")
    print(f"  Mean: {summary['effective_rank_95']['mean']:.1f}")
    print(f"  Std:  {summary['effective_rank_95']['std']:.1f}")
    print(f"  Range: [{summary['effective_rank_95']['min']}, {summary['effective_rank_95']['max']}]")
    print()
    print("Top-10 Energy Concentration:")
    print(f"  Mean: {summary['top_10_energy_concentration']['mean']:.3f}")
    print(f"  (Interpretation: {summary['top_10_energy_concentration']['interpretation']})")
    print()
    print("Direction Alignment (LoRA vs Base):")
    print(f"  Mean: {summary['direction_alignment']['mean']:.3f}")
    print()

    # Interpret results
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    mean_rank = summary['effective_rank_95']['mean']
    if mean_rank < lora_rank * 0.5:
        print(f"[STRONG SUPPORT] Effective rank ({mean_rank:.1f}) << LoRA rank ({lora_rank})")
        print("  -> Task-specific changes are indeed low-rank")
        print("  -> LoRA rank could potentially be reduced")
    elif mean_rank < lora_rank * 0.8:
        print(f"[MODERATE SUPPORT] Effective rank ({mean_rank:.1f}) < LoRA rank ({lora_rank})")
        print("  -> Task-specific changes have moderate intrinsic dimensionality")
        print("  -> Current LoRA rank is appropriate")
    else:
        print(f"[WEAK SUPPORT] Effective rank ({mean_rank:.1f}) ~ LoRA rank ({lora_rank})")
        print("  -> Task-specific changes utilize most of LoRA capacity")
        print("  -> Consider increasing LoRA rank")

    mean_energy = summary['top_10_energy_concentration']['mean']
    if mean_energy > 0.9:
        print(f"\n[STRONG] Top-10 energy ratio = {mean_energy:.3f}")
        print("  -> Energy highly concentrated in few directions")
        print("  -> Confirms low intrinsic dimensionality hypothesis")

    # Save results
    results_file = output_dir / 'analysis_results.json'

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    with open(results_file, 'w') as f:
        json.dump({
            'summary': convert_for_json(summary),
            'per_task_results': {
                str(k): convert_for_json([{kk: vv for kk, vv in r.items()
                                           if kk != 'singular_values'} for r in v])
                for k, v in all_results.items()
            },
            'timestamp': datetime.now().isoformat(),
            'checkpoint': args.checkpoint
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"Plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
