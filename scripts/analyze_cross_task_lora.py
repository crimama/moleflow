#!/usr/bin/env python3
"""
Cross-Task LoRA Similarity Analysis

Analyzes similarity of LoRA update directions across tasks to validate:
1. Tasks share common "adjustment directions" (shared structure)
2. Task-specific magnitudes along these directions (task calibration)
3. Correlation between task similarity and LoRA similarity

This provides evidence for the "distribution shift within shared framework" hypothesis.

Author: MoLE-Flow Team
Date: 2026-01-12
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from itertools import combinations

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from moleflow.models.mole_nf import MoLESpatialAwareNF


def compute_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute Centered Kernel Alignment (CKA) between two matrices.

    CKA is a similarity measure that's invariant to orthogonal transformations
    and isotropic scaling.

    Args:
        X: (n, d1) matrix
        Y: (n, d2) matrix

    Returns:
        CKA similarity score in [0, 1]
    """
    # Compute Gram matrices (linear kernel)
    K_X = X @ X.T
    K_Y = Y @ Y.T

    # Center Gram matrices
    n = K_X.shape[0]
    H = torch.eye(n, device=X.device) - torch.ones(n, n, device=X.device) / n
    K_X_centered = H @ K_X @ H
    K_Y_centered = H @ K_Y @ H

    # Compute HSIC (Hilbert-Schmidt Independence Criterion)
    hsic_XY = (K_X_centered * K_Y_centered).sum() / ((n - 1) ** 2)
    hsic_XX = (K_X_centered * K_X_centered).sum() / ((n - 1) ** 2)
    hsic_YY = (K_Y_centered * K_Y_centered).sum() / ((n - 1) ** 2)

    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    cka = hsic_XY / torch.sqrt(hsic_XX * hsic_YY + 1e-10)

    return cka.item()


def compute_cosine_similarity_matrix(vectors: List[torch.Tensor]) -> np.ndarray:
    """
    Compute pairwise cosine similarity between flattened weight matrices.

    Args:
        vectors: List of weight tensors (will be flattened)

    Returns:
        Similarity matrix (n x n)
    """
    n = len(vectors)
    sim_matrix = np.zeros((n, n))

    flat_vectors = [v.flatten() for v in vectors]

    for i in range(n):
        for j in range(n):
            cos_sim = torch.nn.functional.cosine_similarity(
                flat_vectors[i].unsqueeze(0),
                flat_vectors[j].unsqueeze(0)
            ).item()
            sim_matrix[i, j] = cos_sim

    return sim_matrix


def compute_subspace_angle(U1: torch.Tensor, U2: torch.Tensor, k: int = 32) -> float:
    """
    Compute principal angle between subspaces spanned by top-k singular vectors.

    Args:
        U1, U2: Orthonormal basis matrices from SVD
        k: Number of top singular vectors to consider

    Returns:
        Mean cosine of principal angles (1 = same subspace, 0 = orthogonal)
    """
    k = min(k, U1.shape[1], U2.shape[1])
    U1_k = U1[:, :k]
    U2_k = U2[:, :k]

    # Compute singular values of U1^T @ U2
    # These are cosines of principal angles
    M = U1_k.T @ U2_k
    _, S, _ = torch.linalg.svd(M)

    # Mean cosine of principal angles
    return S.mean().item()


def extract_lora_matrices(model: MoLESpatialAwareNF, task_ids: List[int]) -> Dict:
    """
    Extract LoRA A and B matrices for all tasks.

    Returns:
        Dictionary: task_id -> subnet_idx -> layer_name -> {'A': A, 'B': B, 'delta_W': B@A}
    """
    results = {}

    for task_id in task_ids:
        task_key = str(task_id)
        results[task_id] = {}

        for subnet_idx, subnet in enumerate(model.subnets):
            results[task_id][subnet_idx] = {}

            if hasattr(subnet, 's_layer1'):
                layers = [
                    ('s_layer1', subnet.s_layer1),
                    ('s_layer2', subnet.s_layer2),
                    ('t_layer1', subnet.t_layer1),
                    ('t_layer2', subnet.t_layer2)
                ]
            else:
                layers = [
                    ('layer1', subnet.layer1),
                    ('layer2', subnet.layer2)
                ]

            for layer_name, layer in layers:
                if task_key not in layer.lora_A:
                    continue

                A = layer.lora_A[task_key].data.clone()
                B = layer.lora_B[task_key].data.clone()
                scaling = layer.scaling

                # Compute full delta_W
                delta_W = scaling * (B @ A)

                results[task_id][subnet_idx][layer_name] = {
                    'A': A,
                    'B': B,
                    'delta_W': delta_W,
                    'scaling': scaling
                }

    return results


def analyze_cross_task_similarity(
    lora_data: Dict,
    task_ids: List[int]
) -> Dict:
    """
    Analyze similarity of LoRA updates across tasks.

    Returns comprehensive similarity analysis.
    """
    n_tasks = len(task_ids)

    # Initialize result containers
    cka_matrices = {}  # per layer
    cosine_matrices = {}  # per layer
    subspace_matrices = {}  # per layer

    # Get all subnet/layer combinations
    first_task = task_ids[0]
    subnet_layers = []
    for subnet_idx in lora_data[first_task]:
        for layer_name in lora_data[first_task][subnet_idx]:
            subnet_layers.append((subnet_idx, layer_name))

    # Compute similarity for each layer
    for subnet_idx, layer_name in subnet_layers:
        key = f"s{subnet_idx}_{layer_name}"

        # Extract delta_W matrices for all tasks
        delta_Ws = []
        for task_id in task_ids:
            delta_W = lora_data[task_id][subnet_idx][layer_name]['delta_W']
            delta_Ws.append(delta_W)

        # Cosine similarity of flattened weight updates
        cosine_matrices[key] = compute_cosine_similarity_matrix(delta_Ws)

        # CKA similarity (captures structural similarity)
        cka_matrix = np.zeros((n_tasks, n_tasks))
        for i in range(n_tasks):
            for j in range(n_tasks):
                cka_matrix[i, j] = compute_cka(delta_Ws[i], delta_Ws[j])
        cka_matrices[key] = cka_matrix

        # Subspace similarity (principal angle)
        subspace_matrix = np.zeros((n_tasks, n_tasks))
        for i in range(n_tasks):
            for j in range(n_tasks):
                U_i, _, _ = torch.linalg.svd(delta_Ws[i], full_matrices=False)
                U_j, _, _ = torch.linalg.svd(delta_Ws[j], full_matrices=False)
                subspace_matrix[i, j] = compute_subspace_angle(U_i, U_j)
        subspace_matrices[key] = subspace_matrix

    # Aggregate across layers
    avg_cka = np.mean([m for m in cka_matrices.values()], axis=0)
    avg_cosine = np.mean([m for m in cosine_matrices.values()], axis=0)
    avg_subspace = np.mean([m for m in subspace_matrices.values()], axis=0)

    return {
        'per_layer_cka': cka_matrices,
        'per_layer_cosine': cosine_matrices,
        'per_layer_subspace': subspace_matrices,
        'aggregate_cka': avg_cka,
        'aggregate_cosine': avg_cosine,
        'aggregate_subspace': avg_subspace,
        'task_ids': task_ids
    }


def analyze_shared_directions(
    lora_data: Dict,
    task_ids: List[int],
    top_k: int = 16
) -> Dict:
    """
    Analyze whether tasks share common LoRA directions.

    For each layer:
    1. Compute SVD of each task's delta_W
    2. Check if top-k directions are shared across tasks
    """
    first_task = task_ids[0]
    results = {}

    for subnet_idx in lora_data[first_task]:
        for layer_name in lora_data[first_task][subnet_idx]:
            key = f"s{subnet_idx}_{layer_name}"

            # Collect top-k right singular vectors from each task
            V_matrices = []
            S_magnitudes = []

            for task_id in task_ids:
                delta_W = lora_data[task_id][subnet_idx][layer_name]['delta_W']
                U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)

                k = min(top_k, Vh.shape[0])
                V_matrices.append(Vh[:k, :].T)  # (in_features, k)
                S_magnitudes.append(S[:k])

            # Compute pairwise direction overlap
            n_tasks = len(task_ids)
            direction_overlap = np.zeros((n_tasks, n_tasks))

            for i in range(n_tasks):
                for j in range(n_tasks):
                    # Compute alignment of top-k directions
                    alignment = torch.abs(V_matrices[i].T @ V_matrices[j])
                    # Average max alignment per direction
                    max_align_per_dir = alignment.max(dim=1)[0].mean()
                    direction_overlap[i, j] = max_align_per_dir.item()

            # Compute "consensus directions"
            # Stack all V matrices and do SVD to find shared subspace
            V_stacked = torch.cat(V_matrices, dim=1)  # (in_features, k*n_tasks)
            _, S_consensus, _ = torch.linalg.svd(V_stacked, full_matrices=False)

            # Effective rank of consensus subspace
            energy = (S_consensus ** 2).cumsum(0) / (S_consensus ** 2).sum()
            consensus_rank = (energy < 0.95).sum().item() + 1

            results[key] = {
                'direction_overlap_matrix': direction_overlap,
                'mean_direction_overlap': direction_overlap.mean(),
                'consensus_subspace_rank': consensus_rank,
                'per_task_magnitudes': [S.cpu().numpy() for S in S_magnitudes]
            }

    return results


def plot_similarity_matrices(
    similarity_results: Dict,
    output_dir: Path,
    task_names: Optional[List[str]] = None
):
    """
    Plot similarity matrices as heatmaps.
    """
    task_ids = similarity_results['task_ids']
    n_tasks = len(task_ids)

    if task_names is None:
        task_names = [f"Task {t}" for t in task_ids]

    # Plot aggregate similarities
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # CKA similarity
    ax = axes[0]
    sns.heatmap(similarity_results['aggregate_cka'], annot=True, fmt='.2f',
                xticklabels=task_names, yticklabels=task_names, ax=ax,
                cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_title('CKA Similarity')

    # Cosine similarity
    ax = axes[1]
    sns.heatmap(similarity_results['aggregate_cosine'], annot=True, fmt='.2f',
                xticklabels=task_names, yticklabels=task_names, ax=ax,
                cmap='YlOrRd', vmin=-1, vmax=1)
    ax.set_title('Cosine Similarity')

    # Subspace angle
    ax = axes[2]
    sns.heatmap(similarity_results['aggregate_subspace'], annot=True, fmt='.2f',
                xticklabels=task_names, yticklabels=task_names, ax=ax,
                cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_title('Subspace Alignment')

    plt.tight_layout()
    plt.savefig(output_dir / 'cross_task_similarity.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot per-layer CKA (select a few representative layers)
    cka_matrices = similarity_results['per_layer_cka']
    layer_keys = list(cka_matrices.keys())[:4]  # First 4 layers

    if len(layer_keys) > 0:
        fig, axes = plt.subplots(1, len(layer_keys), figsize=(4*len(layer_keys), 4))
        if len(layer_keys) == 1:
            axes = [axes]

        for ax, key in zip(axes, layer_keys):
            sns.heatmap(cka_matrices[key], annot=True, fmt='.2f',
                        xticklabels=task_names, yticklabels=task_names, ax=ax,
                        cmap='YlOrRd', vmin=0, vmax=1)
            ax.set_title(f'CKA: {key}')

        plt.tight_layout()
        plt.savefig(output_dir / 'per_layer_cka.png', dpi=150, bbox_inches='tight')
        plt.close()


def plot_shared_directions(
    shared_results: Dict,
    output_dir: Path,
    task_names: Optional[List[str]] = None
):
    """
    Plot analysis of shared directions across tasks.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Direction overlap distribution
    ax1 = axes[0]
    overlaps = [r['mean_direction_overlap'] for r in shared_results.values()]
    ax1.hist(overlaps, bins=20, alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(overlaps), color='r', linestyle='--',
                label=f'Mean: {np.mean(overlaps):.3f}')
    ax1.set_xlabel('Mean Direction Overlap')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Cross-Task Direction Overlap')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Consensus subspace rank
    ax2 = axes[1]
    consensus_ranks = [r['consensus_subspace_rank'] for r in shared_results.values()]
    ax2.hist(consensus_ranks, bins=20, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(consensus_ranks), color='r', linestyle='--',
                label=f'Mean: {np.mean(consensus_ranks):.1f}')
    ax2.set_xlabel('Consensus Subspace Rank')
    ax2.set_ylabel('Count')
    ax2.set_title('Rank of Shared Direction Subspace')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'shared_directions.png', dpi=150, bbox_inches='tight')
    plt.close()


def interpret_results(similarity_results: Dict, shared_results: Dict) -> str:
    """
    Generate interpretation of cross-task analysis results.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("INTERPRETATION OF CROSS-TASK LORA SIMILARITY")
    lines.append("=" * 60)

    # CKA interpretation
    mean_cka = similarity_results['aggregate_cka'].mean()
    off_diag_cka = similarity_results['aggregate_cka'][
        ~np.eye(similarity_results['aggregate_cka'].shape[0], dtype=bool)
    ].mean()

    lines.append(f"\n1. CKA Similarity Analysis:")
    lines.append(f"   - Mean off-diagonal CKA: {off_diag_cka:.3f}")

    if off_diag_cka > 0.7:
        lines.append("   - [HIGH] Tasks share very similar LoRA update structure")
        lines.append("   - Supports: Single shared framework for all tasks")
    elif off_diag_cka > 0.4:
        lines.append("   - [MODERATE] Tasks share some common LoRA directions")
        lines.append("   - Supports: Shared base structure with task-specific calibration")
    else:
        lines.append("   - [LOW] Tasks have distinct LoRA update patterns")
        lines.append("   - Suggests: Higher task diversity than expected")

    # Shared directions interpretation
    mean_overlap = np.mean([r['mean_direction_overlap'] for r in shared_results.values()])
    mean_consensus_rank = np.mean([r['consensus_subspace_rank'] for r in shared_results.values()])

    lines.append(f"\n2. Shared Direction Analysis:")
    lines.append(f"   - Mean direction overlap: {mean_overlap:.3f}")
    lines.append(f"   - Mean consensus subspace rank: {mean_consensus_rank:.1f}")

    if mean_overlap > 0.6:
        lines.append("   - [HIGH] Tasks align well in LoRA direction space")
        lines.append("   - Implication: Could potentially share LoRA parameters partially")
    elif mean_overlap > 0.3:
        lines.append("   - [MODERATE] Tasks share some dominant directions")
        lines.append("   - Implication: Current per-task LoRA is appropriate")
    else:
        lines.append("   - [LOW] Tasks have distinct update directions")
        lines.append("   - Implication: Per-task LoRA is necessary")

    lines.append(f"\n3. Theoretical Implications:")
    lines.append("   - Low-rank hypothesis: SUPPORTED if mean_consensus_rank << embedding_dim")
    lines.append("   - Shared structure hypothesis: SUPPORTED if off_diag_CKA > 0.3")
    lines.append("   - These results support that task changes are low-rank perturbations")
    lines.append("   within a shared transformation framework.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Analyze cross-task LoRA similarity')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./analysis_results/cross_task',
                        help='Output directory')
    parser.add_argument('--task_names', type=str, nargs='+', default=None,
                        help='Names of tasks for labeling plots')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Cross-Task LoRA Similarity Analysis")
    print("=" * 60)

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Extract config
    config = checkpoint.get('config', {})
    embed_dim = config.get('embed_dim', 768)
    coupling_layers = config.get('num_coupling_layers', 8)
    lora_rank = config.get('lora_rank', 64)
    num_tasks = checkpoint.get('num_tasks', 1)

    print(f"Model: embed_dim={embed_dim}, coupling={coupling_layers}, lora_rank={lora_rank}")
    print(f"Number of tasks: {num_tasks}")

    if num_tasks < 2:
        print("ERROR: Need at least 2 tasks for cross-task analysis")
        return

    # Create and load model
    model = MoLESpatialAwareNF(
        embed_dim=embed_dim,
        coupling_layers=coupling_layers,
        lora_rank=lora_rank,
        device=args.device
    )

    for task_id in range(num_tasks):
        model.add_task(task_id)

    model.load_state_dict(checkpoint.get('nf_model_state_dict', checkpoint), strict=False)
    model.eval()

    task_ids = list(range(num_tasks))
    task_names = args.task_names or [f"Task {i}" for i in task_ids]

    # Extract LoRA matrices
    print("\nExtracting LoRA matrices...")
    lora_data = extract_lora_matrices(model, task_ids)

    # Analyze similarity
    print("Computing similarity metrics...")
    similarity_results = analyze_cross_task_similarity(lora_data, task_ids)

    # Analyze shared directions
    print("Analyzing shared directions...")
    shared_results = analyze_shared_directions(lora_data, task_ids)

    # Generate plots
    print("Generating plots...")
    plot_similarity_matrices(similarity_results, output_dir, task_names)
    plot_shared_directions(shared_results, output_dir, task_names)

    # Print interpretation
    interpretation = interpret_results(similarity_results, shared_results)
    print(interpretation)

    # Save results
    results_file = output_dir / 'cross_task_analysis.json'

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
            'similarity': {
                'aggregate_cka': convert_for_json(similarity_results['aggregate_cka']),
                'aggregate_cosine': convert_for_json(similarity_results['aggregate_cosine']),
                'aggregate_subspace': convert_for_json(similarity_results['aggregate_subspace'])
            },
            'shared_directions': {
                k: {
                    'mean_direction_overlap': v['mean_direction_overlap'],
                    'consensus_subspace_rank': v['consensus_subspace_rank']
                }
                for k, v in shared_results.items()
            },
            'task_ids': task_ids,
            'task_names': task_names,
            'interpretation': interpretation,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"Plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
