#!/usr/bin/env python3
"""
SVD Analysis of Trained LoRA Weights

Analyzes the effective rank and singular value spectrum of learned LoRA weights
to validate the "low-rank adaptation sufficiency" hypothesis.

Author: MoLE-Flow Team
Date: 2026-01-16
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
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from moleflow.models.mole_nf import MoLESpatialAwareNF
from moleflow.models.lora import LoRALinear
from moleflow.extractors.cnn_extractor import CNNPatchCoreExtractor
from moleflow.extractors.vit_extractor import ViTPatchCoreExtractor
from moleflow.models.position_embedding import PositionalEmbeddingGenerator
from moleflow.config.ablation import AblationConfig


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


def analyze_lora_layer(lora_A: torch.Tensor, lora_B: torch.Tensor,
                       layer_name: str, scaling: float = 1.0) -> Dict:
    """
    Analyze a single LoRA layer's weight update.

    LoRA update: Delta W = scaling * (B @ A)
    """
    # Compute the effective weight update
    delta_W = scaling * (lora_B @ lora_A)

    # SVD analysis
    U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)

    # Effective ranks at different thresholds
    eff_rank_90 = compute_effective_rank(S, 0.90)
    eff_rank_95 = compute_effective_rank(S, 0.95)
    eff_rank_99 = compute_effective_rank(S, 0.99)

    # Energy at common LoRA ranks
    energy_8 = compute_energy_at_rank(S, 8)
    energy_16 = compute_energy_at_rank(S, 16)
    energy_32 = compute_energy_at_rank(S, 32)
    energy_64 = compute_energy_at_rank(S, 64)

    # Norms
    frobenius_norm = torch.norm(delta_W, 'fro').item()
    spectral_norm = S[0].item() if len(S) > 0 else 0

    return {
        'layer_name': layer_name,
        'shape': tuple(delta_W.shape),
        'lora_rank': lora_A.shape[0],
        'max_rank': min(delta_W.shape),
        'effective_rank_90': eff_rank_90,
        'effective_rank_95': eff_rank_95,
        'effective_rank_99': eff_rank_99,
        'energy_at_8': energy_8,
        'energy_at_16': energy_16,
        'energy_at_32': energy_32,
        'energy_at_64': energy_64,
        'frobenius_norm': frobenius_norm,
        'spectral_norm': spectral_norm,
        'singular_values': S.detach().cpu().numpy()
    }


def extract_and_analyze_lora_weights(model: MoLESpatialAwareNF, task_id: int) -> List[Dict]:
    """Extract and analyze all LoRA weights for a specific task."""
    results = []

    # Convert task_id to string since LoRA dicts use string keys
    task_key = str(task_id)

    for subnet_idx, subnet in enumerate(model.subnets):
        # Check for different subnet types
        if hasattr(subnet, 's_layer1'):
            # MoLEContextSubnet
            layers = [
                ('s_layer1', subnet.s_layer1),
                ('s_layer2', subnet.s_layer2),
                ('t_layer1', subnet.t_layer1),
                ('t_layer2', subnet.t_layer2)
            ]
        elif hasattr(subnet, 'layer1'):
            # MoLESubnet
            layers = [
                ('layer1', subnet.layer1),
                ('layer2', subnet.layer2)
            ]
        else:
            continue

        for layer_name, layer in layers:
            if not isinstance(layer, LoRALinear):
                continue

            # Use string key for lookup
            if task_key not in layer.lora_A:
                continue

            lora_A = layer.lora_A[task_key]
            lora_B = layer.lora_B[task_key]
            scaling = layer.scaling

            full_name = f"subnet{subnet_idx}_{layer_name}"
            analysis = analyze_lora_layer(lora_A, lora_B, full_name, scaling)
            results.append(analysis)

    return results


def plot_results(all_results: Dict[int, List[Dict]], output_dir: Path):
    """Generate visualization plots."""

    # Plot 1: Singular value spectrum for each task
    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]

    for ax, (task_id, results) in zip(axes, all_results.items()):
        for r in results[:4]:  # Plot first 4 layers
            S = r['singular_values']
            ax.semilogy(S, label=f"{r['layer_name'][:15]}")

        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('Singular Value (log scale)')
        ax.set_title(f'Task {task_id} - SVD Spectrum')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'svd_spectrum_by_task.png', dpi=150)
    plt.close()

    # Plot 2: Effective rank distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Collect all effective ranks
    all_eff_ranks_95 = []
    all_eff_ranks_99 = []
    task_labels = []

    for task_id, results in all_results.items():
        for r in results:
            all_eff_ranks_95.append(r['effective_rank_95'])
            all_eff_ranks_99.append(r['effective_rank_99'])
            task_labels.append(f"T{task_id}")

    # Histogram
    ax1 = axes[0]
    ax1.hist(all_eff_ranks_95, bins=20, alpha=0.7, label='95% energy', color='blue')
    ax1.hist(all_eff_ranks_99, bins=20, alpha=0.7, label='99% energy', color='green')
    ax1.axvline(np.mean(all_eff_ranks_95), color='blue', linestyle='--',
                label=f'Mean 95%: {np.mean(all_eff_ranks_95):.1f}')
    ax1.axvline(np.mean(all_eff_ranks_99), color='green', linestyle='--',
                label=f'Mean 99%: {np.mean(all_eff_ranks_99):.1f}')
    ax1.axvline(64, color='red', linestyle=':', label='LoRA rank=64')
    ax1.set_xlabel('Effective Rank')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Effective Ranks')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Energy at different ranks
    ax2 = axes[1]
    ranks = [8, 16, 32, 64]
    avg_energies = []
    std_energies = []

    for rank in ranks:
        energies = []
        for results in all_results.values():
            for r in results:
                energies.append(r[f'energy_at_{rank}'])
        avg_energies.append(np.mean(energies))
        std_energies.append(np.std(energies))

    bars = ax2.bar(range(len(ranks)), avg_energies, yerr=std_energies,
                   capsize=5, color='steelblue')
    ax2.set_xticks(range(len(ranks)))
    ax2.set_xticklabels([f'r={r}' for r in ranks])
    ax2.set_xlabel('LoRA Rank')
    ax2.set_ylabel('Energy Captured')
    ax2.set_title('Energy Captured vs LoRA Rank')
    ax2.axhline(0.95, color='r', linestyle='--', label='95% threshold')
    ax2.axhline(0.99, color='g', linestyle=':', label='99% threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    # Add percentage labels on bars
    for bar, energy in zip(bars, avg_energies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{energy*100:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'effective_rank_analysis.png', dpi=150)
    plt.close()


def train_model_for_analysis(tasks: List[str], args) -> Tuple[MoLESpatialAwareNF, object]:
    """Train a MoLE-Flow model and return it for analysis."""
    from moleflow.trainer.continual_trainer import MoLEContinualTrainer
    from moleflow.data.datasets import create_task_dataset, TaskDataset
    from moleflow.models.position_embedding import PositionalEmbeddingGenerator
    from torch.utils.data import DataLoader
    import argparse

    device = args.device if torch.cuda.is_available() else 'cpu'

    # Initialize extractor
    if 'resnet' in args.backbone.lower() or 'wide_resnet' in args.backbone.lower():
        extractor = CNNPatchCoreExtractor(
            backbone_name=args.backbone,
            device=device,
            input_shape=(3, 224, 224)
        )
        embed_dim = extractor.target_embed_dimension
    else:
        extractor = ViTPatchCoreExtractor(
            backbone_name=args.backbone,
            device=device,
            input_shape=(3, 224, 224)
        )
        embed_dim = extractor.embed_dim

    # Position embedding
    pos_embed_gen = PositionalEmbeddingGenerator(device=device)

    # Ablation config
    ablation_config = AblationConfig()
    ablation_config.use_lora = True
    ablation_config.use_dia = True
    ablation_config.dia_n_blocks = args.dia_n_blocks

    # Initialize model
    model = MoLESpatialAwareNF(
        embed_dim=embed_dim,
        coupling_layers=args.coupling_layers,
        lora_rank=args.lora_rank,
        device=device,
        ablation_config=ablation_config
    )

    # Create args object for trainer
    trainer_args = argparse.Namespace(
        data_path=args.data_path,
        img_size=224,
        msk_size=224,
        dataset='mvtec',
        lr=3e-4,
        num_epochs=args.num_epochs,
        batch_size=16,
        # Add required loss/training params
        use_tail_aware_loss=True,
        tail_weight=0.7,
        tail_top_k=2,
        tail_top_k_ratio=0.02,
        lambda_logdet=1e-4,
        score_aggregation_mode='top_k',
        score_aggregation_top_k=3,
        scale_context_kernel=5,
        spatial_context_kernel=3,
    )

    # Initialize trainer
    trainer = MoLEContinualTrainer(
        vit_extractor=extractor,
        pos_embed_generator=pos_embed_gen,
        nf_model=model,
        args=trainer_args,
        device=device,
        ablation_config=ablation_config
    )

    # Create data args object for create_task_dataset
    data_args = argparse.Namespace(
        data_path=args.data_path,
        img_size=224,
        msk_size=224,
        dataset='mvtec'
    )

    # Global class to idx mapping
    global_class_to_idx = {cls: idx for idx, cls in enumerate(tasks)}

    # Train each task
    for task_id, task_class in enumerate(tasks):
        print(f"\n{'='*50}")
        print(f"Training Task {task_id}: {task_class}")
        print(f"{'='*50}")

        # Create dataset
        train_dataset = create_task_dataset(
            data_args,
            [task_class],
            global_class_to_idx,
            train=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=4,
            drop_last=True
        )

        trainer.train_task(
            task_id=task_id,
            task_classes=[task_class],
            train_loader=train_loader,
            num_epochs=args.num_epochs,
            lr=3e-4,
            log_interval=10,
            global_class_to_idx=global_class_to_idx
        )

    return model, extractor


def main():
    parser = argparse.ArgumentParser(description='Analyze trained LoRA weights SVD')
    parser.add_argument('--backbone', type=str, default='wide_resnet50_2')
    parser.add_argument('--coupling_layers', type=int, default=6)
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--dia_n_blocks', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--data_path', type=str, default='/Data/MVTecAD')
    parser.add_argument('--tasks', type=str, nargs='+', default=['leather', 'grid', 'transistor'])
    parser.add_argument('--output_dir', type=str, default='./analysis_results/lora_svd')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip_training', action='store_true', help='Skip training, analyze random weights')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("SVD Analysis of Trained LoRA Weights")
    print("=" * 70)

    if not args.skip_training:
        # Train model first
        print(f"\n[Phase 1] Training MoLE-Flow on {len(args.tasks)} tasks...")
        model, extractor = train_model_for_analysis(args.tasks, args)
        num_tasks = len(args.tasks)
    else:
        # Initialize extractor to get embed_dim
        print("\n[SKIP_TRAINING] Analyzing random initialized weights...")
        if 'resnet' in args.backbone.lower() or 'wide_resnet' in args.backbone.lower():
            extractor = CNNPatchCoreExtractor(
                backbone_name=args.backbone,
                device=device,
                input_shape=(3, 224, 224)
            )
            embed_dim = extractor.target_embed_dimension
        else:
            extractor = ViTPatchCoreExtractor(
                backbone_name=args.backbone,
                device=device,
                input_shape=(3, 224, 224)
            )
            embed_dim = extractor.embed_dim

        # Initialize model
        ablation_config = AblationConfig()
        ablation_config.use_lora = True
        ablation_config.use_dia = True
        ablation_config.dia_n_blocks = args.dia_n_blocks

        model = MoLESpatialAwareNF(
            embed_dim=embed_dim,
            coupling_layers=args.coupling_layers,
            lora_rank=args.lora_rank,
            device=device,
            ablation_config=ablation_config
        )

        # Add tasks to initialize LoRA weights
        num_tasks = len(args.tasks)
        for task_id in range(num_tasks):
            model.add_task(task_id)

    print(f"\n[Phase 2] Analyzing LoRA weights for {num_tasks} tasks...")

    all_results = {}
    for task_id in range(num_tasks):
        print(f"\n  Task {task_id}:")
        results = extract_and_analyze_lora_weights(model, task_id)
        all_results[task_id] = results

        if results:
            eff_ranks_95 = [r['effective_rank_95'] for r in results]
            eff_ranks_99 = [r['effective_rank_99'] for r in results]
            energies_64 = [r['energy_at_64'] for r in results]

            print(f"    Layers analyzed: {len(results)}")
            print(f"    Effective Rank (95%): {np.mean(eff_ranks_95):.1f} +/- {np.std(eff_ranks_95):.1f}")
            print(f"    Effective Rank (99%): {np.mean(eff_ranks_99):.1f} +/- {np.std(eff_ranks_99):.1f}")
            print(f"    Energy at r=64: {np.mean(energies_64)*100:.1f}% +/- {np.std(energies_64)*100:.1f}%")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_results(all_results, output_dir)

    # Generate summary report
    print("\nGenerating summary report...")

    all_eff_ranks_95 = []
    all_eff_ranks_99 = []
    all_energies = {8: [], 16: [], 32: [], 64: []}

    for results in all_results.values():
        for r in results:
            all_eff_ranks_95.append(r['effective_rank_95'])
            all_eff_ranks_99.append(r['effective_rank_99'])
            for rank in [8, 16, 32, 64]:
                all_energies[rank].append(r[f'energy_at_{rank}'])

    summary = {
        'experiment_config': {
            'backbone': args.backbone,
            'coupling_layers': args.coupling_layers,
            'lora_rank': args.lora_rank,
            'num_tasks': num_tasks
        },
        'effective_rank': {
            'mean_95': float(np.mean(all_eff_ranks_95)),
            'std_95': float(np.std(all_eff_ranks_95)),
            'mean_99': float(np.mean(all_eff_ranks_99)),
            'std_99': float(np.std(all_eff_ranks_99))
        },
        'energy_at_ranks': {
            f'rank_{rank}': {
                'mean': float(np.mean(all_energies[rank])),
                'std': float(np.std(all_energies[rank]))
            } for rank in [8, 16, 32, 64]
        },
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\nEffective Rank (95% energy):")
    print(f"  Mean: {np.mean(all_eff_ranks_95):.1f}")
    print(f"  Std:  {np.std(all_eff_ranks_95):.1f}")

    print(f"\nEffective Rank (99% energy):")
    print(f"  Mean: {np.mean(all_eff_ranks_99):.1f}")
    print(f"  Std:  {np.std(all_eff_ranks_99):.1f}")

    print(f"\nEnergy captured at different ranks:")
    for rank in [8, 16, 32, 64]:
        mean_e = np.mean(all_energies[rank])
        print(f"  r={rank:2d}: {mean_e*100:.1f}%")

    # Interpretation
    mean_rank_95 = np.mean(all_eff_ranks_95)
    mean_energy_64 = np.mean(all_energies[64])

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if mean_rank_95 < 32:
        print("[STRONG EVIDENCE] LoRA adaptation is intrinsically very low-rank.")
        print(f"  -> Effective rank ({mean_rank_95:.1f}) << configured LoRA rank (64)")
    elif mean_rank_95 < 64:
        print("[MODERATE EVIDENCE] LoRA adaptation has moderate intrinsic dimensionality.")
        print(f"  -> Effective rank ({mean_rank_95:.1f}) < configured LoRA rank (64)")
    else:
        print("[WEAK EVIDENCE] LoRA adaptation may need higher rank.")
        print(f"  -> Effective rank ({mean_rank_95:.1f}) >= configured LoRA rank (64)")

    if mean_energy_64 > 0.99:
        print(f"\n[VERY STRONG] r=64 captures {mean_energy_64*100:.1f}% of total energy.")
        print("  -> Current LoRA rank is more than sufficient!")
    elif mean_energy_64 > 0.95:
        print(f"\n[STRONG] r=64 captures {mean_energy_64*100:.1f}% of total energy.")
        print("  -> Current LoRA rank is appropriate.")
    elif mean_energy_64 > 0.90:
        print(f"\n[MODERATE] r=64 captures {mean_energy_64*100:.1f}% of total energy.")
        print("  -> Consider using a higher LoRA rank.")

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
