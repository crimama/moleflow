#!/usr/bin/env python
"""
Experiment 4: Latent Space Calibration Analysis

Hypothesis H7: Tail-Aware Loss improves latent space calibration (z Gaussianity).

Analysis:
1. Train two models: with and without tail-aware loss
2. Collect latent samples z from the flow model
3. Test Gaussianity using Shapiro-Wilk and KS tests
4. Analyze tail calibration via QQ-plot analysis
5. Compare empirical vs theoretical quantiles at extreme percentiles

Expected: If Tail-Aware Loss improves calibration:
- Higher QQ correlation
- Lower tail calibration error
- Better Gaussianity scores

Author: Claude Code
Date: 2026-01-08
"""

import os
import sys
import json
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, '/Volume/MoLeFlow')

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
from moleflow.config import AblationConfig
from moleflow.analysis.latent_analyzer import LatentSpaceAnalyzer


# Configuration
OUTPUT_DIR = '/Volume/MoLeFlow/analysis_results/latent_calibration'
DATA_PATH = '/Data/MVTecAD'
TASK_CLASSES = ['bottle', 'cable', 'capsule']  # Quick test with 3 classes
NUM_EPOCHS = 20  # Reduced epochs for analysis
SEED = 42


def create_trainer(use_tail_loss: bool, device: str = 'cuda'):
    """Create a trainer with or without tail-aware loss."""

    args = get_config(
        img_size=224,
        msk_size=256,
        data_path=DATA_PATH,
        batch_size=16,
        seed=SEED,
        lr=3e-4,
    )
    args.dataset = 'mvtec'

    # Create ablation config
    ablation_config = AblationConfig()
    ablation_config.use_lora = True
    ablation_config.use_router = True
    ablation_config.use_task_adapter = True
    ablation_config.use_pos_embedding = True
    ablation_config.use_whitening_adapter = True
    ablation_config.use_spatial_context = True
    ablation_config.use_scale_context = True
    ablation_config.use_dia = True
    ablation_config.dia_n_blocks = 2
    ablation_config.lambda_logdet = 1e-4

    # Tail-aware loss settings
    ablation_config.use_tail_aware_loss = use_tail_loss
    ablation_config.tail_weight = 0.7 if use_tail_loss else 0.0
    ablation_config.tail_top_k_ratio = 0.02 if use_tail_loss else None
    ablation_config.score_aggregation_mode = 'top_k'
    ablation_config.score_aggregation_top_k = 3

    # Backbone
    backbone_name = 'wide_resnet50_2'
    embed_dim = 768

    # Create feature extractor
    vit_extractor = create_feature_extractor(
        backbone_name=backbone_name,
        device=device,
    )

    # Positional embedding (only needs device)
    pos_embed_generator = PositionalEmbeddingGenerator(device=device)

    # Create NF model
    nf_model = MoLESpatialAwareNF(
        embed_dim=embed_dim,
        coupling_layers=6,
        lora_rank=64,
        ablation_config=ablation_config,
        device=device,
    )

    # Create trainer
    trainer = MoLEContinualTrainer(
        vit_extractor=vit_extractor,
        pos_embed_generator=pos_embed_generator,
        nf_model=nf_model,
        args=args,
        ablation_config=ablation_config,
        device=device,
    )

    return trainer, args


def train_model(trainer, args, task_classes, num_epochs, condition_name):
    """Train model on specified tasks."""
    print(f"\n{'='*70}")
    print(f"Training: {condition_name}")
    print(f"{'='*70}")

    # Create global class to idx mapping
    global_class_to_idx = {cls: i for i, cls in enumerate(task_classes)}

    for task_id, class_name in enumerate(task_classes):
        print(f"\n--- Task {task_id}: {class_name} ---")

        # Create dataset
        train_dataset = create_task_dataset(
            args,
            task_classes=[class_name],
            global_class_to_idx=global_class_to_idx,
            train=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # Train
        trainer.train_task(
            task_id=task_id,
            train_loader=train_loader,
            num_epochs=num_epochs,
            class_names=[class_name],
        )

    return trainer, global_class_to_idx


def run_analysis():
    """Run the complete latent calibration analysis."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_seeds(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "="*70)
    print("Experiment 4: Latent Space Calibration Analysis")
    print("="*70)
    print(f"Hypothesis H7: Tail-Aware Loss improves latent space calibration")
    print(f"Tasks: {TASK_CLASSES}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Device: {device}")
    print("="*70)

    results = {}

    # Create global class to idx mapping
    global_class_to_idx = {cls: i for i, cls in enumerate(TASK_CLASSES)}

    # Condition 1: With Tail-Aware Loss
    print("\n" + "="*70)
    print("Condition 1: WITH Tail-Aware Loss")
    print("="*70)

    trainer_tail, args = create_trainer(use_tail_loss=True, device=device)
    trainer_tail, _ = train_model(trainer_tail, args, TASK_CLASSES, NUM_EPOCHS, "With Tail Loss")

    # Analyze latent space
    analyzer_tail = LatentSpaceAnalyzer(trainer_tail, device=device)

    # Collect and analyze for final task
    task_id = len(TASK_CLASSES) - 1
    test_dataset = create_task_dataset(
        args,
        task_classes=[TASK_CLASSES[task_id]],
        global_class_to_idx=global_class_to_idx,
        train=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    tail_output_dir = os.path.join(OUTPUT_DIR, 'with_tail_loss')
    results['with_tail'] = analyzer_tail.run_full_analysis(
        test_loader, task_id, tail_output_dir, num_batches=30
    )

    # Condition 2: Without Tail-Aware Loss
    print("\n" + "="*70)
    print("Condition 2: WITHOUT Tail-Aware Loss")
    print("="*70)

    trainer_notail, args = create_trainer(use_tail_loss=False, device=device)
    trainer_notail, _ = train_model(trainer_notail, args, TASK_CLASSES, NUM_EPOCHS, "Without Tail Loss")

    # Analyze latent space
    analyzer_notail = LatentSpaceAnalyzer(trainer_notail, device=device)

    notail_output_dir = os.path.join(OUTPUT_DIR, 'without_tail_loss')
    results['without_tail'] = analyzer_notail.run_full_analysis(
        test_loader, task_id, notail_output_dir, num_batches=30
    )

    # Comparison Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    comparison = {
        'with_tail': {
            'qq_correlation': results['with_tail']['tail_calibration']['qq_correlation'],
            'mean_tail_error': results['with_tail']['tail_calibration']['mean_tail_error'],
            'mean_extreme_error': results['with_tail']['tail_calibration']['mean_extreme_error'],
            'shapiro_pass_rate': results['with_tail']['gaussianity']['shapiro_pass_rate'],
            'ks_pass_rate': results['with_tail']['gaussianity']['ks_pass_rate'],
            'mean_skewness': results['with_tail']['gaussianity']['mean_skewness'],
            'mean_kurtosis': results['with_tail']['gaussianity']['mean_kurtosis'],
        },
        'without_tail': {
            'qq_correlation': results['without_tail']['tail_calibration']['qq_correlation'],
            'mean_tail_error': results['without_tail']['tail_calibration']['mean_tail_error'],
            'mean_extreme_error': results['without_tail']['tail_calibration']['mean_extreme_error'],
            'shapiro_pass_rate': results['without_tail']['gaussianity']['shapiro_pass_rate'],
            'ks_pass_rate': results['without_tail']['gaussianity']['ks_pass_rate'],
            'mean_skewness': results['without_tail']['gaussianity']['mean_skewness'],
            'mean_kurtosis': results['without_tail']['gaussianity']['mean_kurtosis'],
        }
    }

    # Print comparison table
    print("\n--- Tail Calibration Metrics ---")
    print(f"{'Metric':<25} {'With Tail':<15} {'Without Tail':<15} {'Delta':<15}")
    print("-" * 70)

    for metric in ['qq_correlation', 'mean_tail_error', 'mean_extreme_error']:
        with_val = comparison['with_tail'][metric]
        without_val = comparison['without_tail'][metric]
        delta = with_val - without_val
        sign = '+' if delta > 0 else ''
        print(f"{metric:<25} {with_val:<15.4f} {without_val:<15.4f} {sign}{delta:<14.4f}")

    print("\n--- Gaussianity Metrics ---")
    print(f"{'Metric':<25} {'With Tail':<15} {'Without Tail':<15} {'Delta':<15}")
    print("-" * 70)

    for metric in ['shapiro_pass_rate', 'ks_pass_rate', 'mean_skewness', 'mean_kurtosis']:
        with_val = comparison['with_tail'][metric]
        without_val = comparison['without_tail'][metric]
        delta = with_val - without_val
        sign = '+' if delta > 0 else ''
        print(f"{metric:<25} {with_val:<15.4f} {without_val:<15.4f} {sign}{delta:<14.4f}")

    # Interpretation
    print("\n--- Hypothesis Testing ---")
    qq_improved = comparison['with_tail']['qq_correlation'] > comparison['without_tail']['qq_correlation']
    tail_error_improved = comparison['with_tail']['mean_tail_error'] < comparison['without_tail']['mean_tail_error']
    gaussianity_improved = comparison['with_tail']['shapiro_pass_rate'] > comparison['without_tail']['shapiro_pass_rate']

    print(f"QQ Correlation improved: {'YES' if qq_improved else 'NO'}")
    print(f"Tail error reduced: {'YES' if tail_error_improved else 'NO'}")
    print(f"Gaussianity improved: {'YES' if gaussianity_improved else 'NO'}")

    evidence_count = sum([qq_improved, tail_error_improved, gaussianity_improved])
    if evidence_count >= 2:
        conclusion = "SUPPORTED: Tail-Aware Loss improves latent space calibration"
    elif evidence_count == 1:
        conclusion = "PARTIAL SUPPORT: Mixed evidence for H7"
    else:
        conclusion = "NOT SUPPORTED: Tail-Aware Loss does not improve calibration"

    print(f"\nCONCLUSION: {conclusion}")

    # Save results
    comparison['hypothesis_supported'] = evidence_count >= 2
    comparison['conclusion'] = conclusion
    comparison['timestamp'] = datetime.now().isoformat()

    with open(os.path.join(OUTPUT_DIR, 'comparison_results.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_DIR}")

    return comparison


if __name__ == '__main__':
    run_analysis()
