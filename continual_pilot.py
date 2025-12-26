#!/usr/bin/env python
"""
MoLE-Flow: Continual Anomaly Detection (Backward Compatible Entry Point)

This file provides backward compatibility with the original monolithic implementation.
All modules are now imported from the moleflow package.

Optimizer: AdamP (Adaptive Momentum with Decoupled Weight Decay)
- Better generalization than AdamW through projection-based gradient correction
- Automatically falls back to AdamW if adamp is not installed

Usage:
    python continual_pilot.py --task_classes bottle cable --num_epochs 10
"""

import os
import sys

# Add paths
sys.path.insert(0, '/Volume/MoLeFlow')
sys.path.insert(0, '/Volume/NFCAD')
os.chdir('/Volume/NFCAD/')

import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from easydict import EasyDict

# Import from NFCAD (for config and dataset)
from utils.arguments import parse_args, save_config
from utils.utils import init_seeds, setting_lr_parameters
from trainer.continual_train import create_task_dataset

# =============================================================================
# Import from modular moleflow package
# =============================================================================

# Utils
from moleflow.utils.logger import TrainingLogger, setup_training_logger

# Extractors
from moleflow.extractors.hooks import (
    LastLayerToExtractReachedException,
    ForwardHook,
    NetworkFeatureAggregator
)
from moleflow.extractors.patch import PatchMaker
from moleflow.extractors.preprocessing import MeanMapper, Preprocessing
from moleflow.extractors.cnn_extractor import PatchCoreExtractor
from moleflow.extractors.vit_extractor import ViTFeatureAggregator, ViTPatchCoreExtractor

# Models
from moleflow.models.position_embedding import PositionalEmbeddingGenerator, positionalencoding2d
from moleflow.models.lora import LoRALinear, MoLESubnet
from moleflow.models.adapters import FeatureStatistics, TaskInputAdapter, SimpleTaskAdapter
from moleflow.models.routing import TaskPrototype, PrototypeRouter
from moleflow.models.mole_nf import MoLESpatialAwareNF

# Trainer (includes AdamP optimizer)
from moleflow.trainer.continual_trainer import (
    MoLEContinualTrainer,
    AdamP,
    ADAMP_CONFIG,
    create_optimizer
)

# Evaluation
from moleflow.evaluation.evaluator import (
    evaluate_class,
    evaluate_all_tasks,
    evaluate_routing_performance
)


# =============================================================================
# Main Execution (same as original)
# =============================================================================

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='MoLE-Flow: Continual Anomaly Detection')
    parser.add_argument('--task_classes', type=str, nargs='+',
                        default=['bottle', 'cable'],
                        help='Classes to learn sequentially')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs per task')
    parser.add_argument('--lora_rank', type=int, default=32,
                        help='LoRA rank for adaptation (32 recommended for cross-domain like Object vs Texture)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    # Slow-Fast configuration
    parser.add_argument('--slow_lr_ratio', type=float, default=0.2,
                        help='LR ratio for slow update (Stage 2 LR = base_lr * slow_lr_ratio, increased from 0.1)')
    parser.add_argument('--slow_blocks_k', type=int, default=2,
                        help='Number of last coupling blocks to unfreeze in Stage 2')
    parser.add_argument('--enable_slow_stage', action='store_true',
                        help='Enable Stage 2 (SLOW consolidation). Default: FAST only')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save log files')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment (default: auto-generated with timestamp)')
    parsed_args = parser.parse_args()

    # Setup base args
    args = parse_args(jupyter=True, **{'img_size': 518, 'task_classes': parsed_args.task_classes})

    init_seeds(args.seed)
    setting_lr_parameters(args)
    args = EasyDict(vars(args))

    # Setup training logger
    experiment_name = parsed_args.experiment_name
    if experiment_name is None:
        # Generate experiment name from task classes
        task_str = "_".join(parsed_args.task_classes[:3])  # First 3 classes
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"mole_flow_{task_str}_{timestamp}"

    logger = setup_training_logger(
        log_dir=parsed_args.log_dir,
        experiment_name=experiment_name
    )

    # Continual learning setup: bottle -> cable
    CONTINUAL_TASKS = [[cls] for cls in parsed_args.task_classes]
    ALL_CLASSES = parsed_args.task_classes
    GLOBAL_CLASS_TO_IDX = {cls: i for i, cls in enumerate(ALL_CLASSES)}

    print("\n" + "="*70)
    print("MoLE-Flow: Continual Anomaly Detection")
    print("="*70)
    print(f"   Tasks: {CONTINUAL_TASKS}")
    print(f"   Classes: {ALL_CLASSES}")
    print(f"   LoRA Rank: {parsed_args.lora_rank}")
    print(f"   Epochs per Task: {parsed_args.num_epochs}")
    print(f"   Slow-Fast Training: {'FAST+SLOW' if parsed_args.enable_slow_stage else 'FAST only (recommended)'}")
    if parsed_args.enable_slow_stage:
        print(f"      - Slow LR Ratio: {parsed_args.slow_lr_ratio}")
        print(f"      - Slow Blocks K: {parsed_args.slow_blocks_k}")
    print("-"*70)
    # Optimizer info
    optimizer_name = "AdamP" if AdamP is not None else "AdamW (fallback)"
    print(f"   Optimizer: {optimizer_name}")
    if AdamP is not None:
        print(f"      - betas: {ADAMP_CONFIG['betas']}")
        print(f"      - weight_decay: {ADAMP_CONFIG['weight_decay']}")
        print(f"      - delta: {ADAMP_CONFIG['delta']}")
        print(f"      - wd_ratio: {ADAMP_CONFIG['wd_ratio']}")
    print("="*70)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Initialize ViT feature extractor (Frozen)
    vit_extractor = ViTPatchCoreExtractor(
        backbone_name="vit_base_patch14_dinov2.lvd142m",
        blocks_to_extract=[1, 3, 5, 11],
        input_shape=(3, 518, 518),
        target_embed_dimension=512,
        device=device,
        remove_cls_token=True
    )
    print("\nViTPatchCoreExtractor initialized (Frozen)")
    print(f"   - Backbone: vit_base_patch14_dinov2.lvd142m")
    print(f"   - Blocks: {vit_extractor.blocks_to_extract_from}")
    print(f"   - Target dimension: {vit_extractor.target_embed_dimension}")

    # Initialize positional embedding generator
    pos_embed_generator = PositionalEmbeddingGenerator(device=device)

    # Initialize MoLE-Flow model
    nf_model = MoLESpatialAwareNF(
        embed_dim=512,
        coupling_layers=8,
        clamp_alpha=1.9,
        lora_rank=parsed_args.lora_rank,
        lora_alpha=1.0,
        device=device
    )
    print("\nMoLE-Flow NF model initialized")

    # Pass enable_slow_stage to args
    args.enable_slow_stage = parsed_args.enable_slow_stage

    # Initialize continual trainer with Slow-Fast configuration
    trainer = MoLEContinualTrainer(
        vit_extractor=vit_extractor,
        pos_embed_generator=pos_embed_generator,
        nf_model=nf_model,
        args=args,
        device=device,
        slow_lr_ratio=parsed_args.slow_lr_ratio,
        slow_blocks_k=parsed_args.slow_blocks_k
    )

    # Set logger to trainer
    trainer.set_logger(logger)

    stage_mode = "FAST+SLOW" if parsed_args.enable_slow_stage else "FAST only (recommended)"
    logger.info(f"   Slow-Fast Config: {stage_mode}")
    if parsed_args.enable_slow_stage:
        logger.info(f"      - slow_lr_ratio={parsed_args.slow_lr_ratio}, slow_blocks={parsed_args.slow_blocks_k}")
    print(f"   Slow-Fast Config: {stage_mode}")
    if parsed_args.enable_slow_stage:
        print(f"      - slow_lr_ratio={parsed_args.slow_lr_ratio}, slow_blocks={parsed_args.slow_blocks_k}")

    # ==========================================================================
    # Continual Learning Loop
    # ==========================================================================

    for task_id, task_classes in enumerate(CONTINUAL_TASKS):
        print(f"\n{'#'*70}")
        print(f"# Task {task_id}: {task_classes}")
        print(f"{'#'*70}")

        # Create task dataset
        args.class_to_idx = {cls: GLOBAL_CLASS_TO_IDX[cls] for cls in task_classes}
        args.n_classes = len(task_classes)

        train_dataset = create_task_dataset(args, task_classes, GLOBAL_CLASS_TO_IDX, train=True)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=False, drop_last=True
        )

        # Train on this task
        trainer.train_task(
            task_id=task_id,
            task_classes=task_classes,
            train_loader=train_loader,
            num_epochs=parsed_args.num_epochs,
            lr=parsed_args.lr,
            log_interval=10
        )

        # Evaluate all tasks seen so far
        logger.info(f"\nEvaluation after Task {task_id}")
        print(f"\nEvaluation after Task {task_id}")
        try:
            results = evaluate_all_tasks(
                trainer, args, use_router=True, target_size=224
            )

            # Log evaluation results
            if logger and results:
                eval_metrics = {
                    'mean_img_auc': results.get('mean_img_auc', 0.0),
                    'mean_pixel_auc': results.get('mean_pixel_auc', 0.0),
                    'mean_routing_accuracy': results.get('mean_routing_accuracy', None)
                }
                logger.log_evaluation(task_id, eval_metrics)
        except Exception as e:
            logger.warning(f"Error during evaluation: {e}")
            print(f"Error during evaluation: {e}")
            print("   Continuing with next task...")
            results = {
                'class_img_aucs': {},
                'class_pixel_aucs': {},
                'class_routing_accuracies': {},
                'task_avg_img_aucs': {},
                'mean_img_auc': 0.0,
                'mean_pixel_auc': 0.0,
            }

        # Print forgetting analysis if not first task
        if task_id > 0 and results and 'task_avg_img_aucs' in results:
            logger.info("\nForgetting Analysis:")
            print("\nForgetting Analysis:")
            for prev_task_id in range(task_id):
                if prev_task_id in results['task_avg_img_aucs']:
                    prev_classes = CONTINUAL_TASKS[prev_task_id]
                    msg = f"   Task {prev_task_id} ({prev_classes}): Image AUC={results['task_avg_img_aucs'][prev_task_id]:.4f}"
                    logger.info(msg)
                    print(msg)

    # ==========================================================================
    # Final Summary
    # ==========================================================================

    print("\n" + "="*70)
    print("Continual Learning Completed!")
    print("="*70)
    print(f"   Total Tasks: {len(CONTINUAL_TASKS)}")
    print(f"   Total Classes: {len(ALL_CLASSES)}")
    if results and 'mean_img_auc' in results and 'mean_pixel_auc' in results:
        print(f"   Final Mean Image AUC (with routing): {results['mean_img_auc']:.4f}")
        print(f"   Final Mean Pixel AUC (with routing): {results['mean_pixel_auc']:.4f}")
    else:
        print("   Final evaluation results not available")
    print("="*70)

    # ==========================================================================
    # Routing Performance Analysis - Oracle vs Routing
    # ==========================================================================

    print("\n" + "="*70)
    print("Routing Performance Analysis")
    print("="*70)
    print("Comparing Oracle (ground truth task_id) vs Router (predicted task_id)")
    print("This helps identify whether the issue is:")
    print("  - Routing problem (oracle >> router)")
    print("  - NF/LoRA adaptation problem (oracle approx router, both low)")
    print("="*70)

    # Detailed routing performance evaluation
    print("\nEvaluating Routing Accuracy...")
    routing_metrics = evaluate_routing_performance(trainer, args, target_size=224)

    # Evaluate with Oracle (use_router=False)
    print("\nEvaluating with Oracle (ground truth task_id)...")
    try:
        oracle_results = evaluate_all_tasks(
            trainer, args, use_router=False, target_size=224
        )
    except Exception as e:
        print(f"Error during oracle evaluation: {e}")
        oracle_results = {
            'class_img_aucs': {},
            'class_pixel_aucs': {},
            'class_routing_accuracies': {},
            'mean_img_auc': 0.0,
            'mean_pixel_auc': 0.0,
        }

    # Compare results
    print("\n" + "="*70)
    print("Performance Comparison: Router vs Oracle")
    print("="*70)

    if results is not None and 'class_img_aucs' in results and \
       oracle_results is not None and 'class_img_aucs' in oracle_results:

        print(f"{'Class':<15} {'Router AUC':<12} {'Oracle AUC':<12} {'AUC Gap':<10} {'Route Acc':<10} {'Diagnosis'}")
        print("-"*70)

        routing_issues = []
        nf_issues = []
        good_performance = []

        for task_id, task_classes in enumerate(CONTINUAL_TASKS):
            for class_name in task_classes:
                router_auc = results.get('class_img_aucs', {}).get(class_name, 0.0)
                oracle_auc = oracle_results.get('class_img_aucs', {}).get(class_name, 0.0)
                routing_acc = results.get('class_routing_accuracies', {}).get(class_name, None)
                gap = oracle_auc - router_auc

                routing_acc_str = f"{routing_acc:.1%}" if routing_acc is not None else "N/A"

                # Diagnosis
                if oracle_auc > 0.85:
                    if gap > 0.10:
                        diagnosis = "Routing Issue"
                        routing_issues.append((class_name, gap, routing_acc))
                    else:
                        diagnosis = "Good"
                        good_performance.append(class_name)
                else:
                    if gap > 0.10:
                        diagnosis = "Both Issues"
                        routing_issues.append((class_name, gap, routing_acc))
                        nf_issues.append((class_name, oracle_auc))
                    else:
                        diagnosis = "NF/LoRA Issue"
                        nf_issues.append((class_name, oracle_auc))

                print(f"{class_name:<15} {router_auc:<12.4f} {oracle_auc:<12.4f} "
                      f"{gap:>+9.4f} {routing_acc_str:<10} {diagnosis}")

        print("="*70)
    else:
        print("Warning: Router or Oracle results not available. Skipping comparison.")
        print("="*70)
        routing_issues = []
        nf_issues = []
        good_performance = []

    # Summary statistics
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)

    router_mean_img = results.get('mean_img_auc', 0.0) if results else 0.0
    oracle_mean_img = oracle_results.get('mean_img_auc', 0.0) if oracle_results else 0.0
    router_mean_pixel = results.get('mean_pixel_auc', 0.0) if results else 0.0
    oracle_mean_pixel = oracle_results.get('mean_pixel_auc', 0.0) if oracle_results else 0.0

    print(f"Overall Mean Image AUC:")
    print(f"  - With Router:  {router_mean_img:.4f}")
    print(f"  - With Oracle:  {oracle_mean_img:.4f}")
    print(f"  - Gap:          {oracle_mean_img - router_mean_img:+.4f}")
    print()
    print(f"Overall Mean Pixel AUC:")
    print(f"  - With Router:  {router_mean_pixel:.4f}")
    print(f"  - With Oracle:  {oracle_mean_pixel:.4f}")
    print(f"  - Gap:          {oracle_mean_pixel - router_mean_pixel:+.4f}")
    print()

    if routing_metrics and 'overall_accuracy' in routing_metrics:
        print(f"Overall Routing Accuracy: {routing_metrics['overall_accuracy']:.2%}")
        if 'task_accuracies' in routing_metrics:
            for task_id, acc in routing_metrics['task_accuracies'].items():
                if hasattr(trainer, 'task_classes') and task_id in trainer.task_classes:
                    classes = trainer.task_classes[task_id]
                    print(f"  - Task {task_id} ({', '.join(classes)}): {acc:.2%}")
    print("="*70)

    # Close logger and save final summary
    logger.close()
