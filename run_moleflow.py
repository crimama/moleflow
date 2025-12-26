#!/usr/bin/env python
"""
MoLE-Flow: Continual Anomaly Detection Runner

This is the main entry point for running MoLE-Flow training and evaluation.
Uses the modular moleflow package (fully independent, no NFCAD dependency).

Ablation Support:
    --ablation_preset: Use predefined ablation configuration
    --no_lora: Disable LoRA adaptation
    --no_router: Disable router (use oracle task_id)
    --no_task_adapter: Disable task input adapter
    --no_pos_embedding: Disable positional embedding
    --no_task_bias: Disable task-specific bias
    --no_mahalanobis: Use Euclidean distance instead of Mahalanobis
"""

import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

# Import everything from moleflow package (no NFCAD dependency)
from moleflow import (
    MoLESpatialAwareNF,
    MoLEContinualTrainer,
    ViTPatchCoreExtractor,
    PositionalEmbeddingGenerator,
    TrainingLogger,
    setup_training_logger,
    evaluate_all_tasks,
    evaluate_routing_performance,
    # Utilities
    init_seeds,
    setting_lr_parameters,
    get_config,
    # Data
    create_task_dataset,
)
from moleflow.config import (
    AblationConfig,
    add_ablation_args,
    parse_ablation_args
)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='MoLE-Flow: Continual Anomaly Detection')
    parser.add_argument('--task_classes', type=str, nargs='+',
                        default=['bottle', 'cable'],
                        help='Classes to learn sequentially')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs per task')
    parser.add_argument('--lora_rank', type=int, default=32,
                        help='LoRA rank for adaptation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--slow_lr_ratio', type=float, default=0.2,
                        help='LR ratio for slow update')
    parser.add_argument('--slow_blocks_k', type=int, default=2,
                        help='Number of last coupling blocks to unfreeze in Stage 2')
    parser.add_argument('--enable_slow_stage', action='store_true',
                        help='Enable Stage 2 (SLOW consolidation)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save log files')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment')
    parser.add_argument('--backbone_name', type=str,
                        default='vit_base_patch14_dinov2.lvd142m',
                        help='ViT backbone model name (e.g., vit_base_patch14_dinov2.lvd142m)')
    parser.add_argument('--img_size', type=int, default=518,
                        help='Input image size (default: 518)')
    parser.add_argument('--msk_size', type=int, default=256,
                        help='Mask size for evaluation (default: 256)')
    parser.add_argument('--num_coupling_layers', type=int, default=8,
                        help='Number of coupling layers in NF model (default: 8)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--data_path', type=str, default='/Data/MVTecAD',
                        help='Path to MVTec AD dataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--embed_dim', type=int, default=None,
                        help='Embedding dimension for NF model (default: same as ViT output dim)')

    # Add ablation arguments
    add_ablation_args(parser)

    parsed_args = parser.parse_args()

    # Parse ablation configuration
    ablation_config = parse_ablation_args(parsed_args)

    # Setup configuration using moleflow's config system
    args = get_config(
        img_size=parsed_args.img_size,
        msk_size=parsed_args.msk_size,
        data_path=parsed_args.data_path,
        batch_size=parsed_args.batch_size,
        seed=parsed_args.seed,
        lr=parsed_args.lr,
    )

    # Initialize seeds
    init_seeds(args.seed)
    setting_lr_parameters(args)

    # Determine embed_dim early (before config saving)
    # Infer default embed_dim from backbone name
    def get_default_embed_dim(backbone_name: str) -> int:
        """Infer embedding dimension from backbone model name."""
        backbone_lower = backbone_name.lower()
        if 'vit_giant' in backbone_lower or 'vit_g' in backbone_lower:
            return 1536
        elif 'vit_huge' in backbone_lower or 'vit_h' in backbone_lower:
            return 1280
        elif 'vit_large' in backbone_lower or 'vit_l' in backbone_lower:
            return 1024
        elif 'vit_base' in backbone_lower or 'vit_b' in backbone_lower:
            return 768
        elif 'vit_small' in backbone_lower or 'vit_s' in backbone_lower:
            return 384
        elif 'vit_tiny' in backbone_lower or 'vit_t' in backbone_lower:
            return 192
        else:
            # Fallback: try to get from timm
            try:
                import timm
                model = timm.create_model(backbone_name, pretrained=False)
                if hasattr(model, 'embed_dim'):
                    dim = model.embed_dim
                    del model
                    return dim
            except:
                pass
            return 768  # Default fallback

    embed_dim = parsed_args.embed_dim if parsed_args.embed_dim is not None else get_default_embed_dim(parsed_args.backbone_name)

    # Setup training logger
    experiment_name = parsed_args.experiment_name
    if experiment_name is None:
        task_str = "_".join(parsed_args.task_classes[:3])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include ablation info in experiment name
        ablation_suffix = ablation_config.get_experiment_name()
        experiment_name = f"mole_flow_{task_str}_{ablation_suffix}_{timestamp}"

    logger = setup_training_logger(
        log_dir=parsed_args.log_dir,
        experiment_name=experiment_name
    )

    # Save config to log folder
    config = {
        'experiment_name': experiment_name,
        'task_classes': parsed_args.task_classes,
        'num_epochs': parsed_args.num_epochs,
        'lora_rank': parsed_args.lora_rank,
        'lr': parsed_args.lr,
        'slow_lr_ratio': parsed_args.slow_lr_ratio,
        'slow_blocks_k': parsed_args.slow_blocks_k,
        'enable_slow_stage': parsed_args.enable_slow_stage,
        'backbone_name': parsed_args.backbone_name,
        'img_size': parsed_args.img_size,
        'msk_size': parsed_args.msk_size,
        'num_coupling_layers': parsed_args.num_coupling_layers,
        'batch_size': parsed_args.batch_size,
        'data_path': parsed_args.data_path,
        'seed': parsed_args.seed,
        'embed_dim': embed_dim,
        # Ablation settings
        'ablation': {
            'use_lora': ablation_config.use_lora,
            'use_router': ablation_config.use_router,
            'use_task_adapter': ablation_config.use_task_adapter,
            'use_pos_embedding': ablation_config.use_pos_embedding,
            'use_task_bias': ablation_config.use_task_bias,
            'use_mahalanobis': ablation_config.use_mahalanobis,
        },
        'device': str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    logger.save_config(config)

    # Setup continual learning tasks
    CONTINUAL_TASKS = [[cls] for cls in parsed_args.task_classes]
    ALL_CLASSES = parsed_args.task_classes
    GLOBAL_CLASS_TO_IDX = {cls: i for i, cls in enumerate(ALL_CLASSES)}

    print("\n" + "="*70)
    print("MoLE-Flow: Continual Anomaly Detection")
    print("="*70)
    print(f"   Tasks: {CONTINUAL_TASKS}")
    print(f"   Classes: {ALL_CLASSES}")
    print(f"   Backbone: {parsed_args.backbone_name}")
    print(f"   Image Size: {parsed_args.img_size}")
    print(f"   Coupling Layers: {parsed_args.num_coupling_layers}")
    print(f"   Embedding Dim: {embed_dim}")
    print(f"   LoRA Rank: {parsed_args.lora_rank}")
    print(f"   Epochs per Task: {parsed_args.num_epochs}")
    print(f"   Data Path: {parsed_args.data_path}")
    print(f"   Log Directory: {parsed_args.log_dir}")
    print(f"   Slow-Fast Training: {'FAST+SLOW' if parsed_args.enable_slow_stage else 'FAST only'}")
    print("-"*70)
    print(f"   {ablation_config}")
    print("="*70)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Initialize ViT feature extractor
    vit_extractor = ViTPatchCoreExtractor(
        backbone_name=parsed_args.backbone_name,
        blocks_to_extract=[1, 3, 5, 11],
        input_shape=(3, parsed_args.img_size, parsed_args.img_size),
        target_embed_dimension=embed_dim,
        device=device,
        remove_cls_token=True
    )
    print(f"\nViTPatchCoreExtractor initialized with {parsed_args.backbone_name} (Frozen)")
    print(f"   Embedding Dimension: {embed_dim}")

    # Initialize positional embedding generator
    pos_embed_generator = PositionalEmbeddingGenerator(device=device)

    # Initialize MoLE-Flow model with ablation config
    nf_model = MoLESpatialAwareNF(
        embed_dim=embed_dim,
        coupling_layers=parsed_args.num_coupling_layers,
        clamp_alpha=1.9,
        lora_rank=parsed_args.lora_rank,
        lora_alpha=1.0,
        device=device,
        ablation_config=ablation_config
    )
    print("MoLE-Flow NF model initialized")

    # Pass enable_slow_stage to args
    args.enable_slow_stage = parsed_args.enable_slow_stage

    # Initialize continual trainer with ablation config
    trainer = MoLEContinualTrainer(
        vit_extractor=vit_extractor,
        pos_embed_generator=pos_embed_generator,
        nf_model=nf_model,
        args=args,
        device=device,
        slow_lr_ratio=parsed_args.slow_lr_ratio,
        slow_blocks_k=parsed_args.slow_blocks_k,
        ablation_config=ablation_config
    )
    trainer.set_logger(logger)

    # Training loop
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
        try:
            # Use router only if enabled
            use_router = ablation_config.use_router
            results = evaluate_all_tasks(trainer, args, use_router=use_router, target_size=224)

            if logger and results:
                eval_metrics = {
                    'mean_img_auc': results.get('mean_img_auc', 0.0),
                    'mean_pixel_auc': results.get('mean_pixel_auc', 0.0),
                    'mean_img_ap': results.get('mean_img_ap', 0.0),
                    'mean_pixel_ap': results.get('mean_pixel_ap', 0.0),
                    'mean_routing_accuracy': results.get('mean_routing_accuracy', None)
                }
                logger.log_evaluation(task_id, eval_metrics)

                # Save evaluation results to CSV
                evaluated_classes = results.get('classes', [])
                img_aucs = results.get('img_aucs', [])
                pixel_aucs = results.get('pixel_aucs', [])
                routing_accuracies = results.get('routing_accuracies', None)

                # Save per-task evaluation results
                logger.save_evaluation_results_csv(
                    task_id=task_id,
                    epoch=parsed_args.num_epochs - 1,
                    class_names=evaluated_classes,
                    img_aucs=img_aucs,
                    pixel_aucs=pixel_aucs,
                    routing_accuracies=routing_accuracies
                )

                # Save continual learning results (including all previous tasks)
                logger.save_continual_results_csv(
                    task_id=task_id,
                    current_classes=task_classes,
                    all_classes=evaluated_classes,
                    img_aucs=img_aucs,
                    pixel_aucs=pixel_aucs,
                    continual_tasks=CONTINUAL_TASKS,
                    routing_accuracies=routing_accuracies
                )

                # Save unified evaluation results
                logger.save_unified_evaluation_csv(
                    task_id=task_id,
                    epoch=parsed_args.num_epochs - 1,
                    all_classes=evaluated_classes,
                    img_aucs=img_aucs,
                    pixel_aucs=pixel_aucs,
                    ALL_CLASSES=ALL_CLASSES,
                    routing_accuracies=routing_accuracies
                )

                # Save evaluation metrics to history
                logger.save_evaluation_metrics(results)

        except Exception as e:
            logger.warning(f"Error during evaluation: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            results = {}

    # Final Summary
    print("\n" + "="*70)
    print("Continual Learning Completed!")
    print("="*70)
    print(f"   Total Tasks: {len(CONTINUAL_TASKS)}")
    print(f"   Total Classes: {len(ALL_CLASSES)}")
    if results and 'mean_img_auc' in results:
        print(f"   Final Mean Image AUC: {results['mean_img_auc']:.4f}")
        print(f"   Final Mean Pixel AUC: {results['mean_pixel_auc']:.4f}")
        print(f"   Final Mean Image AP: {results.get('mean_img_ap', 0.0):.4f}")
        print(f"   Final Mean Pixel AP: {results.get('mean_pixel_ap', 0.0):.4f}")
    print("="*70)

    # Routing Performance Analysis (only if router is enabled)
    routing_metrics = None
    if ablation_config.use_router:
        print("\nRouting Performance Analysis")
        routing_metrics = evaluate_routing_performance(trainer, args, target_size=224)

        # Oracle evaluation
        print("\nEvaluating with Oracle (ground truth task_id)...")
        try:
            oracle_results = evaluate_all_tasks(trainer, args, use_router=False, target_size=224)
        except Exception as e:
            print(f"Error during oracle evaluation: {e}")
            oracle_results = {}
    else:
        print("\n[Ablation] Router disabled - using oracle task_id for all evaluations")
        oracle_results = results

    # Save final summary
    if results and 'mean_img_auc' in results:
        # Save final results table (simple format)
        logger.save_final_results_table(results, CONTINUAL_TASKS)

        # Prepare additional metrics
        additional_metrics = {}
        if routing_metrics is not None:
            additional_metrics['overall_routing_accuracy'] = routing_metrics.get('overall_accuracy', -1.0)
            for task_id_key, acc in routing_metrics.get('task_accuracies', {}).items():
                additional_metrics[f'task_{task_id_key}_routing_accuracy'] = acc

        if oracle_results and 'mean_img_auc' in oracle_results:
            additional_metrics['oracle_mean_img_auc'] = oracle_results['mean_img_auc']
            additional_metrics['oracle_mean_pixel_auc'] = oracle_results['mean_pixel_auc']

        # Save final summary to CSV
        logger.save_final_summary(
            strategy_name="MoLE-Flow",
            all_classes=results.get('classes', ALL_CLASSES),
            img_aucs=results.get('img_aucs', []),
            pixel_aucs=results.get('pixel_aucs', []),
            num_tasks=len(CONTINUAL_TASKS),
            ablation_config=ablation_config,
            routing_accuracy=results.get('mean_routing_accuracy', None),
            additional_metrics=additional_metrics,
            img_aps=results.get('img_aps', []),
            pixel_aps=results.get('pixel_aps', [])
        )

    # Close logger
    logger.close()

    return trainer, results, oracle_results


if __name__ == '__main__':
    main()
