#!/usr/bin/env python
"""
Flow Diagnostics Runner

Analyzes the behavior of Flow's scale(s) to understand:
- Is scale(s) alive? (responds to anomalies)
- Is scale(s) dead? (no response)
- Is scale(s) noisy? (responds randomly)

Usage:
    # Run diagnostics on trained model
    python run_diagnostics.py --log_dir ./logs/your_experiment \
                              --task_classes bottle cable capsule

    # Quick diagnostics on specific classes
    python run_diagnostics.py --log_dir ./logs/your_experiment \
                              --task_classes bottle --max_samples 200
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from moleflow import (
    MoLESpatialAwareNF,
    MoLEContinualTrainer,
    ViTPatchCoreExtractor,
    PositionalEmbeddingGenerator,
    FlowDiagnostics,
    init_seeds,
    get_config,
    create_task_dataset,
)
from moleflow.config import AblationConfig


def load_config_from_log(log_dir: Path) -> dict:
    """Load config from experiment log directory."""
    import json

    config_path = log_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)

    # Try yaml
    config_path = log_dir / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)
        except ImportError:
            pass

    return {}


def main():
    parser = argparse.ArgumentParser(description='Flow Diagnostics')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Path to experiment log directory')
    parser.add_argument('--task_classes', type=str, nargs='+',
                        default=['bottle', 'cable'],
                        help='Classes to analyze')
    parser.add_argument('--data_path', type=str, default='/Data/MVTecAD',
                        help='Path to MVTec AD dataset')
    parser.add_argument('--max_samples', type=int, default=500,
                        help='Maximum samples per category')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for diagnostics (default: log_dir/diagnostics)')

    # Model parameters (will try to load from config)
    parser.add_argument('--backbone_name', type=str, default='vit_base_patch14_dinov2.lvd142m',
                        help='ViT backbone model name')
    parser.add_argument('--img_size', type=int, default=518,
                        help='Input image size')
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='Embedding dimension')
    parser.add_argument('--num_coupling_layers', type=int, default=8,
                        help='Number of coupling layers')
    parser.add_argument('--lora_rank', type=int, default=32,
                        help='LoRA rank')

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        sys.exit(1)

    # Try to load config
    config = load_config_from_log(log_dir)
    if config:
        print(f"Loaded config from {log_dir}")
        # Override args with config values
        if 'backbone_name' in config:
            args.backbone_name = config['backbone_name']
        if 'img_size' in config:
            args.img_size = config['img_size']
        if 'embed_dim' in config:
            args.embed_dim = config['embed_dim']
        if 'num_coupling_layers' in config:
            args.num_coupling_layers = config['num_coupling_layers']
        if 'lora_rank' in config:
            args.lora_rank = config['lora_rank']
        if 'data_path' in config:
            args.data_path = config['data_path']

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else log_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Flow Diagnostics")
    print("=" * 60)
    print(f"Log directory: {log_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Classes: {args.task_classes}")
    print(f"Backbone: {args.backbone_name}")
    print(f"Image size: {args.img_size}")
    print(f"Embed dim: {args.embed_dim}")
    print("=" * 60)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup data config
    data_args = get_config(
        img_size=args.img_size,
        msk_size=256,
        data_path=args.data_path,
        batch_size=args.batch_size,
        seed=0,
        lr=1e-4,
    )

    # Build ablation config from saved config
    ablation_config = AblationConfig()
    if config and 'ablation' in config:
        ablation = config['ablation']
        ablation_config.use_lora = ablation.get('use_lora', True)
        ablation_config.use_router = ablation.get('use_router', True)
        ablation_config.use_task_adapter = ablation.get('use_task_adapter', True)
        ablation_config.use_pos_embedding = ablation.get('use_pos_embedding', True)
        ablation_config.use_task_bias = ablation.get('use_task_bias', True)
        ablation_config.adapter_mode = ablation.get('adapter_mode', 'standard')
        ablation_config.use_spatial_context = ablation.get('use_spatial_context', False)
        ablation_config.spatial_context_mode = ablation.get('spatial_context_mode', 'depthwise_residual')

    # Initialize models
    print("\nInitializing models...")

    vit_extractor = ViTPatchCoreExtractor(
        backbone_name=args.backbone_name,
        blocks_to_extract=[1, 3, 5, 11],
        input_shape=(3, args.img_size, args.img_size),
        target_embed_dimension=args.embed_dim,
        device=device,
        remove_cls_token=True
    )

    pos_embed_generator = PositionalEmbeddingGenerator(device=device)

    nf_model = MoLESpatialAwareNF(
        embed_dim=args.embed_dim,
        coupling_layers=args.num_coupling_layers,
        clamp_alpha=1.9,
        lora_rank=args.lora_rank,
        lora_alpha=1.0,
        device=device,
        ablation_config=ablation_config
    )

    # Try to load model checkpoint
    checkpoint_path = log_dir / "model_checkpoint.pt"
    if not checkpoint_path.exists():
        # Try to find any checkpoint
        checkpoints = list(log_dir.glob("*.pt")) + list(log_dir.glob("*.pth"))
        if checkpoints:
            checkpoint_path = checkpoints[0]

    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                nf_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                nf_model.load_state_dict(checkpoint)
            print("Checkpoint loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            print("Running diagnostics with untrained model")
    else:
        print("Warning: No checkpoint found. Running diagnostics with untrained model")

    # Setup task mapping
    GLOBAL_CLASS_TO_IDX = {cls: i for i, cls in enumerate(args.task_classes)}

    # Add tasks to model
    for task_id, cls_name in enumerate(args.task_classes):
        nf_model.add_task(task_id)

    # Initialize diagnostics
    diagnostics = FlowDiagnostics(save_dir=str(output_dir), max_samples=args.max_samples)

    # Collect diagnostics
    print("\nCollecting diagnostics...")
    nf_model.eval()
    vit_extractor.eval()

    for task_id, cls_name in enumerate(args.task_classes):
        print(f"\nProcessing Task {task_id}: {cls_name}")
        nf_model.set_active_task(task_id)

        try:
            # Create test dataset
            data_args.class_to_idx = {cls_name: GLOBAL_CLASS_TO_IDX[cls_name]}
            data_args.n_classes = 1

            test_dataset = create_task_dataset(data_args, [cls_name], GLOBAL_CLASS_TO_IDX, train=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

            n_collected = 0
            with torch.no_grad():
                for images, labels, masks, _, _ in test_loader:
                    images = images.to(device)

                    # Extract features
                    patch_embeddings, spatial_shape = vit_extractor(images, return_spatial_shape=True)

                    if ablation_config.use_pos_embedding:
                        patch_embeddings_with_pos = pos_embed_generator(spatial_shape, patch_embeddings)
                    else:
                        B = patch_embeddings.shape[0]
                        H, W = spatial_shape
                        patch_embeddings_with_pos = patch_embeddings.reshape(B, H, W, -1)

                    # Forward through NF
                    z, logdet_patch = nf_model.forward(patch_embeddings_with_pos, reverse=False)

                    # Collect
                    is_anomaly = (labels > 0).long()
                    diagnostics.collect(
                        z=z,
                        logdet_patch=logdet_patch,
                        is_anomaly=is_anomaly,
                        images=images,
                        masks=masks,
                        class_name=cls_name
                    )
                    n_collected += images.shape[0]

            print(f"  Collected {n_collected} samples")

        except Exception as e:
            print(f"  Error processing {cls_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate plots
    print("\nGenerating diagnostic plots...")
    diagnostics.analyze_and_save()

    print("\n" + "=" * 60)
    print("Diagnostics Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    print("\nKey files to check:")
    print("  1. diagnostics_summary.txt - Summary statistics and diagnosis")
    print("  2. 1_logdet_std_comparison*.png - logdet std: normal vs anomaly")
    print("  3. 2_logdet_z_correlation*.png - ||z|| vs logdet scatter")
    print("  4. 5_spatial_heatmap_*.png - Spatial heatmaps")


if __name__ == '__main__':
    main()
