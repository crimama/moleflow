#!/usr/bin/env python3
"""
Run Tail-Aware Loss Mechanistic Analysis.

This script performs comprehensive analysis to understand WHY
Tail-Aware Loss improves anomaly detection performance.

Usage:
    python scripts/run_tail_analysis.py --checkpoint_path <path> --class_name <class>
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader
from moleflow.data.datasets import MVTEC
from moleflow.analysis import TailAwareAnalyzer, LatentSpaceAnalyzer, ScoreDistributionAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description='Tail-Aware Loss Mechanism Analysis')

    # Required arguments
    parser.add_argument('--data_path', type=str, default='/Data/MVTecAD',
                        help='Path to MVTec AD dataset')
    parser.add_argument('--class_name', type=str, default='leather',
                        help='Class to analyze')

    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                        help='Directory to save analysis results')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for analysis')
    parser.add_argument('--num_batches', type=int, default=30,
                        help='Number of batches to analyze')
    parser.add_argument('--tail_ratio', type=float, default=0.02,
                        help='Tail ratio (default: 0.02 = 2%)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    # Analysis selection
    parser.add_argument('--run_spatial', action='store_true', default=True,
                        help='Run spatial distribution analysis')
    parser.add_argument('--run_train_test', action='store_true', default=True,
                        help='Run train-test relationship analysis')
    parser.add_argument('--run_latent', action='store_true', default=True,
                        help='Run latent space analysis')
    parser.add_argument('--run_score', action='store_true', default=True,
                        help='Run score distribution analysis')
    parser.add_argument('--run_all', action='store_true', default=False,
                        help='Run all analyses')

    return parser.parse_args()


def create_mock_trainer(args):
    """
    Create a mock trainer for analysis when no checkpoint is provided.

    This trains a simple model for demonstration purposes.
    """
    print("\n" + "=" * 70)
    print("Creating trainer for analysis...")
    print("=" * 70)

    from moleflow.extractors.vit_extractor import ResNetExtractor
    from moleflow.models.mole_nf import MoLESpatialAwareNF
    from moleflow.models.pos_embedding import SinusoidalPositionalEmbedding
    from moleflow.trainer.continual_trainer import MoLEContinualTrainer
    from moleflow.config.ablation import AblationConfig

    # Create config
    config = AblationConfig(
        use_lora=True,
        use_router=True,
        use_pos_embedding=True,
        use_whitening_adapter=True,
        use_tail_aware_loss=True,
        tail_weight=1.0,
        tail_top_k_ratio=args.tail_ratio,
        score_aggregation_mode='top_k',
        score_aggregation_top_k=3,
    )

    # Create components
    extractor = ResNetExtractor(
        backbone_name='wide_resnet50_2',
        img_size=args.img_size,
        device=args.device
    )

    feature_dim = extractor.out_channels
    H = W = args.img_size // 16  # Feature map size

    pos_embed = SinusoidalPositionalEmbedding(
        feature_dim=feature_dim,
        max_h=H,
        max_w=W
    ).to(args.device)

    nf_model = MoLESpatialAwareNF(
        feature_dim=feature_dim,
        num_coupling_layers=6,
        lora_rank=64,
        use_lora=True,
        use_dia=True,
        dia_n_blocks=2,
    ).to(args.device)

    trainer = MoLEContinualTrainer(
        vit_extractor=extractor,
        pos_embed_generator=pos_embed,
        nf_model=nf_model,
        device=args.device,
        ablation_config=config,
    )

    return trainer


def main():
    args = parse_args()

    # Create output directory
    output_dir = os.path.join(args.output_dir, args.class_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("Tail-Aware Loss Mechanistic Analysis")
    print("=" * 70)
    print(f"  Class: {args.class_name}")
    print(f"  Data path: {args.data_path}")
    print(f"  Output: {output_dir}")
    print(f"  Tail ratio: {args.tail_ratio}")

    # Load or create trainer
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"\nLoading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        trainer = create_mock_trainer(args)
        trainer.nf_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("\nNo checkpoint provided - creating untrained model for analysis demo")
        print("Note: Results will be less meaningful without a trained model")
        trainer = create_mock_trainer(args)

        # Quick training for demo
        print("\nRunning quick training for demonstration...")
        train_dataset = MVTEC(
            args.data_path,
            class_name=args.class_name,
            train=True,
            img_size=args.img_size,
            crp_size=args.img_size,
            msk_size=args.img_size
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True
        )

        # Train for a few epochs
        trainer.train_task(
            task_id=0,
            task_classes=[args.class_name],
            train_loader=train_loader,
            num_epochs=5,  # Quick training
            lr=3e-4
        )

    # Create data loaders
    print("\nCreating data loaders...")
    train_dataset = MVTEC(
        args.data_path,
        class_name=args.class_name,
        train=True,
        img_size=args.img_size,
        crp_size=args.img_size,
        msk_size=args.img_size
    )
    test_dataset = MVTEC(
        args.data_path,
        class_name=args.class_name,
        train=False,
        img_size=args.img_size,
        crp_size=args.img_size,
        msk_size=args.img_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Run analyses
    results = {}

    if args.run_all or args.run_spatial or args.run_train_test:
        # Create main analyzer
        analyzer = TailAwareAnalyzer(
            trainer=trainer,
            train_loader=train_loader,
            test_loader=test_loader,
            tail_top_k_ratio=args.tail_ratio,
            device=args.device
        )

        if args.run_all:
            results = analyzer.run_all_analyses(
                output_dir=output_dir,
                num_batches=args.num_batches
            )
        else:
            if args.run_spatial:
                results['spatial'] = analyzer.analyze_tail_spatial_distribution(
                    num_batches=args.num_batches,
                    save_path=os.path.join(output_dir, 'spatial_distribution.png')
                )

            if args.run_train_test:
                results['train_test'] = analyzer.analyze_train_test_relationship(
                    num_train_batches=args.num_batches,
                    save_path=os.path.join(output_dir, 'train_test_relationship.png')
                )

    if args.run_latent:
        print("\n" + "=" * 70)
        print("Latent Space Analysis")
        print("=" * 70)

        latent_analyzer = LatentSpaceAnalyzer(
            trainer=trainer,
            device=args.device
        )

        results['latent'] = latent_analyzer.run_full_analysis(
            data_loader=train_loader,
            task_id=0,
            output_dir=output_dir,
            num_batches=args.num_batches
        )

    if args.run_score:
        print("\n" + "=" * 70)
        print("Score Distribution Analysis")
        print("=" * 70)

        score_analyzer = ScoreDistributionAnalyzer(
            trainer=trainer,
            device=args.device
        )

        results['score'] = score_analyzer.analyze_full(
            test_loader=test_loader,
            task_id=0,
            save_path=os.path.join(output_dir, 'score_distribution.png')
        )

    # Save summary
    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)
    print(f"  Results saved to: {output_dir}")

    # Save results as JSON
    import json

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    import numpy as np
    results_serializable = convert_to_serializable(results)

    with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"  Saved JSON results to: {os.path.join(output_dir, 'analysis_results.json')}")


if __name__ == '__main__':
    main()
