#!/bin/bash

# MoLE-Flow: Continual Anomaly Detection
# =======================================
# All settings below are now defaults - just run: python run_moleflow.py
#
# Default configuration:
# - Backbone: vit_base_patch16_224.augreg2_in21k_ft_in1k
# - Image size: 224
# - Adapter mode: soft_ln (SoftLN init_scale=0.01)
# - Spatial context: enabled (depthwise_residual, kernel=3)
# - Scale context: enabled (s-network only)

# Classes: leather grid transistor carpet zipper hazelnut toothbrush metal_nut screw wood tile capsule pill cable bottle

# Basic run with all defaults
# CUDA_VISIBLE_DEVICES=0 python run_moleflow.py \    
#     --experiment_name Version2-CouplingLayers_16

# V5: Score Aggregation Experiments
# ==================================
# Test different aggregation methods for image-level anomaly score
# Hypothesis: top-k averaging is more robust than 99th percentile
#
# Modes:
#   percentile: Use p-th percentile (current default, p=0.99)
#   top_k: Average of top K patches (e.g., K=10)
#   top_k_percent: Average of top K% patches (e.g., 5%)
#   max: Maximum patch score
#   mean: Mean of all patches

# Pilot experiments (3 classes: leather, grid, transistor)
# Run in parallel to compare aggregation methods

# V4.2: Score Aggregation with Top-K (K=3)
# ========================================
# Baseline configuration with whitening adapter and top-k aggregation

CUDA_VISIBLE_DEVICES=0 python run_moleflow.py --run_diagnostics \
      --use_whitening_adapter --use_dia \
      --score_aggregation_mode top_k \
      --score_aggregation_top_k 3 \
      --experiment_name Version4.2-ScoreAgg_topk3

# V4.4: LayerNorm Ablation (Fair Comparison)
# ============================================
# Compare WhiteningAdapter (with LN) vs WhiteningAdapterNoLN (without LN)
# Same architecture except for LayerNorm

# CUDA_VISIBLE_DEVICES=0 python run_moleflow.py --run_diagnostics \
#       --use_dia \
#       --score_aggregation_mode top_k \
#       --score_aggregation_top_k 3 \
#       --adapter_mode whitening_no_ln \
#       --experiment_name Version4.4-whitening_no_ln

# # GPU 0: Baseline (percentile 99%)
# (
#   CUDA_VISIBLE_DEVICES=0 python run_moleflow.py --run_diagnostics \
#       --task_classes leather grid transistor carpet zipper hazelnut toothbrush metal_nut screw wood tile capsule pill cable bottle \
#       --use_whitening_adapter --use_dia \
#       --score_aggregation_mode top_k \
#       --score_aggregation_top_k 3 \
#       --experiment_name Version5.1-ScoreAgg_topk3_all_classes
# ) &

# # GPU 1: Top-K averaging (K=10)
# (
#   CUDA_VISIBLE_DEVICES=1 python run_moleflow.py --run_diagnostics \
#       --task_classes bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper \
#       --use_whitening_adapter --use_dia \
#       --score_aggregation_mode top_k \
#       --score_aggregation_top_k 3 \
#       --experiment_name Version5.2-ScoreAgg_topk3_all_classes_alphabet_order
# ) &

# wait
# echo "V4.3 Pilot 완료"