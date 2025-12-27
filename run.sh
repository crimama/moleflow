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

# V4.1: Complete Separation Architecture (Improved)
# ==================================================
# Changes from V4:
# - SpatialMixer frozen after Task 0
# - context_conv frozen after Task 0 (NEW in V4.1)
# - context_scale_param frozen after Task 0 (NEW in V4.1)
# All shared parameters are now frozen → true complete separation

CUDA_VISIBLE_DEVICES=0 python run_moleflow.py --run_diagnostics \
      --use_whitening_adapter \
      --use_dia \
      --experiment_name Version4.1-CompleteSeparation

# # GPU 0: Original class order
# (
#   CUDA_VISIBLE_DEVICES=0 python run_moleflow.py --run_diagnostics \
#       --task_classes leather grid transistor carpet zipper hazelnut toothbrush metal_nut screw wood tile capsule pill cable bottle \
#       --use_whitening_adapter \
#       --use_dia \
#       --experiment_name Version4.1-CompleteSeparation_all_classes
# ) &

# # GPU 1: Alphabet order
# (
#   CUDA_VISIBLE_DEVICES=1 python run_moleflow.py --run_diagnostics \
#       --task_classes bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper \
#       --use_whitening_adapter \
#       --use_dia \
#       --experiment_name Version4.1-CompleteSeparation_all_classes_alphabet_order
# ) &

wait
echo "V4.1 실험 완료"

# Previous experiments (commented out)
# ====================================
# V4 experiments
# CUDA_VISIBLE_DEVICES=0 python run_moleflow.py --run_diagnostics \
#     --task_classes leather grid transistor carpet zipper hazelnut toothbrush metal_nut screw wood tile capsule pill cable bottle \
#     --use_whitening_adapter --use_dia \
#     --experiment_name Version4-CompleteSeparation_all_classes

# V3 experiments
# CUDA_VISIBLE_DEVICES=1s python run_moleflow.py --run_diagnostics --use_whitening_adapter \
#     --use_dia --use_ogp --task_classes leather bottle cable capsule carpet grid hazelnut metal_nut pill screw tile toothbrush transistor wood zipper \
#     --experiment_name Version3-WhiteningAdapter_DIA_OGP_all_classes_alphabet_order_leather_first