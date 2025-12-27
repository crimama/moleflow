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

# V4: Complete Separation Architecture
# =====================================
# Key change: SpatialMixer is now frozen after Task 0 (automatic in mole_nf.py)
# This prevents catastrophic forgetting caused by shared parameter drift
# OGP is removed since there are no shared trainable parameters to protect

# CUDA_VISIBLE_DEVICES=0 python run_moleflow.py --run_diagnostics \
#     --task_classes leather bottle cable capsule carpet grid hazelnut metal_nut pill screw tile toothbrush transistor wood zipper \
#     --use_whitening_adapter \
#     --use_dia \
#     --experiment_name Version4-CompleteSeparation

# Previous V3 experiments (commented out)
# CUDA_VISIBLE_DEVICES=1s python run_moleflow.py --run_diagnostics --use_whitening_adapter \
#     --use_dia --use_ogp --task_classes leather bottle cable capsule carpet grid hazelnut metal_nut pill screw tile toothbrush transistor wood zipper \
#     --experiment_name Version3-WhiteningAdapter_DIA_OGP_all_classes_alphabet_order_leather_first

# GPU 0번에서 실행할 WhiteningAdapter 관련 실험들 (백그라운드)
(
  # 1. WhiteningAdapter only
  CUDA_VISIBLE_DEVICES=0 python run_moleflow.py --run_diagnostics \
      --task_classes leather grid transistor carpet zipper hazelnut toothbrush metal_nut screw wood tile capsule pill cable bottle \
      --use_whitening_adapter \
      --use_dia \
      --experiment_name Version4-CompleteSeparation_all_classes
  
) &

# GPU 1번에서 실행할 MS-Context 관련 실험들 (백그라운드)
(
  CUDA_VISIBLE_DEVICES=1 python run_moleflow.py --run_diagnostics \
      --task_classes bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper \
      --use_whitening_adapter \
      --use_dia \
      --experiment_name Version4-CompleteSeparation_all_classes_alphabet_order
) &

# 모든 백그라운드 작업이 완료될 때까지 대기
wait

echo "모든 실험이 완료되었습니다."