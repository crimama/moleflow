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

CUDA_VISIBLE_DEVICES=0 python run_moleflow.py --run_diagnostics --use_whitening_adapter \
    --use_dia --use_ogp --task_classes leather grid transistor carpet zipper hazelnut toothbrush metal_nut screw wood tile capsule pill cable bottle \
    --experiment_name Version3-WhiteningAdapter_DIA_OGP_all_classes

# # GPU 0번에서 실행할 WhiteningAdapter 관련 실험들 (백그라운드)
# (
#   # 1. WhiteningAdapter only
#   CUDA_VISIBLE_DEVICES=0 python run_moleflow.py --run_diagnostics --use_whitening_adapter --experiment_name Version3-WhiteningAdapter_only

#   # 2. WhiteningAdapter + DIA
#   CUDA_VISIBLE_DEVICES=0 python run_moleflow.py --run_diagnostics --use_whitening_adapter --use_dia --experiment_name Version3-WhiteningAdapter_DIA

#   # 3. WhiteningAdapter + OGP
#   CUDA_VISIBLE_DEVICES=0 python run_moleflow.py --run_diagnostics --use_whitening_adapter --use_ogp --experiment_name Version3-WhiteningAdapter_OGP

#   # 4. WhiteningAdapter + DIA + OGP
#   CUDA_VISIBLE_DEVICES=0 python run_moleflow.py --run_diagnostics --use_whitening_adapter --use_dia --use_ogp --experiment_name Version3-WhiteningAdapter_DIA_OGP
# ) &

# # GPU 1번에서 실행할 MS-Context 관련 실험들 (백그라운드)
# (
#   # 1. MS-Context only
#   CUDA_VISIBLE_DEVICES=1 python run_moleflow.py --run_diagnostics --use_ms_context --experiment_name Version3-MSContext_only

#   # 2. MS-Context + DIA
#   CUDA_VISIBLE_DEVICES=1 python run_moleflow.py --run_diagnostics --use_ms_context --use_dia --experiment_name Version3-MSContext_DIA

#   # 3. MS-Context + OGP
#   CUDA_VISIBLE_DEVICES=1 python run_moleflow.py --run_diagnostics --use_ms_context --use_ogp --experiment_name Version3-MSContext_OGP

#   # 4. MS-Context + DIA + OGP
#   CUDA_VISIBLE_DEVICES=1 python run_moleflow.py --run_diagnostics --use_ms_context --use_dia --use_ogp --experiment_name Version3-MSContext_DIA_OGP
# ) &

# # 모든 백그라운드 작업이 완료될 때까지 대기
# wait

# echo "모든 실험이 완료되었습니다."