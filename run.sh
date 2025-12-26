#!/bin/bash

# MoLE-Flow Experiments
# =====================
# 개선된 버전:
# - LoRA scaling 2배 증가 (alpha/rank)
# - Task 0도 LoRA 사용 (모든 task 동등한 adaptation)

# Classes: leather grid transistor carpet zipper hazelnut toothbrush metal_nut screw wood tile capsule pill cable bottle

# =============================================================================
# v8: v3 기반 + LoRA Rank 증가 실험
# =============================================================================
# v3 (FiLM-style InputAdapter)가 현재 best (Image AUC: 0.9504, Pixel AUC: 0.9313)
# LoRA rank 증가로 adaptation capacity 확대하여 추가 성능 향상 기대
#
# Baseline v3 설정:
# - Backbone: vit_base_patch14_dinov2.lvd142m (DINOv2)
# - Image size: 518
# - LoRA rank: 32
# - FiLM-style InputAdapter (LayerNorm, gate=0.5, larger MLP)

# v8.1: LoRA rank 64
CUDA_VISIBLE_DEVICES=0 python run_moleflow.py \
    --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --img_size 224 \
    --adapter_mode soft_ln \
    --soft_ln_init_scale 0.01 \
    --use_spatial_context \
    --run_diagnostic \
    --experiment_name baseline_v1.5_patch_wise_context_k3_refactoring