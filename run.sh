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
CUDA_VISIBLE_DEVICES=1 python run_moleflow.py \
    --task_classes leather grid transistor carpet zipper hazelnut toothbrush metal_nut screw wood tile capsule pill cable bottle\
    --num_epochs 40 \
    --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --img_size 224 \
    --num_coupling_layers 8 \
    --lora_rank 64 \
    --experiment_name baseline_v3_embed_fix_jacobian_fix

# v8.2: LoRA rank 128
# CUDA_VISIBLE_DEVICES=1 python run_moleflow.py \
#     --task_classes leather grid transistor carpet zipper hazelnut toothbrush metal_nut screw wood tile capsule pill cable bottle\
#     --num_epochs 40 \
#     --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
#     --img_size 224 \
#     --num_coupling_layers 8 \
#     --lora_rank 128 \
#     --experiment_name baseline_v8.2_lora_rank128_all_classes_in21k

# =============================================================================
# v7: Focal NLL Loss for Hard Sample Mining (archived)
# =============================================================================
# 어려운 샘플(낮은 probability)에 더 큰 가중치 부여
# Standard NLL: L = -log p(x)
# Focal NLL:    L = (1 - p)^γ * (-log p(x))
# gamma=0: standard, gamma>0: focus on hard samples
# python run_moleflow.py \
#     --task_classes leather grid transistor carpet zipper hazelnut toothbrush metal_nut screw wood tile capsule pill cable bottle \
#     --num_epochs 40 \
#     --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
#     --img_size 224 \
#     --num_coupling_layers 8 \
#     --lora_rank 32 \
#     --focal_gamma 2.0 \
#     --experiment_name baseline_v7_focal_loss_all_classes

# =============================================================================
# v6: Patch Self-Attention for Contextual Anomaly Detection (실패)
# =============================================================================
# Patch 간 관계를 모델링하여 contextual anomaly 탐지 향상
# 표준 NF: p(x_i) 독립적으로 모델링
# Patch Attention: p(x_i | context) - 주변 패치와의 관계 고려
# python run_moleflow.py \
#     --task_classes leather grid transistor \
#     --num_epochs 40 \
#     --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
#     --img_size 224 \
#     --num_coupling_layers 8 \
#     --lora_rank 32 \
#     --use_patch_attention \
#     --experiment_name baseline_v6.1_patch_attention

# =============================================================================
# v5: Center Loss for Discriminative Feature Learning (optional)
# =============================================================================
# Center Loss를 통해 normal feature를 더 compact하게 만들어서
# anomaly detection 성능 향상
# python run_moleflow.py \
#     --task_classes leather grid transistor \
#     --num_epochs 40 \
#     --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
#     --img_size 224 \
#     --num_coupling_layers 8 \
#     --lora_rank 32 \
#     --center_loss_weight 0.05 \
#     --experiment_name baseline_v5.1_center_loss

# =============================================================================
# Baseline v3: FiLM-style InputAdapter + Task 0 Self-Adaptation
# =============================================================================
# v3 Changes:
# - InputAdapter: Instance Norm → Layer Norm (preserves spatial info)
# - InputAdapter: FiLM-style modulation (gamma * x + beta)
# - InputAdapter: residual_gate 0 → 0.5 (MLP active from start)
# - Task 0 also uses InputAdapter (self-adaptation for pixel-level performance)
# python run_moleflow.py \
#     --task_classes leather grid transistor \
#     --num_epochs 40 \
#     --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
#     --img_size 224 \
#     --num_coupling_layers 8 \
#     --lora_rank 32 \
#     --experiment_name baseline_v3_vit_base

# =============================================================================
# Baseline v4: CPCF + SC-LoRA (Experimental - needs more tuning)
# =============================================================================
# python run_moleflow.py \
#     --task_classes leather grid transistor \
#     --num_epochs 40 \
#     --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
#     --img_size 224 \
#     --num_coupling_layers 8 \
#     --lora_rank 32 \
#     --use_cpcf \
#     --use_spatial_lora \
#     --sc_lora_grid_size 4 \
#     --cpcf_context_kernel 3 \
#     --experiment_name baseline_v4

# =============================================================================
# 비교 실험들 (필요시 주석 해제)
# =============================================================================

# # LoRA 없이 (InputAdapter만)
# python run_moleflow.py \
#     --task_classes leather grid transistor \
#     --num_epochs 40 \
#     --backbone_name vit_base_patch14_dinov2.lvd142m \
#     --img_size 518 \
#     --num_coupling_layers 8 \
#     --lora_rank 32 \
#     --no_lora \
#     --experiment_name wo_lora_v2

# # Router 없이 (Oracle task_id 사용)
# python run_moleflow.py \
#     --task_classes leather grid transistor \
#     --num_epochs 40 \
#     --backbone_name vit_base_patch14_dinov2.lvd142m \
#     --img_size 518 \
#     --num_coupling_layers 8 \
#     --lora_rank 32 \
#     --no_router \
#     --experiment_name wo_router_v2

# # 더 높은 LoRA rank 테스트
# python run_moleflow.py \
#     --task_classes leather grid transistor \
#     --num_epochs 40 \
#     --backbone_name vit_base_patch14_dinov2.lvd142m \
#     --img_size 518 \
#     --num_coupling_layers 8 \
#     --lora_rank 64 \
#     --experiment_name baseline_rank64
