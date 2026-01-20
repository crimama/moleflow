#!/bin/bash

# =============================================================================
# MoLE-Flow: VisA Dataset Optimal Configuration Experiments
# =============================================================================
# 4 experiments optimized for VisA dataset based on MVTec AD analysis
# GPU allocation: 0, 1, 4, 5
# =============================================================================

echo "============================================================="
echo "MoLE-Flow: VisA Optimal Configuration Experiments"
echo "============================================================="
echo "Timestamp: $(date)"
echo "============================================================="

# VisA classes (12 classes total)
VISA_CLASSES="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"
DATA_PATH="/Data/VISA"

# =============================================================================
# Experiment 1: Baseline (MVTec MAIN configuration)
# GPU 0 - MVTec에서 검증된 최적 설정을 VisA에 적용
# =============================================================================
echo "[GPU 0] Starting Experiment 1: Baseline (MoLE6-DIA2)"
CUDA_VISIBLE_DEVICES=0 python run_moleflow.py \
    --dataset visa \
    --data_path $DATA_PATH \
    --task_classes $VISA_CLASSES \
    --backbone_name wide_resnet50_2 \
    --num_epochs 60 \
    --lr 3e-4 \
    --lora_rank 64 \
    --num_coupling_layers 6 \
    --dia_n_blocks 2 \
    --batch_size 16 \
    --use_whitening_adapter \
    --use_tail_aware_loss \
    --tail_weight 0.7 \
    --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 3 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5 \
    --spatial_context_kernel 3 \
    --experiment_name "VisA-Exp1-Baseline-MoLE6-DIA2" &

# =============================================================================
# Experiment 2: High Pixel AP Focus (Tail-Aware 강화 + 긴 학습)
# GPU 1 - Pixel-level localization 성능 최대화
# =============================================================================
echo "[GPU 1] Starting Experiment 2: High Pixel AP (TailW1.0-100ep)"
CUDA_VISIBLE_DEVICES=1 python run_moleflow.py \
    --dataset visa \
    --data_path $DATA_PATH \
    --task_classes $VISA_CLASSES \
    --backbone_name wide_resnet50_2 \
    --num_epochs 100 \
    --lr 3e-4 \
    --lora_rank 64 \
    --num_coupling_layers 6 \
    --dia_n_blocks 2 \
    --batch_size 16 \
    --use_whitening_adapter \
    --use_tail_aware_loss \
    --tail_weight 1.0 \
    --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 3 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5 \
    --spatial_context_kernel 3 \
    --experiment_name "VisA-Exp2-HighPixAP-TailW1.0-100ep" &

# =============================================================================
# Experiment 3: Stronger Regularization (DIA 강화)
# GPU 4 - VisA의 복잡한 배경과 텍스처에 robust한 학습
# =============================================================================
echo "[GPU 4] Starting Experiment 3: Strong Regularization (DIA4)"
CUDA_VISIBLE_DEVICES=4 python run_moleflow.py \
    --dataset visa \
    --data_path $DATA_PATH \
    --task_classes $VISA_CLASSES \
    --backbone_name wide_resnet50_2 \
    --num_epochs 80 \
    --lr 2e-4 \
    --lora_rank 32 \
    --num_coupling_layers 6 \
    --dia_n_blocks 4 \
    --batch_size 16 \
    --use_whitening_adapter \
    --use_tail_aware_loss \
    --tail_weight 0.8 \
    --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 5 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5 \
    --spatial_context_kernel 3 \
    --experiment_name "VisA-Exp3-StrongReg-DIA4-lr2e-4" &

# =============================================================================
# Experiment 4: Maximum Pixel AP (MoLE8 + DIA4 + Lambda 강화)
# GPU 5 - 최대 Pixel AP를 위한 공격적 설정
# =============================================================================
echo "[GPU 5] Starting Experiment 4: Max Pixel AP (MoLE8-DIA4)"
CUDA_VISIBLE_DEVICES=5 python run_moleflow.py \
    --dataset visa \
    --data_path $DATA_PATH \
    --task_classes $VISA_CLASSES \
    --backbone_name wide_resnet50_2 \
    --num_epochs 100 \
    --lr 2e-4 \
    --lora_rank 64 \
    --num_coupling_layers 8 \
    --dia_n_blocks 4 \
    --batch_size 16 \
    --use_whitening_adapter \
    --use_tail_aware_loss \
    --tail_weight 1.0 \
    --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 3 \
    --lambda_logdet 2e-4 \
    --scale_context_kernel 5 \
    --spatial_context_kernel 3 \
    --experiment_name "VisA-Exp4-MaxPixAP-MoLE8-DIA4" &

# =============================================================================
# Wait for all experiments to complete
# =============================================================================
echo "============================================================="
echo "All 4 experiments started on GPUs 0, 1, 4, 5"
echo "Waiting for completion..."
echo "============================================================="

wait

echo "============================================================="
echo "All experiments completed!"
echo "Timestamp: $(date)"
echo "============================================================="
