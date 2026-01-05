#!/bin/bash

# =============================================================================
# MoLE-Flow: VISA Optimization Experiments (img_size=224 고정)
# =============================================================================
# 목표: Image AUC >= 0.910, Pixel AP >= 0.40
# 현재 최고: Image AUC 0.9052, Pixel AP 0.2878
# =============================================================================
# GPU 0: LogDet & Scale Context 최적화 (Pixel AP 중점)
# GPU 1: Extended Training 실험 (Image AUC 중점)
# GPU 4: Architecture Scaling 실험 (용량 증가)
# GPU 5: DINOv2 & Advanced Features 실험
# =============================================================================

echo "============================================================="
echo "MoLE-Flow: VISA Optimization Experiments"
echo "============================================================="
echo "Target: Image AUC >= 0.910, Pixel AP >= 0.40"
echo "Current Best: Image AUC 0.9052, Pixel AP 0.2878"
echo "Image Size: 224 (fixed)"
echo "============================================================="
echo "Timestamp: $(date)"
echo ""

# =============================================================================
# Common Configuration
# =============================================================================

# VisA classes (12 classes)
VISA_CLASSES="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"

# Base options
BASE_OPTS="--use_whitening_adapter --use_dia --use_tail_aware_loss"
VISA_DATA="--dataset visa --data_path /Data/VISA"

# Output directory
LOG_DIR="./logs/Final"
mkdir -p $LOG_DIR

# Best current settings (from VISA-ViT-lr1e-4-DIA6-Coupling10)
BEST_VIT_BASE="--backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --num_coupling_layers 10 \
    --dia_n_blocks 6 \
    --lora_rank 64 \
    --tail_weight 0.7 \
    --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 3"

# =============================================================================
# GPU 0: LogDet & Scale Context Optimization (Pixel AP Focus)
# =============================================================================
# lambda_logdet 및 scale_context_kernel 튜닝으로 Pixel AP 향상
# =============================================================================

run_gpu0_exp1() {
    echo "[GPU 0-1] VISA-ViT-LogDet2e-4-ScaleK7"
    CUDA_VISIBLE_DEVICES=0 python run_moleflow.py \
        $BASE_OPTS $VISA_DATA \
        --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
        --num_epochs 100 \
        --lr 1e-4 \
        --lora_rank 64 \
        --num_coupling_layers 10 \
        --dia_n_blocks 6 \
        --tail_weight 0.6 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 5 \
        --lambda_logdet 2e-4 \
        --scale_context_kernel 7 \
        --task_classes $VISA_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name VISA-ViT-100ep-LogDet2e-4-ScaleK7
}

run_gpu0_exp2() {
    echo "[GPU 0-2] VISA-ViT-LogDet3e-4-ScaleK9"
    CUDA_VISIBLE_DEVICES=0 python run_moleflow.py \
        $BASE_OPTS $VISA_DATA \
        --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
        --num_epochs 100 \
        --lr 1e-4 \
        --lora_rank 64 \
        --num_coupling_layers 10 \
        --dia_n_blocks 6 \
        --tail_weight 0.5 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 5 \
        --lambda_logdet 3e-4 \
        --scale_context_kernel 9 \
        --task_classes $VISA_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name VISA-ViT-100ep-LogDet3e-4-ScaleK9
}

run_gpu0_exp3() {
    echo "[GPU 0-3] VISA-ViT-LogDet2e-4-ScaleK9-TopK7"
    CUDA_VISIBLE_DEVICES=0 python run_moleflow.py \
        $BASE_OPTS $VISA_DATA \
        --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
        --num_epochs 100 \
        --lr 8e-5 \
        --lora_rank 64 \
        --num_coupling_layers 10 \
        --dia_n_blocks 6 \
        --tail_weight 0.55 \
        --tail_top_k_ratio 0.025 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 7 \
        --lambda_logdet 2e-4 \
        --scale_context_kernel 9 \
        --task_classes $VISA_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name VISA-ViT-100ep-LogDet2e-4-ScaleK9-TopK7
}

run_gpu0_exp4() {
    echo "[GPU 0-4] VISA-ViT-SlowStage-LogDet2e-4"
    CUDA_VISIBLE_DEVICES=0 python run_moleflow.py \
        $BASE_OPTS $VISA_DATA \
        --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
        --num_epochs 80 \
        --lr 1e-4 \
        --lora_rank 64 \
        --num_coupling_layers 10 \
        --dia_n_blocks 6 \
        --tail_weight 0.6 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 5 \
        --lambda_logdet 2e-4 \
        --scale_context_kernel 7 \
        --enable_slow_stage \
        --slow_blocks_k 4 \
        --slow_lr_ratio 0.1 \
        --task_classes $VISA_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name VISA-ViT-SlowStage-K4-LogDet2e-4
}

# =============================================================================
# GPU 1: Extended Training & Learning Rate Tuning (Image AUC Focus)
# =============================================================================
# 100+ epochs로 추가 학습, 최적 lr 탐색
# =============================================================================

run_gpu1_exp1() {
    echo "[GPU 1-1] VISA-ViT-100ep-DIA6-C10 (Extended Training)"
    CUDA_VISIBLE_DEVICES=1 python run_moleflow.py \
        $BASE_OPTS $VISA_DATA $BEST_VIT_BASE \
        --num_epochs 100 \
        --lr 1e-4 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --task_classes $VISA_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name VISA-ViT-100ep-DIA6-C10
}

run_gpu1_exp2() {
    echo "[GPU 1-2] VISA-ViT-120ep-lr8e-5 (More Epochs, Lower LR)"
    CUDA_VISIBLE_DEVICES=1 python run_moleflow.py \
        $BASE_OPTS $VISA_DATA $BEST_VIT_BASE \
        --num_epochs 120 \
        --lr 8e-5 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --task_classes $VISA_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name VISA-ViT-120ep-lr8e-5
}

run_gpu1_exp3() {
    echo "[GPU 1-3] VISA-ViT-100ep-TailW0.8-TopK4 (Stronger Tail Awareness)"
    CUDA_VISIBLE_DEVICES=1 python run_moleflow.py \
        $BASE_OPTS $VISA_DATA \
        --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
        --num_epochs 100 \
        --lr 1e-4 \
        --lora_rank 64 \
        --num_coupling_layers 10 \
        --dia_n_blocks 6 \
        --tail_weight 0.8 \
        --tail_top_k_ratio 0.025 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 4 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --task_classes $VISA_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name VISA-ViT-100ep-TailW0.8-TopK4
}

run_gpu1_exp4() {
    echo "[GPU 1-4] VISA-ViT-100ep-Coupling12-DIA6 (Deeper Flow)"
    CUDA_VISIBLE_DEVICES=1 python run_moleflow.py \
        $BASE_OPTS $VISA_DATA \
        --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
        --num_epochs 100 \
        --lr 1e-4 \
        --lora_rank 64 \
        --num_coupling_layers 12 \
        --dia_n_blocks 6 \
        --tail_weight 0.7 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 3 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --task_classes $VISA_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name VISA-ViT-100ep-Coupling12-DIA6
}

# =============================================================================
# GPU 4: Architecture Scaling (Capacity Increase)
# =============================================================================
# LoRA rank 증가, DIA blocks 증가, Coupling layers 증가
# =============================================================================

run_gpu4_exp1() {
    echo "[GPU 4-1] VISA-ViT-LoRA128-DIA8-C12 (High Capacity)"
    CUDA_VISIBLE_DEVICES=4 python run_moleflow.py \
        $BASE_OPTS $VISA_DATA \
        --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
        --num_epochs 100 \
        --lr 8e-5 \
        --lora_rank 128 \
        --num_coupling_layers 12 \
        --dia_n_blocks 8 \
        --tail_weight 0.7 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 3 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --task_classes $VISA_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name VISA-ViT-LoRA128-DIA8-C12
}

run_gpu4_exp2() {
    echo "[GPU 4-2] VISA-ViT-LoRA128-DIA6-100ep (Higher LoRA Rank)"
    CUDA_VISIBLE_DEVICES=4 python run_moleflow.py \
        $BASE_OPTS $VISA_DATA \
        --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
        --num_epochs 100 \
        --lr 1e-4 \
        --lora_rank 128 \
        --num_coupling_layers 10 \
        --dia_n_blocks 6 \
        --tail_weight 0.7 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 3 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --task_classes $VISA_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name VISA-ViT-LoRA128-DIA6-100ep
}

run_gpu4_exp3() {
    echo "[GPU 4-3] VISA-ViT-DIA8-C10-100ep (More DIA Blocks)"
    CUDA_VISIBLE_DEVICES=4 python run_moleflow.py \
        $BASE_OPTS $VISA_DATA \
        --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
        --num_epochs 100 \
        --lr 1e-4 \
        --lora_rank 64 \
        --num_coupling_layers 10 \
        --dia_n_blocks 8 \
        --tail_weight 0.7 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 3 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --task_classes $VISA_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name VISA-ViT-DIA8-C10-100ep
}

run_gpu4_exp4() {
    echo "[GPU 4-4] VISA-ViT-Combined-Optimal (All Best Settings)"
    CUDA_VISIBLE_DEVICES=4 python run_moleflow.py \
        $BASE_OPTS $VISA_DATA \
        --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
        --num_epochs 100 \
        --lr 1e-4 \
        --lora_rank 128 \
        --num_coupling_layers 10 \
        --dia_n_blocks 6 \
        --tail_weight 0.75 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 4 \
        --lambda_logdet 1.5e-4 \
        --scale_context_kernel 6 \
        --task_classes $VISA_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name VISA-ViT-Combined-Optimal
}

# =============================================================================
# GPU 5: DINOv2 & Advanced Features
# =============================================================================
# DINOv2 backbone 시도 및 고급 기능 조합
# =============================================================================

run_gpu5_exp1() {
    echo "[GPU 5-1] VISA-DINOv2-Base (DINOv2 Backbone)"
    CUDA_VISIBLE_DEVICES=5 python run_moleflow.py \
        $BASE_OPTS $VISA_DATA \
        --backbone_name vit_base_patch14_dinov2.lvd142m \
        --num_epochs 80 \
        --lr 5e-5 \
        --lora_rank 64 \
        --num_coupling_layers 10 \
        --dia_n_blocks 6 \
        --tail_weight 0.7 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 3 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --task_classes $VISA_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name VISA-DINOv2-Base-DIA6-C10
}

run_gpu5_exp2() {
    echo "[GPU 5-2] VISA-DINOv2-Base-LogDet2e-4 (DINOv2 + Stronger LogDet)"
    CUDA_VISIBLE_DEVICES=5 python run_moleflow.py \
        $BASE_OPTS $VISA_DATA \
        --backbone_name vit_base_patch14_dinov2.lvd142m \
        --num_epochs 80 \
        --lr 5e-5 \
        --lora_rank 64 \
        --num_coupling_layers 10 \
        --dia_n_blocks 6 \
        --tail_weight 0.6 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 5 \
        --lambda_logdet 2e-4 \
        --scale_context_kernel 7 \
        --task_classes $VISA_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name VISA-DINOv2-Base-LogDet2e-4-ScaleK7
}

run_gpu5_exp3() {
    echo "[GPU 5-3] VISA-ViT-SlowStage-K3 (FAST+SLOW Training)"
    CUDA_VISIBLE_DEVICES=5 python run_moleflow.py \
        $BASE_OPTS $VISA_DATA \
        --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
        --num_epochs 80 \
        --lr 1e-4 \
        --lora_rank 64 \
        --num_coupling_layers 10 \
        --dia_n_blocks 6 \
        --tail_weight 0.7 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 3 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --enable_slow_stage \
        --slow_blocks_k 3 \
        --slow_lr_ratio 0.1 \
        --task_classes $VISA_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name VISA-ViT-SlowStage-K3
}

run_gpu5_exp4() {
    echo "[GPU 5-4] VISA-ViT-LoRA128-LogDet2e-4-ScaleK7 (Balanced Best)"
    CUDA_VISIBLE_DEVICES=5 python run_moleflow.py \
        $BASE_OPTS $VISA_DATA \
        --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
        --num_epochs 100 \
        --lr 1e-4 \
        --lora_rank 128 \
        --num_coupling_layers 10 \
        --dia_n_blocks 6 \
        --tail_weight 0.65 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 4 \
        --lambda_logdet 2e-4 \
        --scale_context_kernel 7 \
        --task_classes $VISA_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name VISA-ViT-LoRA128-LogDet2e-4-ScaleK7
}

# =============================================================================
# Sequential Execution Functions
# =============================================================================

run_gpu0_all() {
    echo "Starting GPU 0 experiments (LogDet & Scale Context)..."
    run_gpu0_exp1 && run_gpu0_exp2 && run_gpu0_exp3 && run_gpu0_exp4
    echo "GPU 0 experiments completed!"
}

run_gpu1_all() {
    echo "Starting GPU 1 experiments (Extended Training)..."
    run_gpu1_exp1 && run_gpu1_exp2 && run_gpu1_exp3 && run_gpu1_exp4
    echo "GPU 1 experiments completed!"
}

run_gpu4_all() {
    echo "Starting GPU 4 experiments (Architecture Scaling)..."
    run_gpu4_exp1 && run_gpu4_exp2 && run_gpu4_exp3 && run_gpu4_exp4
    echo "GPU 4 experiments completed!"
}

run_gpu5_all() {
    echo "Starting GPU 5 experiments (DINOv2 & Advanced)..."
    run_gpu5_exp1 && run_gpu5_exp2 && run_gpu5_exp3 && run_gpu5_exp4
    echo "GPU 5 experiments completed!"
}

# =============================================================================
# Print Experiment Summary
# =============================================================================

print_summary() {
    echo ""
    echo "============================================================="
    echo "VISA Optimization Experiment Summary (16 experiments)"
    echo "============================================================="
    echo "Target: Image AUC >= 0.910, Pixel AP >= 0.40"
    echo "Current Best: Image AUC 0.9052, Pixel AP 0.2878"
    echo "Image Size: 224 (fixed)"
    echo "============================================================="
    echo ""
    echo "GPU 0: LogDet & Scale Context (Pixel AP Focus) - 4 experiments"
    echo "  1. LogDet2e-4-ScaleK7 (100ep, TailW0.6, TopK5)"
    echo "  2. LogDet3e-4-ScaleK9 (100ep, TailW0.5, TopK5)"
    echo "  3. LogDet2e-4-ScaleK9-TopK7 (100ep, TailW0.55)"
    echo "  4. SlowStage-K4-LogDet2e-4 (80ep, FAST+SLOW)"
    echo ""
    echo "GPU 1: Extended Training (Image AUC Focus) - 4 experiments"
    echo "  1. 100ep-DIA6-C10 (Extended training)"
    echo "  2. 120ep-lr8e-5 (More epochs, lower LR)"
    echo "  3. 100ep-TailW0.8-TopK4 (Stronger tail awareness)"
    echo "  4. 100ep-Coupling12-DIA6 (Deeper flow)"
    echo ""
    echo "GPU 4: Architecture Scaling - 4 experiments"
    echo "  1. LoRA128-DIA8-C12 (High capacity)"
    echo "  2. LoRA128-DIA6-100ep (Higher LoRA rank)"
    echo "  3. DIA8-C10-100ep (More DIA blocks)"
    echo "  4. Combined-Optimal (All best settings)"
    echo ""
    echo "GPU 5: DINOv2 & Advanced - 4 experiments"
    echo "  1. DINOv2-Base-DIA6-C10 (DINOv2 backbone)"
    echo "  2. DINOv2-Base-LogDet2e-4-ScaleK7 (DINOv2 + Pixel AP opt)"
    echo "  3. SlowStage-K3 (FAST+SLOW training)"
    echo "  4. LoRA128-LogDet2e-4-ScaleK7 (Balanced best)"
    echo "============================================================="
    echo ""
    echo "Expected Outcomes:"
    echo "  - GPU 1 experiments most likely to hit Image AUC 0.910+"
    echo "  - GPU 0 experiments target Pixel AP 0.40"
    echo "  - GPU 5 DINOv2 may provide breakthrough on both metrics"
    echo "  - GPU 4 experiments explore capacity limits"
    echo "============================================================="
}

# =============================================================================
# Main Execution
# =============================================================================

case "${1:-help}" in
    "gpu0")
        run_gpu0_all
        ;;
    "gpu1")
        run_gpu1_all
        ;;
    "gpu4")
        run_gpu4_all
        ;;
    "gpu5")
        run_gpu5_all
        ;;
    "all")
        print_summary
        echo "Starting all GPU processes in parallel..."
        echo ""

        run_gpu0_all &
        PID_GPU0=$!

        run_gpu1_all &
        PID_GPU1=$!

        run_gpu4_all &
        PID_GPU4=$!

        run_gpu5_all &
        PID_GPU5=$!

        echo "Process IDs:"
        echo "  GPU 0: $PID_GPU0"
        echo "  GPU 1: $PID_GPU1"
        echo "  GPU 4: $PID_GPU4"
        echo "  GPU 5: $PID_GPU5"
        echo ""
        echo "Monitor progress:"
        echo "  tail -f $LOG_DIR/VISA-*/training.log"
        echo ""

        wait

        echo ""
        echo "============================================================="
        echo "All 16 experiments completed!"
        echo "============================================================="
        echo "Timestamp: $(date)"
        ;;
    "summary")
        print_summary
        ;;
    *)
        echo "Usage: $0 {gpu0|gpu1|gpu4|gpu5|all|summary}"
        echo ""
        echo "Options:"
        echo "  gpu0    - Run GPU 0 experiments (LogDet & Scale Context)"
        echo "  gpu1    - Run GPU 1 experiments (Extended Training)"
        echo "  gpu4    - Run GPU 4 experiments (Architecture Scaling)"
        echo "  gpu5    - Run GPU 5 experiments (DINOv2 & Advanced)"
        echo "  all     - Run all experiments in parallel"
        echo "  summary - Print experiment summary"
        echo ""
        print_summary
        ;;
esac
