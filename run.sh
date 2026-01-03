#!/bin/bash

# =============================================================
# MoLE-Flow: Pixel AP 0.6 달성을 위한 최적화 실험
# =============================================================
# 현재 최고: Pixel AP 0.5449 (TailW0.75-TopK5-TailTopK2-ScaleK5)
# 목표: Image AUC >= 0.98 유지, Pixel AP >= 0.6 달성
# =============================================================

echo "============================================================="
echo "MoLE-Flow: Experiments for Pixel AP 0.6+"
echo "Current Best: Pixel AP 0.5449"
echo "Target: Image AUC >= 0.98, Pixel AP >= 0.60"
echo "============================================================="
echo "Timestamp: $(date)"
echo ""

# MVTec classes (15 classes)
MVTEC_CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"

# Base configuration
BASE_OPTS="--use_whitening_adapter --use_dia --use_tail_aware_loss"
MVTEC_DATA="--dataset mvtec --data_path /Data/MVTecAD"

# Output directory
LOG_DIR="./logs/Final"
mkdir -p $LOG_DIR

# =============================================================
# GPU 0: 우선순위 1 - 가장 유망한 조합 (예상 Pixel AP: 0.555-0.57)
# TailW0.85 + TailTopK2% + LogdetReg2e-4 + lr3e-4
# =============================================================
run_gpu0() {
    echo "[GPU 0] Priority 1: Most Promising Configuration"
    echo "[GPU 0] Expected: Pixel AP 0.555-0.57, Image AUC ~0.981"

    CUDA_VISIBLE_DEVICES=0 python run_moleflow.py \
        $BASE_OPTS $MVTEC_DATA \
        --backbone_name wide_resnet50_2 \
        --num_epochs 60 \
        --lr 3e-4 \
        --lora_rank 64 \
        --num_coupling_layers 8 \
        --dia_n_blocks 4 \
        --tail_weight 0.85 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 5 \
        --lambda_logdet 2e-4 \
        --scale_context_kernel 5 \
        --task_classes $MVTEC_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name PixelAP-Target-v1-TailW0.85-TailTopK2-LogdetReg2e-4-lr3e-4

    echo "[GPU 0] Experiment completed!"
}

# =============================================================
# GPU 1: 우선순위 2 - 공격적 설정 (예상 Pixel AP: 0.56-0.58)
# TailW0.9 + TailTopK2% + LogdetReg2e-4 + 80epochs
# =============================================================
run_gpu1() {
    echo "[GPU 1] Priority 2: Aggressive Configuration"
    echo "[GPU 1] Expected: Pixel AP 0.56-0.58, Image AUC ~0.978 (slight drop possible)"

    CUDA_VISIBLE_DEVICES=1 python run_moleflow.py \
        $BASE_OPTS $MVTEC_DATA \
        --backbone_name wide_resnet50_2 \
        --num_epochs 80 \
        --lr 3e-4 \
        --lora_rank 64 \
        --num_coupling_layers 8 \
        --dia_n_blocks 4 \
        --tail_weight 0.9 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 5 \
        --lambda_logdet 2e-4 \
        --scale_context_kernel 5 \
        --task_classes $MVTEC_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name PixelAP-Target-v2-TailW0.9-TailTopK2-LogdetReg2e-4-80ep

    echo "[GPU 1] Experiment completed!"
}

# =============================================================
# GPU 4: 우선순위 3 - 안전한 조합 (예상 Pixel AP: 0.55-0.56)
# TailW0.8 + TailTopK2% + lr3e-4 (Image AUC 안정적 유지)
# =============================================================
run_gpu4() {
    echo "[GPU 4] Priority 3: Safe Configuration"
    echo "[GPU 4] Expected: Pixel AP 0.55-0.56, Image AUC ~0.982 (stable)"

    CUDA_VISIBLE_DEVICES=4 python run_moleflow.py \
        $BASE_OPTS $MVTEC_DATA \
        --backbone_name wide_resnet50_2 \
        --num_epochs 60 \
        --lr 3e-4 \
        --lora_rank 64 \
        --num_coupling_layers 8 \
        --dia_n_blocks 4 \
        --tail_weight 0.8 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 5 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --task_classes $MVTEC_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name PixelAP-Target-v3-TailW0.8-TailTopK2-lr3e-4

    echo "[GPU 4] Experiment completed!"
}

# =============================================================
# Print Experiment Summary
# =============================================================
print_summary() {
    echo "============================================================="
    echo "3 Experiments for Pixel AP 0.6+ (3 GPUs)"
    echo "============================================================="
    echo ""
    echo "Current Best: Pixel AP 0.5449"
    echo "Target: Pixel AP >= 0.60, Image AUC >= 0.98"
    echo ""
    echo "Key Changes from Current Best:"
    echo "  - tail_weight: 0.55 -> 0.8~0.9 (핵심 변경)"
    echo "  - tail_top_k_ratio: 5% -> 2% (더 집중된 학습)"
    echo "  - logdet_reg: 1e-4 -> 2e-4 (안정성 향상)"
    echo ""
    echo "============================================================="
    echo "GPU 0 - Priority 1 (Most Promising):"
    echo "  Config: TailW0.85 + TailTopK2% + LogdetReg2e-4 + lr3e-4"
    echo "  Expected: Pixel AP 0.555-0.57, Image AUC ~0.981"
    echo ""
    echo "GPU 1 - Priority 2 (Aggressive):"
    echo "  Config: TailW0.9 + TailTopK2% + LogdetReg2e-4 + 80ep"
    echo "  Expected: Pixel AP 0.56-0.58, Image AUC ~0.978"
    echo ""
    echo "GPU 4 - Priority 3 (Safe):"
    echo "  Config: TailW0.8 + TailTopK2% + lr3e-4"
    echo "  Expected: Pixel AP 0.55-0.56, Image AUC ~0.982"
    echo "============================================================="
}

# =============================================================
# Main Execution
# =============================================================
print_summary

echo ""
echo "Starting all GPU processes in parallel..."
echo ""

# Start all GPU processes in parallel
run_gpu0 &
PID_GPU0=$!

run_gpu1 &
PID_GPU1=$!

run_gpu4 &
PID_GPU4=$!

echo "Process IDs:"
echo "  GPU 0 (Priority 1): $PID_GPU0"
echo "  GPU 1 (Priority 2): $PID_GPU1"
echo "  GPU 4 (Priority 3): $PID_GPU4"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_DIR/PixelAP-Target-*/training.log"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""

# Wait for all processes to complete
wait

echo ""
echo "============================================================="
echo "All 3 Experiments Completed!"
echo "============================================================="
echo "Timestamp: $(date)"
echo ""

# Print results summary
echo "Results Summary:"
echo ""
printf "%-60s | %-10s | %-10s | %-10s\n" "Experiment" "Image AUC" "Pixel AUC" "Pixel AP"
printf "%-60s-+-%-10s-+-%-10s-+-%-10s\n" "------------------------------------------------------------" "----------" "----------" "----------"

for exp in $LOG_DIR/PixelAP-Target-v*; do
    if [ -d "$exp" ] && [ -f "$exp/final_results.csv" ]; then
        name=$(basename "$exp")
        overall=$(grep "Overall" "$exp/final_results.csv" 2>/dev/null | tail -1)
        if [ -n "$overall" ]; then
            img_auc=$(echo "$overall" | cut -d',' -f4)
            pix_auc=$(echo "$overall" | cut -d',' -f6)
            pix_ap=$(echo "$overall" | cut -d',' -f7)
            printf "%-60s | %-10s | %-10s | %-10s\n" "$name" "$img_auc" "$pix_auc" "$pix_ap"
        fi
    fi
done

echo ""
echo "============================================================="
echo "Analysis:"
echo "============================================================="
echo ""
echo "If Pixel AP < 0.6 with all experiments:"
echo "  1. Consider architecture changes:"
echo "     - Image size: 224 -> 448"
echo "     - Backbone: WRN50 -> ViT-L/DINOv2"
echo "  2. Focus on bottleneck classes:"
echo "     - screw (0.21), grid (0.27), zipper (0.35)"
echo ""
echo "Full results saved to: $LOG_DIR/"
