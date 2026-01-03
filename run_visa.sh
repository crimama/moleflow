#!/bin/bash

# =============================================================
# MoLE-Flow: VisA Dataset Optimization Experiments
# =============================================================
# Current Best: Image AUC 0.8801 (ViT), Pixel AP 0.2878 (WRN50)
# Target: Image AUC >= 0.95, Pixel AP >= 0.40
# =============================================================
# Key Insight: MVTec에서 검증된 최적 설정들이 VisA에 미적용됨
#   - use_tail_aware_loss: False -> True
#   - lambda_logdet: 1e-5 -> 1e-4
#   - scale_context_kernel: 3 -> 5
#   - tail_weight: N/A -> 0.75-0.8
# =============================================================

echo "============================================================="
echo "MoLE-Flow: VisA Dataset Optimization"
echo "Current Best: Image AUC 0.8801, Pixel AP 0.2878"
echo "Target: Image AUC >= 0.95, Pixel AP >= 0.40"
echo "============================================================="
echo "Timestamp: $(date)"
echo ""

# VisA classes (12 classes)
VISA_CLASSES="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"

# Base configuration (MVTec에서 검증된 최적 설정 적용)
BASE_OPTS="--use_whitening_adapter --use_dia --use_tail_aware_loss"
VISA_DATA="--dataset visa --data_path /Data/VISA"

# Output directory
LOG_DIR="./logs/Final"
mkdir -p $LOG_DIR

# =============================================================
# GPU 0: 우선순위 1 - MVTec 최적 설정 전이 (가장 유망)
# WRN50 + TailW0.8 + TopK5 + TailTopK2% + DIA5 + lr3e-4 + C10
# =============================================================
run_gpu0() {
    echo "[GPU 0] Priority 1: MVTec Optimal Settings Transfer"
    echo "[GPU 0] Expected: Image AUC 0.88-0.91, Pixel AP 0.32-0.38"

    CUDA_VISIBLE_DEVICES=0 python run_moleflow.py \
        $BASE_OPTS $VISA_DATA \
        --backbone_name wide_resnet50_2 \
        --num_epochs 80 \
        --lr 3e-4 \
        --lora_rank 128 \
        --num_coupling_layers 10 \
        --dia_n_blocks 5 \
        --tail_weight 0.8 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 5 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --task_classes $VISA_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name VISA-Optimal-TailW0.8-TopK5-TailTopK2-DIA5-C10-lr3e-4

    echo "[GPU 0] Experiment completed!"
}

# =============================================================
# GPU 1: 우선순위 2 - DIA 강화 + 안정적 설정
# WRN50 + TailW0.75 + TopK5 + TailTopK2% + DIA7 + lr2e-4 + C10
# =============================================================
run_gpu1() {
    echo "[GPU 1] Priority 2: DIA-Enhanced + Stable Settings"
    echo "[GPU 1] Expected: Image AUC 0.87-0.90, Pixel AP 0.30-0.35"

    CUDA_VISIBLE_DEVICES=1 python run_moleflow.py \
        $BASE_OPTS $VISA_DATA \
        --backbone_name wide_resnet50_2 \
        --num_epochs 80 \
        --lr 2e-4 \
        --lora_rank 128 \
        --num_coupling_layers 10 \
        --dia_n_blocks 7 \
        --tail_weight 0.75 \
        --tail_top_k_ratio 0.02 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 5 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --task_classes $VISA_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name VISA-DIA7-TailW0.75-TopK5-TailTopK2-C10-lr2e-4

    echo "[GPU 1] Experiment completed!"
}

# =============================================================
# GPU 4: 우선순위 3 - ViT Backbone (Image AUC 중시)
# ViT-Base + TailW0.7 + TopK5 + DIA4 + lr1e-4 + C8
# =============================================================
run_gpu4() {
    echo "[GPU 4] Priority 3: ViT Backbone (Image AUC Focus)"
    echo "[GPU 4] Expected: Image AUC 0.89-0.92, Pixel AP 0.28-0.35"

    CUDA_VISIBLE_DEVICES=4 python run_moleflow.py \
        $BASE_OPTS $VISA_DATA \
        --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
        --num_epochs 80 \
        --lr 1e-4 \
        --lora_rank 64 \
        --num_coupling_layers 8 \
        --dia_n_blocks 4 \
        --tail_weight 0.7 \
        --tail_top_k_ratio 0.03 \
        --score_aggregation_mode top_k \
        --score_aggregation_top_k 5 \
        --lambda_logdet 1e-4 \
        --scale_context_kernel 5 \
        --task_classes $VISA_CLASSES \
        --log_dir $LOG_DIR \
        --experiment_name VISA-ViT-TailW0.7-TopK5-DIA4-C8-lr1e-4

    echo "[GPU 4] Experiment completed!"
}

# =============================================================
# Print Experiment Summary
# =============================================================
print_summary() {
    echo "============================================================="
    echo "3 Experiments for VisA Dataset Optimization (3 GPUs)"
    echo "============================================================="
    echo ""
    echo "Current Best: Image AUC 0.8801, Pixel AP 0.2878"
    echo "Target: Image AUC >= 0.95, Pixel AP >= 0.40"
    echo ""
    echo "Key Changes from Current VisA Best:"
    echo "  - use_tail_aware_loss: False -> True (핵심 변경)"
    echo "  - lambda_logdet: 1e-5 -> 1e-4 (Pixel AP +4.15%)"
    echo "  - scale_context_kernel: 3 -> 5 (Pixel AP +2-3%)"
    echo "  - tail_weight: N/A -> 0.7-0.8 (Pixel AP +3-5%)"
    echo ""
    echo "============================================================="
    echo "GPU 0 - Priority 1 (Most Promising):"
    echo "  Config: WRN50 + TailW0.8 + TopK5 + TailTopK2% + DIA5 + lr3e-4"
    echo "  Expected: Image AUC 0.88-0.91, Pixel AP 0.32-0.38"
    echo ""
    echo "GPU 1 - Priority 2 (DIA-Enhanced + Stable):"
    echo "  Config: WRN50 + TailW0.75 + TopK5 + TailTopK2% + DIA7 + lr2e-4"
    echo "  Expected: Image AUC 0.87-0.90, Pixel AP 0.30-0.35"
    echo ""
    echo "GPU 4 - Priority 3 (Image AUC Focus):"
    echo "  Config: ViT + TailW0.7 + TopK5 + DIA4 + lr1e-4"
    echo "  Expected: Image AUC 0.89-0.92, Pixel AP 0.28-0.35"
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
echo "  tail -f $LOG_DIR/VISA-*/training.log"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""

# Wait for all processes to complete
wait

echo ""
echo "============================================================="
echo "All 3 VisA Experiments Completed!"
echo "============================================================="
echo "Timestamp: $(date)"
echo ""

# Print results summary
echo "Results Summary:"
echo ""
printf "%-65s | %-10s | %-10s | %-10s\n" "Experiment" "Image AUC" "Pixel AUC" "Pixel AP"
printf "%-65s-+-%-10s-+-%-10s-+-%-10s\n" "-----------------------------------------------------------------" "----------" "----------" "----------"

for exp in $LOG_DIR/VISA-*; do
    if [ -d "$exp" ] && [ -f "$exp/final_results.csv" ]; then
        name=$(basename "$exp")
        overall=$(grep "Overall" "$exp/final_results.csv" 2>/dev/null | tail -1)
        if [ -n "$overall" ]; then
            img_auc=$(echo "$overall" | cut -d',' -f4)
            pix_auc=$(echo "$overall" | cut -d',' -f6)
            pix_ap=$(echo "$overall" | cut -d',' -f7)
            printf "%-65s | %-10s | %-10s | %-10s\n" "$name" "$img_auc" "$pix_auc" "$pix_ap"
        fi
    fi
done

echo ""
echo "============================================================="
echo "Analysis:"
echo "============================================================="
echo ""
echo "If targets not met:"
echo "  1. Image AUC < 0.95:"
echo "     - Increase image size: --img_size 448"
echo "     - Use DINOv2 backbone: --backbone_name dinov2_vitl14"
echo "     - Model ensemble"
echo ""
echo "  2. Pixel AP < 0.4:"
echo "     - Increase lambda_logdet: 2e-4 or 3e-4"
echo "     - Increase tail_weight: 0.9 (aggressive)"
echo "     - Multi-scale evaluation ensemble"
echo ""
echo "Bottleneck Classes (need special attention):"
echo "  - macaroni1 (Image AUC 0.79, Pixel AP 0.06)"
echo "  - macaroni2 (Image AUC 0.71, Pixel AP 0.01)"
echo "  - capsules (Image AUC 0.71, Pixel AP 0.22)"
echo ""
echo "Full results saved to: $LOG_DIR/"
