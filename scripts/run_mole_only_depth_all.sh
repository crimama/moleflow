#!/bin/bash
# MoLE-Only (No DIA) Depth Scaling Experiment - All batches
# NCL: 4, 6, 8, 10, 12, 14, 16, 18
# GPU: 0, 1, 4, 5
# Runs batch 2 automatically after batch 1 completes

TASK_CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
DATA_PATH="/Data/MVTecAD"
LOG_BASE="/Volume/MoLeFlow/logs/Ablation/MoLE-Only-Depth"

mkdir -p $LOG_BASE

# Common settings (default)
COMMON_ARGS="--dataset mvtec \
    --data_path $DATA_PATH \
    --task_classes $TASK_CLASSES \
    --backbone_name wide_resnet50_2 \
    --num_epochs 60 \
    --lr 3e-4 \
    --lora_rank 64 \
    --batch_size 16 \
    --use_whitening_adapter \
    --use_tail_aware_loss \
    --tail_weight 0.7 \
    --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 3 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5 \
    --no_dia"

run_experiment() {
    local gpu=$1
    local ncl=$2
    local name="MoLE-Only-NCL${ncl}"
    local logfile="$LOG_BASE/gpu${gpu}_ncl${ncl}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu: Starting NCL=$ncl"
    CUDA_VISIBLE_DEVICES=$gpu python run_moleflow.py \
        $COMMON_ARGS \
        --num_coupling_layers $ncl \
        --experiment_name "$name" \
        --log_dir $LOG_BASE \
        > $logfile 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu: Completed NCL=$ncl"
}

# ============================================================
# Batch 1: NCL 4, 6, 8, 10
# ============================================================
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Batch 1: NCL 4, 6, 8, 10"
echo "============================================================"

run_experiment 0 4 &
PID0=$!
run_experiment 1 6 &
PID1=$!
run_experiment 4 8 &
PID4=$!
run_experiment 5 10 &
PID5=$!

echo "Batch 1 PIDs: GPU0=$PID0, GPU1=$PID1, GPU4=$PID4, GPU5=$PID5"
echo "Waiting for Batch 1 to complete..."

wait $PID0 $PID1 $PID4 $PID5

echo ""
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch 1 completed!"
echo "============================================================"
echo ""

# ============================================================
# Batch 2: NCL 12, 14, 16, 18
# ============================================================
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Batch 2: NCL 12, 14, 16, 18"
echo "============================================================"

run_experiment 0 12 &
PID0=$!
run_experiment 1 14 &
PID1=$!
run_experiment 4 16 &
PID4=$!
run_experiment 5 18 &
PID5=$!

echo "Batch 2 PIDs: GPU0=$PID0, GPU1=$PID1, GPU4=$PID4, GPU5=$PID5"
echo "Waiting for Batch 2 to complete..."

wait $PID0 $PID1 $PID4 $PID5

echo ""
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All experiments completed!"
echo "============================================================"

# Print summary
echo ""
echo "=== Results Summary ==="
for ncl in 4 6 8 10 12 14 16 18; do
    result_file="$LOG_BASE/MoLE-Only-NCL${ncl}/final_results.csv"
    if [ -f "$result_file" ]; then
        echo "--- NCL=$ncl ---"
        tail -1 "$result_file"
    else
        echo "--- NCL=$ncl: No results found ---"
    fi
done
