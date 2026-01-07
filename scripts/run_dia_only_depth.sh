#!/bin/bash
# DIA-Only Depth Scaling Experiment
# DIA: 4, 6, 8, 10
# GPU: 0, 1, 4, 5
# NCL=0 (No MoLE blocks, DIA only)

TASK_CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
DATA_PATH="/Data/MVTecAD"
LOG_BASE="/Volume/MoLeFlow/logs/Ablation/DIA-Only-Depth"

mkdir -p $LOG_BASE

# Common settings (default) - NCL=0 for DIA-only
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
    --num_coupling_layers 0 \
    --use_dia"

run_experiment() {
    local gpu=$1
    local dia=$2
    local name="DIA-Only-DIA${dia}"
    local logfile="$LOG_BASE/gpu${gpu}_dia${dia}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu: Starting DIA=$dia"
    CUDA_VISIBLE_DEVICES=$gpu python run_moleflow.py \
        $COMMON_ARGS \
        --dia_n_blocks $dia \
        --experiment_name "$name" \
        --log_dir $LOG_BASE \
        > $logfile 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu: Completed DIA=$dia"
}

echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting DIA-Only Experiments: DIA 4, 6, 8, 10"
echo "============================================================"

run_experiment 0 4 &
PID0=$!
run_experiment 1 6 &
PID1=$!
run_experiment 4 8 &
PID4=$!
run_experiment 5 10 &
PID5=$!

echo "PIDs: GPU0=$PID0, GPU1=$PID1, GPU4=$PID4, GPU5=$PID5"
echo "Waiting for experiments to complete..."

wait $PID0 $PID1 $PID4 $PID5

echo ""
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All experiments completed!"
echo "============================================================"

# Print summary
echo ""
echo "=== Results Summary ==="
for dia in 4 6 8 10; do
    result_file="$LOG_BASE/DIA-Only-DIA${dia}/final_results.csv"
    if [ -f "$result_file" ]; then
        echo "--- DIA=$dia ---"
        tail -1 "$result_file"
    else
        echo "--- DIA=$dia: No results found ---"
    fi
done
