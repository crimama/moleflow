#!/bin/bash
# MoLE-Only (No DIA) Depth Scaling Experiment
# NCL: 4, 6, 8, 10, 12, 14, 16, 18
# GPU: 0, 1, 4, 5

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

# First batch: NCL 4, 6, 8, 10 on GPU 0, 1, 4, 5
echo "Starting first batch: NCL 4, 6, 8, 10"

CUDA_VISIBLE_DEVICES=0 nohup python run_moleflow.py \
    $COMMON_ARGS \
    --num_coupling_layers 4 \
    --experiment_name "MoLE-Only-NCL4" \
    --log_dir $LOG_BASE \
    > $LOG_BASE/gpu0_ncl4.log 2>&1 &
echo "GPU 0: NCL=4 started (PID: $!)"

CUDA_VISIBLE_DEVICES=1 nohup python run_moleflow.py \
    $COMMON_ARGS \
    --num_coupling_layers 6 \
    --experiment_name "MoLE-Only-NCL6" \
    --log_dir $LOG_BASE \
    > $LOG_BASE/gpu1_ncl6.log 2>&1 &
echo "GPU 1: NCL=6 started (PID: $!)"

CUDA_VISIBLE_DEVICES=4 nohup python run_moleflow.py \
    $COMMON_ARGS \
    --num_coupling_layers 8 \
    --experiment_name "MoLE-Only-NCL8" \
    --log_dir $LOG_BASE \
    > $LOG_BASE/gpu4_ncl8.log 2>&1 &
echo "GPU 4: NCL=8 started (PID: $!)"

CUDA_VISIBLE_DEVICES=5 nohup python run_moleflow.py \
    $COMMON_ARGS \
    --num_coupling_layers 10 \
    --experiment_name "MoLE-Only-NCL10" \
    --log_dir $LOG_BASE \
    > $LOG_BASE/gpu5_ncl10.log 2>&1 &
echo "GPU 5: NCL=10 started (PID: $!)"

echo ""
echo "First batch started. Monitor with:"
echo "  tail -f $LOG_BASE/gpu0_ncl4.log"
echo "  tail -f $LOG_BASE/gpu1_ncl6.log"
echo "  tail -f $LOG_BASE/gpu4_ncl8.log"
echo "  tail -f $LOG_BASE/gpu5_ncl10.log"
echo ""
echo "After first batch completes, run second batch with:"
echo "  ./scripts/run_mole_only_depth_batch2.sh"
