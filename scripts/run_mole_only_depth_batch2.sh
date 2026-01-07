#!/bin/bash
# MoLE-Only (No DIA) Depth Scaling Experiment - Batch 2
# NCL: 12, 14, 16, 18
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

# Second batch: NCL 12, 14, 16, 18 on GPU 0, 1, 4, 5
echo "Starting second batch: NCL 12, 14, 16, 18"

CUDA_VISIBLE_DEVICES=0 nohup python run_moleflow.py \
    $COMMON_ARGS \
    --num_coupling_layers 12 \
    --experiment_name "MoLE-Only-NCL12" \
    --log_dir $LOG_BASE \
    > $LOG_BASE/gpu0_ncl12.log 2>&1 &
echo "GPU 0: NCL=12 started (PID: $!)"

CUDA_VISIBLE_DEVICES=1 nohup python run_moleflow.py \
    $COMMON_ARGS \
    --num_coupling_layers 14 \
    --experiment_name "MoLE-Only-NCL14" \
    --log_dir $LOG_BASE \
    > $LOG_BASE/gpu1_ncl14.log 2>&1 &
echo "GPU 1: NCL=14 started (PID: $!)"

CUDA_VISIBLE_DEVICES=4 nohup python run_moleflow.py \
    $COMMON_ARGS \
    --num_coupling_layers 16 \
    --experiment_name "MoLE-Only-NCL16" \
    --log_dir $LOG_BASE \
    > $LOG_BASE/gpu4_ncl16.log 2>&1 &
echo "GPU 4: NCL=16 started (PID: $!)"

CUDA_VISIBLE_DEVICES=5 nohup python run_moleflow.py \
    $COMMON_ARGS \
    --num_coupling_layers 18 \
    --experiment_name "MoLE-Only-NCL18" \
    --log_dir $LOG_BASE \
    > $LOG_BASE/gpu5_ncl18.log 2>&1 &
echo "GPU 5: NCL=18 started (PID: $!)"

echo ""
echo "Second batch started. Monitor with:"
echo "  tail -f $LOG_BASE/gpu0_ncl12.log"
echo "  tail -f $LOG_BASE/gpu1_ncl14.log"
echo "  tail -f $LOG_BASE/gpu4_ncl16.log"
echo "  tail -f $LOG_BASE/gpu5_ncl18.log"
