#!/bin/bash

# MoLE-Flow: EfficientNet-B7 Backbone Experiment
# =============================================================
# GPU 4: MVTecAD (efficientnet_b7 backbone)

echo "Starting EfficientNet-B7 Experiment..."
echo "GPU 4: MVTecAD (EfficientNet-B7)"
echo ""

# V5-Final Configuration
V5_OPTS="--use_whitening_adapter --use_dia --score_aggregation_mode top_k --score_aggregation_top_k 3 --use_tail_aware_loss --tail_weight 0.3"

# MVTecAD classes (15 classes, alphabetical order)
TASK_CLASSES_MVTEC="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"

echo "V5 Options: $V5_OPTS"
echo ""

# GPU 4: MVTecAD with efficientnet_b7 backbone
echo "Starting MVTecAD experiment on GPU 4 (EfficientNet-B7)..."
CUDA_VISIBLE_DEVICES=4 python run_moleflow.py \
    $V5_OPTS \
    --dataset mvtec \
    --data_path /Data/MVTecAD \
    --backbone_name tf_efficientnet_b7.ns_jft_in1k \
    --num_epochs 60 \
    --task_classes $TASK_CLASSES_MVTEC \
    --experiment_name V5-MVTec-TFEfficientNetB7-60epochs &

echo ""
echo "Experiment running in background."
echo "Use 'nvidia-smi' to monitor GPU usage."
echo ""
echo "Monitor logs:"
echo "  tail -f logs/V5-MVTec-EfficientNetB7*/V5-MVTec-EfficientNetB7*.log"

wait
echo "EfficientNet-B7 experiment completed!"
