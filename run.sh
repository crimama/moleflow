#!/bin/bash

# MoLE-Flow: Multi-Backbone Experiment
# =============================================================
# GPU 0: VISA (ViT backbone - default)
# GPU 1: MVTecAD (wide_resnet50_2 backbone)
# GPU 4: MVTecAD (efficientnet_b7 backbone)

echo "Starting Multi-Backbone Experiments..."
echo "GPU 0: VISA (ViT) | GPU 1: MVTecAD (WideResNet50) | GPU 4: MVTecAD (EfficientNet-B7)"
echo ""

# V5-Final Configuration
V5_OPTS="--use_whitening_adapter --use_dia --score_aggregation_mode top_k --score_aggregation_top_k 3 --use_tail_aware_loss --tail_weight 0.3"

# VISA classes (12 classes)
TASK_CLASSES_VISA="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"

# MVTecAD classes (15 classes, alphabetical order)
TASK_CLASSES_MVTEC="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"

echo "V5 Options: $V5_OPTS"
echo ""

# GPU 0: VISA with ViT backbone (default)
echo "Starting VISA experiment on GPU 0 (ViT backbone)..."
CUDA_VISIBLE_DEVICES=0 python run_moleflow.py \
    $V5_OPTS \
    --dataset visa \
    --data_path /Data/VISA \
    --num_epochs 60 \
    --task_classes $TASK_CLASSES_VISA \
    --experiment_name V5-VISA-60epochs &

# GPU 1: MVTecAD with wide_resnet50_2 backbone
echo "Starting MVTecAD experiment on GPU 1 (WideResNet50)..."
CUDA_VISIBLE_DEVICES=1 python run_moleflow.py \
    $V5_OPTS \
    --dataset mvtec \
    --data_path /Data/MVTecAD \
    --backbone_name wide_resnet50_2 \
    --num_epochs 60 \
    --task_classes $TASK_CLASSES_MVTEC \
    --experiment_name V5-MVTec-WideResNet50-60epochs &

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
echo "All 3 experiments running in background."
echo "Use 'nvidia-smi' to monitor GPU usage."
echo ""
echo "Monitor logs:"
echo "  tail -f logs/V5-VISA*/V5-VISA*.log"
echo "  tail -f logs/V5-MVTec-WideResNet50*/V5-MVTec-WideResNet50*.log"
echo "  tail -f logs/V5-MVTec-EfficientNetB7*/V5-MVTec-EfficientNetB7*.log"

wait
echo "All experiments completed!"
