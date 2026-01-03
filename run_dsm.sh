#!/bin/bash

# MoLE-DSM: Denoising Score Matching Experiments
# =============================================================
# V7: Combines NF's conservative field with MULDE-style score matching

echo "Starting MoLE-DSM Experiments..."
echo ""

# Base V5 Configuration + DSM
BASE_OPTS="--use_whitening_adapter --score_aggregation_mode top_k --score_aggregation_top_k 3"

# Backbone Configuration
BACKBONE_OPTS="--backbone_name wide_resnet50_2"

# DSM Configuration
DSM_OPTS="--use_dsm --dsm_mode hybrid --dsm_alpha 0.7 --dsm_sigma_min 0.01 --dsm_sigma_max 1.0"

# 3 classes for initial CL test
# TASK_CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
TASK_CLASSES="cable"

echo "Base Options: $BASE_OPTS"
echo "Backbone: $BACKBONE_OPTS"
echo "DSM Options: $DSM_OPTS"
echo "Task Classes: $TASK_CLASSES"
echo ""

# Experiment 1: Hybrid DSM (alpha=0.7) with WideResNet50
echo "Experiment 1: Hybrid DSM (alpha=0.7) with WideResNet50..."
CUDA_VISIBLE_DEVICES=0 python run_moleflow.py \
    $BASE_OPTS \
    $BACKBONE_OPTS \
    $DSM_OPTS \
    --dataset mvtec \
    --data_path /Data/MVTecAD \
    --num_epochs 40 \
    --task_classes $TASK_CLASSES \
    --batch_size 64 \
    --experiment_name V7-DSM-WideResNet50-hybrid-a07

echo ""
echo "Experiment completed! Check logs at:"
echo "  logs/V7-DSM-WideResNet50-hybrid-a07-all_classes*/final_results.csv"
