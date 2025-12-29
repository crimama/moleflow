#!/bin/bash

# MoLE-Flow: Version 5 Final - Best Configuration
# =============================================================
#
# Version 5 실험 요약:
#   - V5.1a: Tail-Aware Loss + Top-K aggregation (best baseline)
#   - V5.5: Position-Agnostic approaches (Dual Branch failed)
#   - V5.6: Improved Dual Branch (still failed - no gradient signal)
#   - V5.7: Multi-Orientation Ensemble (no improvement for screw)
#   - V5.8: TAPE (learned wrong direction - NLL ≠ AD performance)
#
# 결론: Position Encoding 관련 접근법은 normal-only training의 한계로 실패
# Best Config: V5.1a baseline (LocalConsistency 선택적 사용)

TASK_CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
# TASK_CLASSES="leather grid transistor screw"

echo "Starting MoLE-Flow training..."
echo "Task classes: $TASK_CLASSES"

# Best configuration from Version 5
BASELINE_OPTS="--use_whitening_adapter --use_dia --score_aggregation_mode top_k --score_aggregation_top_k 3 --use_tail_aware_loss --tail_weight 0.3"

# Main experiment: Best V5 configuration
CUDA_VISIBLE_DEVICES=1 python run_moleflow.py --run_diagnostics \
    --task_classes $TASK_CLASSES \
    $BASELINE_OPTS \
    --num_epochs 60 \
    --experiment_name Version5-Final-60epochs_all_classes_alphabet_order

echo "Training completed!"
