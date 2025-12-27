#!/bin/bash

# MoLE-Flow: Continual Anomaly Detection
# =======================================
# All settings below are now defaults - just run: python run_moleflow.py
#
# Default configuration:
# - Backbone: vit_base_patch16_224.augreg2_in21k_ft_in1k
# - Image size: 224
# - Adapter mode: soft_ln (SoftLN init_scale=0.01)
# - Spatial context: enabled (depthwise_residual, kernel=3)
# - Scale context: enabled (s-network only)

# Classes: leather grid transistor carpet zipper hazelnut toothbrush metal_nut screw wood tile capsule pill cable bottle

# Basic run with all defaults
CUDA_VISIBLE_DEVICES=0 python run_moleflow.py \
    --run_diagnostics \
    --experiment_name moleflow_run_tests