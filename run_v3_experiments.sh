#!/bin/bash
# V3 Ablation Experiments for WhiteningAdapter and MS-Context
# 각 모듈의 개별 효과 및 조합 효과 테스트

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=============================================="
echo "V3 Ablation Experiments"
echo "GPU: $GPU_ID"
echo "=============================================="

# ======================================================================
# Part 1: Individual Module Tests (WhiteningAdapter / MS-Context)
# ======================================================================

# 1-1. WhiteningAdapter only
echo ""
echo "[1/8] WhiteningAdapter only..."
python run_moleflow.py \
    --use_whitening_adapter \
    --experiment_name V3-WhiteningAdapter_only

# 1-2. MS-Context only
echo ""
echo "[2/8] MS-Context only..."
python run_moleflow.py \
    --use_ms_context \
    --experiment_name V3-MSContext_only

# ======================================================================
# Part 2: WhiteningAdapter Combinations
# ======================================================================

# 2-1. WhiteningAdapter + DIA
echo ""
echo "[3/8] WhiteningAdapter + DIA..."
python run_moleflow.py \
    --use_whitening_adapter \
    --use_dia \
    --experiment_name V3-WhiteningAdapter_DIA

# 2-2. WhiteningAdapter + OGP
echo ""
echo "[4/8] WhiteningAdapter + OGP..."
python run_moleflow.py \
    --use_whitening_adapter \
    --use_ogp \
    --experiment_name V3-WhiteningAdapter_OGP

# 2-3. WhiteningAdapter + DIA + OGP
echo ""
echo "[5/8] WhiteningAdapter + DIA + OGP..."
python run_moleflow.py \
    --use_whitening_adapter \
    --use_dia \
    --use_ogp \
    --experiment_name V3-WhiteningAdapter_DIA_OGP

# ======================================================================
# Part 3: MS-Context Combinations
# ======================================================================

# 3-1. MS-Context + DIA
echo ""
echo "[6/8] MS-Context + DIA..."
python run_moleflow.py \
    --use_ms_context \
    --use_dia \
    --experiment_name V3-MSContext_DIA

# 3-2. MS-Context + OGP
echo ""
echo "[7/8] MS-Context + OGP..."
python run_moleflow.py \
    --use_ms_context \
    --use_ogp \
    --experiment_name V3-MSContext_OGP

# 3-3. MS-Context + DIA + OGP
echo ""
echo "[8/8] MS-Context + DIA + OGP..."
python run_moleflow.py \
    --use_ms_context \
    --use_dia \
    --use_ogp \
    --experiment_name V3-MSContext_DIA_OGP

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="

# ======================================================================
# Results Summary
# ======================================================================
echo ""
echo "Collecting results..."
python -c "
import pandas as pd
import os
from pathlib import Path

log_dir = Path('logs')
experiments = [
    'V3-WhiteningAdapter_only',
    'V3-MSContext_only',
    'V3-WhiteningAdapter_DIA',
    'V3-WhiteningAdapter_OGP',
    'V3-WhiteningAdapter_DIA_OGP',
    'V3-MSContext_DIA',
    'V3-MSContext_OGP',
    'V3-MSContext_DIA_OGP',
]

results = []
for exp in experiments:
    csv_path = log_dir / exp / 'final_results.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        mean_row = df[df['Class Name'] == 'Overall']
        if not mean_row.empty:
            results.append({
                'Experiment': exp,
                'Image_AUC': mean_row['Image AUC'].values[0],
                'Pixel_AUC': mean_row['Pixel AUC'].values[0],
                'Image_AP': mean_row['Image AP'].values[0],
                'Pixel_AP': mean_row['Pixel AP'].values[0],
            })

if results:
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values('Image_AUC', ascending=False)
    print('\n' + '='*80)
    print('V3 Experiment Results Summary')
    print('='*80)
    print(summary_df.to_string(index=False))
    summary_df.to_csv('logs/v3_experiments_summary.csv', index=False)
    print('\nSummary saved to: logs/v3_experiments_summary.csv')
else:
    print('No results found yet.')
"
