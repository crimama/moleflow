#!/usr/bin/env python3
"""
Extract all experiment results and hyperparameters to CSV.
Flattens nested ablation config and parses params from experiment names.
"""

import os
import json
import csv
import re
from pathlib import Path

LOG_DIR = Path("/Volume/MoLeFlow/logs/Final")
OUTPUT_FILE = Path("/Volume/MoLeFlow/documents/ex_result.csv")


def parse_experiment_name(exp_name):
    """Parse hyperparameters encoded in experiment name."""
    parsed = {}

    # TailW0.7 -> tail_weight = 0.7
    match = re.search(r'TailW([\d.]+)', exp_name)
    if match:
        parsed['tail_weight'] = float(match.group(1))

    # TopK3 or TopK5 -> score_aggregation_top_k = 3 or 5
    # Use negative lookbehind to avoid matching TailTopK
    match = re.search(r'(?<!Tail)TopK(\d+)', exp_name)
    if match:
        parsed['score_aggregation_top_k'] = int(match.group(1))

    # TailTopK2 or TailTopK3 -> tail_top_k (count, not ratio)
    match = re.search(r'TailTopK(\d+)', exp_name)
    if match:
        parsed['tail_top_k'] = int(match.group(1))

    # ScaleK5 or ScaleCtxK5 -> scale_context_kernel = 5
    match = re.search(r'Scale(?:Ctx)?K(\d+)', exp_name)
    if match:
        parsed['scale_context_kernel_from_name'] = int(match.group(1))

    # LogdetReg1e-4 or LogdetReg2e-4 -> lambda_logdet
    match = re.search(r'Logdet(?:Reg)?([\d.e-]+)', exp_name, re.IGNORECASE)
    if match:
        try:
            parsed['lambda_logdet_from_name'] = float(match.group(1))
        except ValueError:
            pass

    # DIA6 or DIA4 -> dia_n_blocks
    match = re.search(r'DIA(\d+)', exp_name)
    if match:
        parsed['dia_n_blocks_from_name'] = int(match.group(1))

    # LoRA128 -> lora_rank
    match = re.search(r'LoRA(\d+)', exp_name)
    if match:
        parsed['lora_rank_from_name'] = int(match.group(1))

    # Coupling12 or C10 -> num_coupling_layers
    match = re.search(r'(?:Coupling|C)(\d+)', exp_name)
    if match:
        val = int(match.group(1))
        if val >= 6:  # Reasonable coupling layer count
            parsed['num_coupling_layers_from_name'] = val

    # lr3e-4 -> lr
    match = re.search(r'lr([\d.e-]+)', exp_name, re.IGNORECASE)
    if match:
        try:
            parsed['lr_from_name'] = float(match.group(1))
        except ValueError:
            pass

    # epochs: 60ep or 80ep
    match = re.search(r'(\d+)ep', exp_name)
    if match:
        parsed['num_epochs_from_name'] = int(match.group(1))

    return parsed


def extract_results(exp_dir):
    """Extract Image AUC and Pixel AP from final_results.csv"""
    results_file = exp_dir / "final_results.csv"
    if not results_file.exists():
        return None

    with open(results_file, 'r') as f:
        lines = f.readlines()

    # Last line should be Mean/Overall
    for line in reversed(lines):
        if 'Mean' in line or 'Overall' in line:
            parts = line.strip().split(',')
            try:
                img_auc = float(parts[3].replace('*', ''))
                pix_ap = float(parts[6].replace('*', ''))
                pix_auc = float(parts[4].replace('*', '')) if len(parts) > 4 else None
                routing_acc = float(parts[2].replace('*', '')) if len(parts) > 2 else None
                return {
                    'img_auc': img_auc,
                    'pix_auc': pix_auc,
                    'pix_ap': pix_ap,
                    'routing_acc': routing_acc
                }
            except (ValueError, IndexError):
                pass
    return None


def flatten_config(config, exp_name):
    """Flatten config, extracting ablation dict and parsing experiment name."""
    flat = {}

    # First, parse values from experiment name
    name_params = parse_experiment_name(exp_name)

    for key, value in config.items():
        if key == 'ablation' and isinstance(value, dict):
            for abl_key, abl_val in value.items():
                flat[abl_key] = abl_val
        elif key == 'task_classes' and isinstance(value, list):
            flat['num_classes'] = len(value)
        elif isinstance(value, (list, dict)):
            flat[key] = str(value)
        else:
            flat[key] = value

    # Override/add values from experiment name parsing
    # These take precedence as they're more explicit
    if 'tail_weight' in name_params:
        flat['tail_weight'] = name_params['tail_weight']
    if 'score_aggregation_top_k' in name_params:
        flat['score_aggregation_top_k'] = name_params['score_aggregation_top_k']
    if 'tail_top_k' in name_params:
        flat['tail_top_k'] = name_params['tail_top_k']

    # Use name values if config doesn't have them
    if 'scale_context_kernel_from_name' in name_params and flat.get('scale_context_kernel') is None:
        flat['scale_context_kernel'] = name_params['scale_context_kernel_from_name']
    if 'lambda_logdet_from_name' in name_params and flat.get('lambda_logdet') is None:
        flat['lambda_logdet'] = name_params['lambda_logdet_from_name']
    if 'dia_n_blocks_from_name' in name_params and flat.get('dia_n_blocks') is None:
        flat['dia_n_blocks'] = name_params['dia_n_blocks_from_name']
    if 'lora_rank_from_name' in name_params and flat.get('lora_rank') is None:
        flat['lora_rank'] = name_params['lora_rank_from_name']
    if 'num_coupling_layers_from_name' in name_params and flat.get('num_coupling_layers') is None:
        flat['num_coupling_layers'] = name_params['num_coupling_layers_from_name']
    if 'lr_from_name' in name_params and flat.get('lr') is None:
        flat['lr'] = name_params['lr_from_name']
    if 'num_epochs_from_name' in name_params and flat.get('num_epochs') is None:
        flat['num_epochs'] = name_params['num_epochs_from_name']

    return flat


def main():
    experiments = []
    all_keys = set()

    for exp_name in sorted(os.listdir(LOG_DIR)):
        exp_dir = LOG_DIR / exp_name
        if not exp_dir.is_dir():
            continue

        config_file = exp_dir / "config.json"
        if not config_file.exists():
            continue

        with open(config_file, 'r') as f:
            config = json.load(f)

        results = extract_results(exp_dir)
        if results is None:
            continue

        flat_config = flatten_config(config, exp_name)
        entry = {'experiment_name': exp_name}  # Use folder name, not config name
        entry.update(flat_config)
        entry['experiment_name'] = exp_name  # Override with folder name again
        entry['Image_AUC'] = results['img_auc']
        entry['Pixel_AUC'] = results.get('pix_auc')
        entry['Pixel_AP'] = results['pix_ap']
        entry['Routing_Acc'] = results.get('routing_acc')

        experiments.append(entry)
        all_keys.update(entry.keys())

    # Define column order
    priority_cols = [
        'experiment_name',
        'dataset',
        'backbone_name',
        'num_classes',
        'num_epochs',
        'lr',
        'batch_size',
        'seed',
        # Model architecture
        'lora_rank',
        'num_coupling_layers',
        'dia_n_blocks',
        # Loss settings
        'lambda_logdet',
        # Tail-aware loss (NEW)
        'tail_weight',
        'score_aggregation_top_k',
        'tail_top_k',
        # Context settings
        'scale_context_kernel',
        'spatial_context_kernel',
        # Module toggles
        'use_lora',
        'use_dia',
        'use_whitening_adapter',
        'use_router',
        'use_pos_embedding',
        'use_spatial_context',
        'use_scale_context',
        'use_task_adapter',
        # Other settings
        'adapter_mode',
        'cl_scenario',
        # Results
        'Image_AUC',
        'Pixel_AUC',
        'Pixel_AP',
        'Routing_Acc'
    ]

    skip_keys = {'data_path', 'device', 'timestamp', 'task_classes', 'img_size', 'msk_size',
                 'embed_dim', 'enable_slow_stage', 'slow_blocks_k', 'slow_lr_ratio',
                 'soft_ln_init_scale', 'use_task_bias', 'use_mahalanobis', 'backbone_type',
                 'spatial_context_mode', 'use_ms_context', 'use_ogp', 'ogp_threshold',
                 'ogp_max_rank', 'use_feature_bank', 'use_distillation', 'use_ewc',
                 'use_hybrid_routing', 'use_regional_prototype'}

    columns = []
    for c in priority_cols:
        if c in all_keys and c not in skip_keys:
            columns.append(c)

    remaining = sorted(all_keys - set(columns) - skip_keys)
    columns.extend(remaining)

    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for exp in experiments:
            writer.writerow(exp)

    print(f"Extracted {len(experiments)} experiments to {OUTPUT_FILE}")
    print(f"Columns ({len(columns)}): {', '.join(columns)}")

    # Show sample of tail-aware loss params
    print("\nSample tail-aware loss params:")
    for exp in experiments[:5]:
        tw = exp.get('tail_weight', '-')
        topk = exp.get('score_aggregation_top_k', '-')
        ttk = exp.get('tail_top_k', '-')
        print(f"  {exp['experiment_name'][:50]}: TailW={tw}, TopK={topk}, TailTopK={ttk}")


if __name__ == "__main__":
    main()
