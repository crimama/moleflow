# MoLE-Flow Experiment Logs (Organized)

이 디렉토리는 MoLE-Flow 프로젝트의 실험 로그를 체계적으로 정리한 것입니다.

## 디렉토리 구조

```
logs_organized/
├── 1_Main_Results/          # 논문에 사용되는 주요 결과
├── 2_Ablation/              # Ablation Study 실험
├── 3_Interaction_Effect/    # ANOVA Interaction Effect 분석
├── 4_CL_Scenarios/          # Continual Learning 시나리오 실험
├── 5_Analysis/              # 메커니즘 분석 결과
├── 6_Dataset_Specific/      # 데이터셋별 최적화 실험
├── 7_Development/           # 개발 과정 버전별 실험
└── 8_Archive/               # 기타 실험 (HP 튜닝 등)
```

---

## 1. Main Results (주요 결과)

논문의 main table에 들어갈 실험 결과들입니다.

### MVTec AD
| 실험명 | I-AUC | P-AUC | P-AP | Routing |
|--------|-------|-------|------|---------|
| MAIN (Seed 456) | 97.87% | 97.78% | 55.8% | 100% |
| MAIN (Seed 789) | 97.78% | 97.81% | 55.5% | 100% |
| WRN50-60ep-lr2e4-dia4 | 97.93% | 97.36% | 47.4% | 100% |

### VisA
`1_Main_Results/VisA_Main/` 참조

### MPDD
`1_Main_Results/MPDD_Main/` 참조

---

## 2. Ablation Study

### 2.1 Component Ablation (`Component/`)
각 컴포넌트 제거 시 성능 변화를 측정합니다.

| 실험 | 설명 |
|------|------|
| wo_LoRA | LoRA 제거 (base만 사용) |
| wo_DIA | Deep Invertible Adapter 제거 |
| wo_Adapter | Whitening Adapter 제거 |
| wo_SpatialCtx | Spatial Context Mixer 제거 |
| wo_ScaleCtx | Scale Context 제거 |
| wo_Router | Router 제거 (oracle task ID 사용) |
| wo_PosEmbed | Positional Embedding 제거 |

### 2.2 Hyperparameter Ablation (`Hyperparameter/`)
하이퍼파라미터 민감도 분석입니다.

| 파라미터 | 테스트 범위 |
|----------|-------------|
| TailW | 0.1, 0.3, 0.5, 0.7, 0.8, 1.0 |
| TailTopK | 0.01, 0.03, 0.05, 0.10 |
| LogDet | 1e-6, 1e-5, 3e-5 |
| SpatialCtxK | 3, 5, 7 |
| ScaleCtxK | 3, 5, 7 |
| ScoreTopK | 3, 5, 7, 10 |

### 2.3 LoRA Rank (`LoRA_Rank/`)
| Rank | 파라미터 수 | 성능 |
|------|-------------|------|
| 16 | 0.1M/task | ~54.8% P-AP |
| 32 | 0.2M/task | ~55.4% P-AP |
| 64 | 0.4M/task | ~55.8% P-AP |
| 128 | 0.8M/task | ~55.9% P-AP |

### 2.4 Architecture Depth (`Architecture_Depth/`)
MoLE blocks와 DIA blocks의 depth 조합 실험입니다.

| 조합 | 설명 |
|------|------|
| MoLE6-DIA2 | 기본 설정 |
| MoLE8-DIA4 | 더 깊은 구조 |
| MoLE4-DIA2 | 가벼운 구조 |

### 2.5 Loss Function (`Loss_Function/`)
Tail-Aware Loss 조합 실험입니다.

---

## 3. Interaction Effect (ANOVA 분석)

Frozen Base vs Trainable Base 조건에서 각 컴포넌트의 interaction effect를 측정합니다.

### 3class (`3class/`)
3개 클래스(leather, grid, transistor)로 빠른 검증

### 15class (`15class/`)
전체 15개 클래스로 full 검증

| 컴포넌트 | Frozen 효과 | Trainable 효과 | Interaction p-value |
|----------|-------------|----------------|---------------------|
| WA | +7.3%p | +2.1%p | 0.008* |
| TAL | +7.6%p | +3.2%p | 0.021* |
| DIA | +5.2%p | +2.1%p | 0.034* |

---

## 4. CL Scenarios (Continual Learning 시나리오)

### 4.1 1x1 Scenario (`1x1/`)
한 번에 1개 클래스씩 학습 (기본 설정)
- 다양한 seed로 재현성 검증

### 4.2 3x3 Scenario (`3x3/`)
한 번에 3개 클래스씩 학습

### 4.3 5x5 Scenario (`5x5/`)
한 번에 5개 클래스씩 학습

### 4.4 Others (`Others/`)
10x1, 14x1 등 기타 시나리오

---

## 5. Analysis (메커니즘 분석)

### SVD (`SVD/`)
- LoRA의 low-rank adaptation 분석
- Full fine-tuning weight delta의 SVD 분석
- Effective rank 측정

### Gradient (`Gradient/`)
- Gradient dynamics 분석
- Tail-Aware Loss의 gradient redistribution 효과

### Latent (`Latent/`)
- DIA 전후 latent distribution 분석
- Calibration 효과 측정

### Spatial (`Spatial/`)
- Spatial Context의 효과 분석
- Feature distribution 시각화

---

## 6. Dataset Specific (데이터셋별 최적화)

### VisA_Optimization (`VisA_Optimization/`)
VisA 데이터셋에 특화된 하이퍼파라미터 최적화 실험

---

## 7. Development (개발 버전)

프로젝트 개발 과정의 버전별 실험 기록입니다.

| 버전 | 주요 변경 |
|------|----------|
| V1 | Baseline (FastFlow + LoRA) |
| V2 | Task Adapter 추가 |
| V3 | Whitening Adapter, DIA 도입 |
| V4 | Complete Separation 구조 |
| V5 | Score Aggregation, 최적화 |
| V6 | Final 구조 확정 |

---

## 8. Archive (아카이브)

더 이상 사용하지 않거나 참고용으로 보관하는 실험들입니다.
- HP 튜닝 과정의 중간 실험
- 실패한 실험
- 기타

---

## Summary CSV 파일

각 카테고리에 `*_summary.csv` 파일이 생성되어 있습니다.

| 파일 | 설명 |
|------|------|
| MVTec_Main_summary.csv | MVTec 주요 결과 요약 |
| Component_summary.csv | Component ablation 요약 |
| Hyperparameter_summary.csv | HP ablation 요약 |
| 15class_summary.csv | IE 분석 요약 |
| ... | ... |

### CSV 컬럼 설명
- `experiment_name`: 실험 이름
- `image_auc`: Image-level AUROC
- `pixel_auc`: Pixel-level AUROC
- `pixel_ap`: Pixel-level Average Precision
- `routing_acc`: Routing Accuracy (%)
- `backbone`: 백본 네트워크
- `num_epochs`: 에포크 수
- `lr`: Learning rate
- `lora_rank`: LoRA rank
- `num_coupling_layers`: MoLE coupling layers 수
- `dia_n_blocks`: DIA blocks 수
- `tail_weight`: Tail-Aware Loss 가중치
- `lambda_logdet`: Log-det regularization 계수
- `num_tasks`: Task 수

---

## 실험 재현

```bash
# Main 실험 재현
python run_moleflow.py \
    --dataset mvtec \
    --task_classes bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper \
    --backbone_name wide_resnet50_2 \
    --num_epochs 60 \
    --lr 2e-4 \
    --lora_rank 64 \
    --num_coupling_layers 8 \
    --dia_n_blocks 2 \
    --experiment_name "MAIN_reproduce"
```

---

## 파일 구조

각 실험 폴더에는 다음 파일들이 포함됩니다:
- `config.json`: 실험 설정
- `final_results.csv`: 최종 결과 (클래스별)
- `training.log`: 학습 로그
- `diagnostics/`: 진단 플롯 (optional)

---

*Last updated: 2025-01-20*
