# **Introduction Outline**

> 핵심 논리 흐름: 배경 → 기존 방법 한계 → Isolation-Efficiency Dilemma → 핵심 통찰 (NF의 구조적 특성) → MoLE-Flow → Contributions

---

## **배경 (Continual AD의 필요성)**

- 딥러닝 기반 이상 탐지(AD)는 정상 데이터만으로 학습하는 One-Class Classification 패러다임을 통해 발전해왔음.
- **산업 현장의 현실:**
    - 제조 현장에서는 신제품 출시, 라인 확장으로 새로운 제품 클래스가 순차적으로 등장
    - 과거 데이터를 모두 축적하는 것은 저장 비용과 프라이버시 규제(GDPR, 의료영상 등)로 인해 불가능
- **파멸적 망각(Catastrophic Forgetting)의 위협:** 새로운 클래스 학습 시 이전 지식을 급격히 소실하는 문제에 직면

---

## **기존 방법론의 한계**

### 1) AD에서 망각이 치명적인 이유
- 분류 모델은 결정 경계(Decision Boundary)만 유지하면 되므로 파라미터 drift에 비교적 강건
- 반면 AD(특히 NF)는 정상 분포의 **확률 밀도(Density)**를 정밀하게 추정해야 함
- 미세한 파라미터 간섭만으로도 우도(Likelihood) 매니폴드가 붕괴 → 치명적 성능 저하
- **결론:** 망각을 '완화(Mitigate)'가 아닌 **'원천 차단(Eliminate)'**해야 함

### 2) Replay 기반 접근의 한계
- 현재 SOTA 성능의 대다수 연구(CADIC, ReplayCAD 등)는 과거 task 정보를 저장/생성하는 **Replay** 방식에 의존
- **문제점:**
    - task 수에 비례하여 **데이터 메모리 비용(Data Cost)** 증가
    - 프라이버시 규제(GDPR, 의료영상 등)와 충돌
    - **통계적 한계:** 제한된 버퍼로는 고차원 데이터의 꼬리 분포(Tail Distribution)를 커버 불가 → **편향된 밀도 추정**

### 3) Hard Isolation의 한계
- Replay 대신 모델 파라미터를 물리적으로 분리하는 **파라미터 격리(Parameter Isolation)** 시도 (예: SurpriseNet, Continual-MEGA)
- **문제점:**
    - task별 독립 네트워크 → **모델 크기 선형 증가 (Linear Growth)**
    - PackNet식 용량 분할 → **용량 포화 (Capacity Saturation)**

### 4) Efficient Isolation의 한계
- 모델 비용을 줄이기 위해 Feature Space에 Prompt/Adapter를 적용하는 **효율적 격리** 시도
- **문제점:**
    - 사전학습된 Backbone 표현력에 크게 의존 → **표현력 병목**
    - Prompt는 특징 공간 내 이동(shift)에는 효과적이나, 이질적인 기하학적 구조 표현에 한계
    - 여전히 **메모리 뱅크, 복잡한 라우팅 모듈** 필요 → 완전한 효율성 미달성

---

## **Isolation-Efficiency Dilemma**

결국 현재의 지속적 이상 탐지 연구들은 이상적인 해결책을 찾지 못한 채 **'격리-효율성 딜레마(Isolation-Efficiency Dilemma)'**에 직면:

| 접근법 | 장점 | 대가 |
|--------|------|------|
| **Hard Isolation** | Zero Forgetting | 모델 비용 폭발, 용량 포화 |
| **Efficient Isolation** | 파라미터 효율적 | 표현력 병목, 보조 메모리/라우팅 비용 |
| **Replay** | 성능 유지 | 데이터 비용, 프라이버시, 통계적 편향 |
| **MoLE-Flow (Ours)** | Zero Forgetting + 파라미터 효율 | Integral Components 필요 |

**핵심 질문:** 격리의 완전성(Zero Forgetting)과 효율성(Low Parameter Cost)을 동시에 달성할 수 있는가?

본 연구는 이 딜레마를 타파하기 위해, 데이터 저장 없이(No Replay) 그리고 모델의 폭발적 증가 없이(Parameter Efficient) **NF의 구조적 특성**을 활용한 새로운 해법을 제안

---

## **핵심 통찰: NF의 구조적 특성이 딜레마를 해결**

### 핵심 아이디어
"Full copy가 필요한 게 아니다. **변화가 필요한 부분만 분리**하면 된다."
- 파라미터를 `W_task = W_shared + ΔW_task`로 분해 (공유: frozen, task별: isolated, small)

### 왜 NF인가? - AD 모델별 분해 가능성 분석

| 모델 유형 | 분해 가능 | 이유 |
|-----------|:--------:|------|
| Embedding/Memory Bank (PatchCore) | ✗ | 학습 파라미터 없음, Memory = Replay |
| Reconstruction (AE, VAE) | ✗ | Encoder-Decoder 결합 → 잠재공간 정합성 붕괴 |
| Teacher-Student (STPM) | ✗ | T-S 정합성 필요 → 분해 시 불안정 |
| **Density - NF Coupling** | **✓** | **Subnet 구현과 무관하게 가역성 보장** |

### NF Coupling Layer의 수학적 기반

**Affine Coupling Layer (RealNVP, 2017):**
```
y₁ = x₁
y₂ = x₂ ⊙ exp(s(x₁)) + t(x₁)
```
여기서 `s(·), t(·): ℝᵈ → ℝᵈ`는 **임의의 함수(Arbitrary Functions)**

**핵심 성질 - Arbitrary Function Property:**
- `s`, `t`가 어떤 함수든 상관없이 **가역성(Invertibility)** 보장
- **Tractable Jacobian:** `log|det J| = Σ s(x₁)`로 효율적 계산

### Structural Connection: Arbitrary Function Property → CL 친화적 구조

**기존 인식:** "Subnet이 어떤 함수든 가능" = 표현력의 자유도

**우리의 재해석:** "Subnet이 어떤 함수든 가능" = **효율적 격리를 가능케 하는 구조적 기반**

**LoRA 분해의 적용:**
```
s(x) = s_base(x) + ΔS_task(x)    where ΔS = B·A (rank r << d)
t(x) = t_base(x) + ΔT_task(x)    where ΔT = B·A (rank r << d)
```

**이론적 보장 (Arbitrary Function Property로부터):**
- ✓ 가역성 유지 (`s_base + ΔS`도 여전히 임의의 함수)
- ✓ Tractable likelihood 유지
- ✓ Zero Forgetting (base frozen → 이전 task 파라미터 불변)

**남은 질문:** ΔW는 얼마나 커야 하는가? Full-rank가 필요한가, Low-rank로 충분한가?
→ "왜 Low-rank로 충분한가?" 섹션에서 별도로 답변 (NF 구조와 독립적인 질문)

---

## **MoLE-Flow 설계**

### Feature-level Adapter와의 핵심 차이

| 위치 | 영향 | 결과 |
|------|------|------|
| **Feature-level** | Density manifold의 **입력**을 교란 | 밀도 추정 왜곡 가능 |
| **NF Coupling-level** | Subnet **내부**의 transformation만 변경 | Arbitrary Function Property로 가역성 보장 |

→ MoLE-Flow는 NF Coupling-level에서 LoRA를 적용하여 가역성을 보장하면서 효율적 격리 달성

### 핵심 설계: Coupling Subnet = Frozen Base + Task-specific LoRA
- Task 0: Base + LoRA_0 함께 학습 → 이후 Base 동결
- Task 1+: LoRA_t만 학습 (Base 재사용)
- **결과:** Task당 8% 파라미터로 완전한 격리 → 딜레마 해결

### 왜 Low-rank로 충분한가? (High-level Intuition)

- Pre-trained backbone이 이미 의미 있는 feature를 추출 (task-agnostic representation)
- NF의 역할은 이 **feature space → Gaussian space로의 mapping**을 학습하는 것
- **가설:** 이 mapping의 기본 구조는 task 간 공유 가능하며, task별 차이는 low-rank adaptation으로 충분
- **실험적 검증 (Sec. 4):**
    - SVD 분석: Full fine-tuning 시 ΔW의 rank 64가 ~95% 에너지 포착 → intrinsically low-rank
    - Task 0 Invariance: 어떤 task로 base를 학습해도 최종 성능 유사 (I-AUC 편차 < 0.2%)
- → 상세 분석은 **Method Section 3.3** 참조

### Integral Components (Introduction에서는 간략히)
- Base freeze + LoRA 구조는 Zero Forgetting을 보장하지만, 두 가지 제약에서 오는 side-effect 존재:
    - **Base Freeze 제약:** Task 간 입력 분포 차이를 frozen base가 흡수하지 못함
    - **Low-rank 제약:** LoRA만으로는 복잡한 local transformation 표현에 한계
- 이를 보상하기 위한 세 가지 구성요소:
    - **Whitening Adapter (WA):** Task 간 feature distribution shift 정규화
    - **Tail-Aware Loss (TAL):** Frozen base의 tail region 학습 부족 보완
    - **Dense Intermediate Attention (DIA):** LoRA의 low-rank 표현력 한계 보충
- **핵심:** Interaction Effect Analysis로 구조적 필수요소임을 검증 (Method에서 상세 설명)

### Task-Agnostic Inference
- Prototype 기반 라우팅으로 task ID 없이 자동 expert 선택
- **결과:** 100% 라우팅 정확도 (MVTec AD)

---

## **Contributions**

1. **Structural Connection:** NF coupling layer의 Arbitrary Function Property와 CL의 파라미터 분해 요구사항 사이의 구조적 연결을 확립. 이를 통해 **zero-forgetting CL이 이론적으로 가능함**을 보이고, **O(rank) 효율의 충분성을 실험적으로 검증**.

2. **Parameter-Efficient Isolation Framework:** MoLE-Flow 제안 - coupling subnet을 frozen shared base와 task-specific LoRA로 분해하여, **task당 8% 파라미터로 완전한 격리** 달성.

3. **Integral Components with Systematic Validation:** Base Freeze의 부작용을 보상하는 세 가지 구성요소(WA, TAL, DIA)를 제안하고, **Interaction Effect Analysis**를 통해 구조적 필수요소임을 검증.

4. **State-of-the-Art Results:** MVTec-AD (15 classes, 5 runs)에서 **98.05±0.12% I-AUC, 55.80±0.35% P-AP**를 **Zero Forgetting**으로 달성.

---
---

# **Method Outline**

> 핵심 논리 흐름: Overview → MoLE Block 구조 → Why Freezing Works (설계 정당화) → Integral Components → Task-Agnostic Inference

---

## **3.1 Overview**

- 전체 파이프라인 설명
- Pre-trained Backbone → Feature Extraction → MoLE-Flow (NF with LoRA) → Density Estimation

---

## **3.2 MoLE Block: LoRA-Integrated Coupling Layer**

### Coupling Subnet 구조
```
s(x) = s_base(x) + ΔS_task(x)    where ΔS = B·A (rank r << d)
t(x) = t_base(x) + ΔT_task(x)    where ΔT = B·A (rank r << d)
```

### 학습 전략
- **Task 0:** Base + LoRA_0 함께 학습 → Base 동결
- **Task 1+:** LoRA_t만 학습 (Base 재사용)

---

## **3.3 Why Freezing Base Works: Theoretical Foundation**

> 핵심 질문: (1) Base를 frozen 해도 되는 이유? (2) Task 1개만 학습해도 되는 이유?

### 3.3.1 Arbitrary Function Property가 Freezing을 가능케 함

**Affine Coupling Layer의 핵심 성질:**
```
y₁ = x₁
y₂ = x₂ ⊙ exp(s(x₁)) + t(x₁)
```
- `s`, `t`가 **어떤 함수든** 가역성(Invertibility) 보장
- 따라서 `s_base + ΔS`도 여전히 valid → **Base frozen + LoRA adaptation이 NF의 이론적 보장을 깨지 않음**

### 3.3.2 왜 Low-rank Adaptation으로 충분한가?

**1) 핵심 통찰:**
- Pre-trained backbone이 이미 **task-agnostic한 의미 있는 feature**를 추출
- NF의 역할은 이 feature space를 **Gaussian space로 mapping**하는 것
- **핵심:** 이 mapping 구조 자체가 task-specific 정보에 크게 의존하지 않음
- → Base를 한 task에서 학습해도, low-rank adaptation만으로 다른 task에 적용 가능

**2) 이론적 기반:**

**(a) Pre-trained Feature의 Task-Agnostic 특성**

Transfer learning 연구들은 ImageNet pre-trained feature가 다양한 downstream task에서 높은 전이성을 보임을 입증:
- **Kornblith et al. [CVPR 2019]**: ImageNet accuracy와 12개 downstream task의 transfer accuracy가 r=0.96 이상의 강한 상관관계
- **Neyshabur et al. [NeurIPS 2020]**: Transfer의 핵심은 **feature reuse**이며, lower layer feature는 도메인에 무관하게 재사용됨

→ Pre-trained feature는 특정 task에 국한되지 않은 **범용적 표현(universal representation)**을 제공

**(b) NF의 역할: Feature → Gaussian Mapping**

NF 기반 AD 방법들의 성공이 이를 뒷받침:
- **CFLOW-AD [WACV 2022]**, **FastFlow [2021]**: Pre-trained encoder + NF로 feature 공간에서 density estimation 수행
- **PaDiM [ICPR 2021]**, **PatchCore [CVPR 2022]**: Pre-trained feature의 분포만 모델링하여 SOTA 달성

→ NF가 학습하는 것은 "특정 제품의 정상성"이 아니라, **feature 분포를 Gaussian으로 변환하는 일반적 구조**

**(c) 논리 도출**

(a)와 (b)를 종합:
- Pre-trained feature가 이미 task-agnostic representation 제공
- NF는 이 위에서 feature → Gaussian mapping을 학습
- 이 mapping의 **기본 구조(base)**는 task 간 공유 가능
- Task별 차이는 **low-rank adaptation(ΔW)**으로 보정 가능

**(d) 보조적 근거: Low-rank Adaptation의 일반적 효과성**

이러한 적응이 low-rank로 충분하다는 점은 최근 연구들과도 일맥상통:
- **Aghajanyan et al. [2021]**: 사전학습이 충분할수록 intrinsic dimension이 급격히 감소
- **Hu et al. [2022]**: LoRA로 rank 4~64만으로 full fine-tuning 성능의 대부분 회복

단, 이러한 발견은 주로 분류 태스크에서 검증되었으며, **밀도 추정(density estimation)**에서의 성립 여부는 아래 실험으로 검증

**3) 실증적 검증 - SVD 분석 (Sec. 4.X):**
- 학습된 LoRA weight (ΔW = B @ A)의 singular value 분석 (leather, grid, transistor)
- **결과:**
  - Task 0 (Base + LoRA 동시학습): Effective Rank (95%) = **14.5 ± 8.6**
  - Task 1, 2 (LoRA만 학습): Effective Rank (95%) = **1.3~1.5** (극히 저차원!)
  - Energy at rank 64 = **100%** (설정된 rank가 과잉)
- → 실제 필요한 adaptation이 intrinsically low-rank임을 확인
- → **AD에서도 low-rank hypothesis 성립, 오히려 더 강하게 성립**

**4) 실증적 검증 - Task 0 Invariance (Sec. 4.X):**
- 서로 다른 task(leather, grid, transistor)를 Task 0로 설정하여 전체 파이프라인 실행
- **결과:** 최종 성능이 Task 0 선택에 robust (I-AUC 편차 < 0.2%)
- → Base가 task-specific knowledge가 아닌 **feature → Gaussian mapping의 general structure**를 학습

**5) 해석:**
- "Base가 보편적 정상성 개념을 학습"이라는 강한 주장 대신,
- **"NF의 feature → Gaussian mapping 구조가 task 간 전이 가능"**함을 실험적으로 입증
- Rank sensitivity (Sec. 4.4): rank 64에서 성능 포화 → low-rank sufficiency 재확인

---

## **3.4 Integral Components**

> Base freeze + LoRA 구조의 두 가지 제약에서 오는 side-effect를 보상

### 제약 1: Base Freeze의 한계
Frozen base는 새로운 task의 입력 분포 차이를 흡수하지 못하고, tail region 학습이 약화됨

### 제약 2: Low-rank의 한계
LoRA의 저차원 적응만으로는 복잡한 local transformation을 충분히 표현하기 어려움

---

### 3.4.1 Whitening Adapter (WA)
- **원인:** Task 간 feature distribution이 다름 → Frozen base가 이를 보정하지 못함
- **해결:** Feature whitening으로 입력 분포를 정규화하여 base에 일관된 입력 제공

### 3.4.2 Tail-Aware Loss (TAL)
- **원인:** Base frozen 시 tail region(low-density 영역)의 gradient가 약화
- **해결:** Tail region에 가중치를 부여하여 학습 신호 강화

### 3.4.3 Dense Intermediate Attention (DIA)
- **원인:** LoRA의 low-rank 제약 → local/fine-grained transformation 표현력 부족
- **해결:** Intermediate feature에 attention을 적용하여 local capacity 보충

### 3.4.4 Interaction Effect Analysis
- 각 component가 **독립적 기여**가 아닌 **구조적 필수요소**임을 검증
- Ablation: 개별 제거 시 성능 저하 패턴 분석

---

## **3.5 Task-Agnostic Inference**

### Prototype-based Routing
- 각 task의 feature prototype (mean, covariance) 저장
- Inference 시 Mahalanobis distance로 가장 가까운 expert 선택
- **결과:** 100% 라우팅 정확도 (MVTec AD)
