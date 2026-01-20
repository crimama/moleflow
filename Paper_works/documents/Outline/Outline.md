# DeCoFlow: Structural Decomposition of Normalizing Flows for Continual Anomaly Detection (가제)

**Hun Im, Jungi Lee, Subeen Cha, Pilsung Kang***

---

## Index

- Abstract
- 1. Introduction
- 2. Related works
- 3. Method
- 4. Experiments
- 5. Conclusion
- 6. Future Works
- Reference

---

## Abstract

---

## 1. Introduction

**배경: Continual AD의 필요성**

- 딥러닝 기반 이상 탐지(AD)는 정상 데이터의 분포만을 학습하는 단일 클래스 분류(One-Class Classification)에서 시작하여 최근 다중 클래스로 확장되었으나, 이는 학습 시점에 모든 데이터가 수집된 정적인 환경(Static Environment)을 가정한다는 한계가 있음
- 실제 제조 현장은 새로운 제품 클래스가 순차적으로 등장하는 역동적인 특성을 가지며, 이는 정적인 학습 가정과 배치됨
- 과거의 모든 데이터를 축적하여 재학습하는 것은 기하급수적으로 증가하는 저장 비용뿐만 아니라, 프라이버시 제약 등으로 인해 현실적으로 불가능함
- 결국 모델은 이전 데이터를 볼 수 없는 상태에서 새로운 태스크를 학습해야 하며, 이 과정에서 이전에 학습한 지식을 급격히 소실하여 시스템의 신뢰성을 무너뜨리는 파멸적 망각 문제에 직면함

**기존 방법론들의 한계**

- 결정 경계만 조정하면 되는 분류 모델과 달리, 이상 탐지, 특히 Normalizing Flow 기반의 방법론은 정상 데이터의 확률 밀도 자체를 정밀하게 추정해야 함
- 따라서 미세한 파라미터 간섭만으로도 우도 매니폴드가 붕괴되어 치명적인 성능 저하를 초래하므로, AD에서는 망각을 단순히 '완화'하는 것에는 한계가 있음
- 많은 SOTA 방법들[CADIC, ReplayCAD 등]이 채택하고 있는 Replay 방식은 Task 수에 비례하여 데이터 메모리 비용이 증가하여 확장성을 허재함. 또한 제한된 버퍼 크기로는 고차원 데이터의 복잡한 꼬리 분포(Tail Distribution)를 커버할 수 없어, 결국 편향된 밀도 추정을 야기함
- 따라서 구조적으로 파라미터를 분리하여 forgetting을 피하기 위한 Parameter isolation 방법들이 시도 됨 [SurpriseNet, Continual-MEGA]
- 이러한 방법론들은 데이터 의존성을 파라미터를 물리적으로 분리하는 방식을 차용하며, Task별로 독립된 네트워크를 할당함에 따라 모델 크기의 선형적 증가를 초래하거나, 고정된 용량을 분할하는 경우 가용 파라미터가 고갈되는 용량 포화 문제에 직면
- 모델 비용을 줄이기 위해 feature space에 prompt 또는 adapter [UCAD, DER, TAPD]를 적용하는 시도가 있었으나, 적절한 프롬프트 선택을 위해 여전히 별도의 메모리 뱅크나 라우팅을 위한 추가적인 프로세스가 요구되어 완전한 효율성을 달성하지 못 함

### 1.1. The Isolation-Efficiency Trade-off

- 기존 Continual AD 연구들은 Isolation(망각 방지)과 Efficiency(비용 효율) 간의 trade-off에 직면해 있음
- 성능을 위해 완전 격리 시 모델 비용이 급증하며, 반면 효율적 격리를 할 시 정밀도 저하 및 여전히 보조 비용이 발생함
- 본 연구는 이 딜레마를 타개하기 위해, 데이터 저장 없이(No Replay) 그리고 모델의 폭발적 증가 없이(Parameter Efficient) NF의 구조적 특성을 활용한 새로운 해법을 제안

### 1.2. Key Insight: Theoretical and Empirical Foundations

**Insight 1: Structural Suitability of Normalizing Flows**

- "전체 모델을 복사할 필요 없이, 변화가 필요한 부분만 분리하면 된다."는 아이디어에 착안하여, 파라미터를 공유되는 부분($w_{shared}$)과 Task별 변화량($\Delta W_{task}$)으로 분해하는 전략을 채택: $w_{task} = w_{shared} + \Delta w_{task}$
- 전체 모델을 복사할 필요 없이, 변화가 필요한 부분만 분리하면 된다"는 아이디어를 실현하기 위해서는, 파라미터 분해가 모델의 수학적 정합성을 해치지 않아야 함
- Memory Bank 기반의 방법론들은 메모리 자체가 리플레이 문제로 직결이 됨
- Reconstruction 기반의 방법론들의 경우 인코더와 디코더가 강하게 결합되어 있어, 파라미터 분해 시 Latent Consistency가 붕괴되어 재구성이 불가능함
- Teacher-Student: teacher와 Student 간의 정교한 특징 정렬이 요구되므로, 분해 시 학습 불안정성을 초래함

**NF의 구조적 적합성 (Arbitrary Function Property)**

- 반면, NF의 Affine Coupling Layer는 $y_2 = x_2 \odot \exp(s(x_1)) + t(x_1)$ 연산에서 스케일($s$)과 이동($t$) 함수가 어떤 복잡한 형태이든 관계없이 전체 변환의 가역성(Invertibility)과 야코비안 계산의 효율성을 구조적으로 보장
- 우리는 이 Arbitrary Function Property를 효율적 격리를 위한 구조적 안전장치로 재해석했습니다. 즉, 서브넷을 $[\text{Frozen Base} + \Delta W]$ 형태로 분해하더라도 이론적 정당성이 완벽히 유지됨을 발견
- 즉 Normalizing Flow의 Coupling layer는 서브넷의 구현 방식과 무관하게 전체 변환의 가역성을 구조적으로 보장하므로 분해에 가장 적합한 모델이라 할 수 있음

**Insight 2: Intrinsic Low-Rank Nature of Adaptation (Empirical Insight)**

- 구조적으로 분해가 가능하더라도, 변화량 $\Delta W$가 커야 한다면 효율성을 달성할 수 없습니다. 우리는 "과연 $\Delta W$는 얼마나 커야 하는가?"에 대한 답을 찾고자 함
- ImageNet으로 사전 학습된 Backbone이 이미 유의미한 특징(Task-agnostic Representation)을 추출한다는 점에 주목
- NF의 역할은 새로운 특징을 학습하는 것이 아니라, 이미 주어진 공간을 가우시안분포로 매핑하는 구조적 변환에 국한
- 이 변환의 기본 골격은 Task간 공유 가능하며, Task별 차이는 Low-Rank 만으로도 충분할 것임
- 실제 학습 시 가중치 변화량에 대해 특이값 분해를 수행한 결과 전체 에너지의 74% 이상이 상위 64개의 랭크에 집중됨을 확인
- 또한 rank ablation 결과 rank 16 ~ 128에서 0.02%로 성능 차이가 미미하며, Task 적응이 본질적으로 저차원임을 확인함

### 1.3. Proposed Method: DeCoFlow

**Coupling-level Decomposition**

- DeCoFlow는 NF의 Coupling-level에서 서브넷 내부의 변환 함수(Transformation)만을 변경하는 접근을 취함. 아핀 커플링 레이어의 스케일(s)과 이동(t): $s(x) = s_{base}(x) + \Delta s_{task}(x)$, where $\Delta s = B \cdot A$ 형태로 분해하여 적용
- $s_{base}(x) + \Delta s_{task}(x)$ 또한 여전히 하나의 함수이므로, NF의 Arbitrary Function Property에 의해 전체 변환의 가역성과 야코비안 계산의 효율성이 수학적으로 보존
- Coupling subnet을 Frozen Base와 Task-specific LoRA로 이원화 하여 구성함. Task 0 이후 base weight를 동결하여 모든 Task가 공유하는 구조적 앵커로 삼고, 이후 task는 저랭크의 LoRA 모듈만 학습
- 이를 통해 Task 당 단 8%의 파라미터 추가만으로 $s_{base}$의 물리적 보존을 통한 완전한 격리를 달성함

**Integral Components for Structural Compensation**

- Base Freeze와 Low-rank 구조를 통해 안정성과 효율성을 모두 확보했으나, 고정된 베이스의 경직성, 저차원 적응의 표현력 한계, 전역적 분포 차이, 미세한 꼬리 분포를 포착하는 데에는 여전히 미세한 구조적 간극이 존재
- 이를 보강하기 위해 TSA(분포 정렬), TAL(꼬리 학습), ACL(국소 표현력 보강)을 도입함

### 1.4. Contribution

- Normalizing Flow의 Arbitrary Function Property를 통해 파라미터 분해가 가능함을 보임
- **DeCoFlow Framework**: Coupling-level LoRA 적응을 통해 데이터 저장 없이 실질적인 파라미터 격리를 달성하는 NF 기반 Continual AD 프레임워크 제안. Task당 8%의 파라미터 추가만으로 망각 없는 확장 가능
- 분포 정렬(TSA), 꼬리 학습(TAL), 국소 표현력(ACL)을 통합하여 Low-rank 적응의 성능 상한을 3%p 향상
- MVTec AD 15-class sequential learning에서 98.05% I-AUC 달성 (기존 SOTA 대비 +3.35%p)

---

## 2. Related works

### 2.1. Unified Multi-class Anomaly Detection

**Paradigm Shift to Unified Models**

- 초기의 이상 탐지 연구들은 제품 클래스(예: 나사, 병)마다 개별 모델을 학습시키는 One-Class Setting에 집중했으나, 이는 다품종 생산이 이루어지는 실제 현장에서 관리 비용을 급증시킴
- 이에 따라 UniAD [You et al., NeurIPS 2022]와 OmniAL [Zhao et al., CVPR 2023]를 기점으로, 단일 모델로 여러 클래스를 동시에 학습하는 Unified (Multi-class) AD 패러다임으로 전환됨
- 최근에는 MambaAD [He et al., NeurIPS 2024]가 State Space Model을 활용하고, DiAD [He et al., AAAI 2024]와 LafitE [Yao et al., 2023]가 Diffusion 기반 접근법을 시도하는 등 다양한 아키텍처가 제안됨

**Challenges: Interference & Identity Mapping**

- 그러나 단일 모델이 상이한 데이터 분포를 동시에 학습할 때, 클래스 간 특징이 충돌하는 'Inter-class Interference' [Lu et al., Arxiv 2024] 문제가 발생함
- 특히 Reconstruction 기반 모델들은 이상(Anomaly)까지 그대로 복원해버리는 Identity Mapping 문제에 취약하며, 이를 해결하기 위해 DecAD [Wang et al., ICCV 2025]는 복잡한 대조 학습을, Revitalizing Reconstruction [Fan et al., arXiv 2024]은 Latent Disentanglement를 필요로 함

**Normalizing Flow**

- Reconstruction 기반 모델과 달리, Normalizing Flow (NF)는 입력을 재구성하지 않고 우도(Likelihood)를 직접 최적화하므로 Identity Mapping 문제를 구조적으로 회피함
- 이러한 특성은 Dinomaly [Guo et al., CVPR 2025]의 "Less is More" 철학에 부합하며, 복잡한 보조 태스크 없이도 명확한 이상 탐지를 가능케 함
- 실제로 FastFlow [Yu et al., CVPR 2022]나 MSFlow [Zhou et al., TPAMI 2024] 등은 공간적 맥락과 다중 스케일 특징을 효과적으로 모델링하며 NF가 이상 탐지를 위한 강력한 프레임워크임을 입증해 옴
- 그러나 초기 NF 모델들은 대부분 One-class 기반으로 설계되었기 때문에, 다중 클래스 환경 하에서는 단일 정규 분포 ($\mathcal{N}(0,1)$)로 강제 매핑하여 클래스 간 분포 중첩 문제를 겪으며, 최신 연구들은 이를 구조적 고도화를 통해 해결함
  - **HGAD** [Yao et al., ECCV 2024]: 잠재 공간을 계층적 GMM으로 모델링하여 클래스별 분포를 확률적으로 분리, 간섭을 구조적으로 방지함
  - **VQ-Flow** [Zhou et al., arXiv 2024]: 계층적 벡터 양자화(Hierarchical Vector Quantization)를 도입하여 멀티 모달 특성을 이산적 코드북(Discrete Codebook)에 정교하게 매핑함으로써 NF의 효용성을 입증함

**Paradigm to Continual Learning**

- 하지만 HGMNF의 GMM 컴포넌트 수나 VQ-Flow의 코드북 크기는 학습 초기에 고정되어 있어, 사전에 정의된 용량 내에서만 클래스를 수용할 수 있다는 구조적 경직성(Structural Rigidity)을 가짐
- 이러한 정적 구조는 새로운 태스크가 유입될 때 기존 구조와 이질적인 데이터 분포 간의 '구조적 불일치(Structural Mismatch)'를 야기하며, 이를 해소하기 위해 공유 파라미터를 강제로 업데이트할 경우 필연적으로 '파멸적 망각(Catastrophic Forgetting)'이 발생
- 결과적으로 지속적으로 신규 제품군이 추가되는 실제 산업 환경에 대응하기 위해서는 기존 지식을 온전히 보존하면서도 새로운 분포를 유연하게 학습할 수 있는 능력이 필수적이며, 이는 자연스럽게 지속적 학습(Continual Learning) 패러다임으로의 전환을 요구됨

### 2.2. Continual Learning

- 지속적 학습(Continual Learning)은 데이터가 순차적으로 유입되는 비정상(Non-stationary) 환경에서, 새로운 지식을 학습하는 가소성(Plasticity)과 이전 지식을 보존하는 안정성(Stability) 사이의 본질적인 딜레마(Stability-Plasticity Dilemma)를 다룬다 [McCloskey & Cohen, 1989; French, 1999]. 앞 절에서 논의한 Unified AD 모델들의 구조적 경직성, 즉 새로운 클래스를 수용하기 위해 기존 파라미터를 변경하면 이전 지식이 훼손되는 문제는 바로 이 딜레마의 전형적인 발현임

**기존 연구의 한계**

- 초기 연구들은 이 딜레마를 '균형'의 관점에서 접근하여 불완전한 Trade-off를 시도함
- Replay 기반 방법론은 과거 데이터를 저장하여 안정성을 확보하려 했으나, 메모리 제약 및 프라이버시 문제로 인해 새로운 데이터를 수용할 버퍼가 부족해져 가소성이 제한됨
- 이와 달리, Regularization 기반 방법론인 EWC [Kirkpatrick et al., 2017]는 중요 파라미터의 변화를 억제하여 안정성을 높였으나, 이는 모델의 학습 능력을 저해하여 가소성을 희생시키는 결과를 낳음

**Structural Isolation & PEFT**

- 최근에는 파라미터 효율적 미세조정(Parameter-Efficient Fine-Tuning, PEFT)을 통해 딜레마를 타협이 아닌 구조적 분리(Structural Decoupling)로 해결하려는 시도가 주류를 이룸
- 특히 LoRA (Low-Rank Adaptation) [Hu et al., ICLR 2022]는 사전학습된 가중치를 동결한 채 저차원 행렬의 곱으로 태스크별 적응을 수행하여, 기존 지식의 보존과 새로운 지식의 학습을 구조적으로 분리하는 효율적인 방법을 제시
- 이러한 PEFT 기반 접근법은 다양한 방향으로 발전하고 있다:
  - **모듈형 전문가 구조**: GainLoRA [Liang et al., arXiv 2025]와 MINGLE [Qiu et al., NeurIPS 2025]은 태스크별 LoRA 모듈(Plasticity)과 게이팅 네트워크를 결합한 전문가 혼합(Mixture of Experts) 구조를 통해 간섭 없는 지식 확장을 구현함
  - **기하학적 및 인과적 제약**: AnaCP [NeurIPS 2025]와 PAID는 기하학적 구조 보존을 통해 학습 간 간섭을 차단하며, CaLoRA [NeurIPS 2025]는 인과적 추론을 통해 과거 지식을 강화하는 후방 전이(Backward Transfer)까지 달성
  - **동적 부분공간 할당**: CoSO [Cheng et al., NeurIPS 2025]는 학습 중 중요한 부분공간을 동적으로 할당하여 효율적인 가소성을 확보함

**Gap: Decision Boundary vs Density Manifold**

- 이러한 PEFT 기법들은 분류 모델에서는 효과적이었으나, AD 태스크에는 본질적인 한계가 존재
- 분류 모델은 결정 경계(Decision Boundary)만 유지된다면 파라미터의 미세한 변화(Drift)에도 강건. 그러나 NF는 정상 데이터의 확률 밀도(Probability Density) 자체를 정밀하게 모델링해야 하는 생성적 특성을 가짐
- NF의 핵심인 가역적 매핑(Bijective Mapping)은 파라미터 변화에 매우 민감하여, 단순한 Adapter 삽입만으로도 우도 매니폴드(Likelihood Manifold) 전체가 붕괴될 수 있으며, 따라서 NF의 가역성을 보존하면서 파라미터를 격리하는 새로운 방법론이 요구됨

### 2.3. Continual Learning in Anomaly Detection

- 앞서 언급한 일반적인 CL 기법들의 한계를 극복하기 위해, 이상 탐지(AD) 도메인에 특화된 다양한 지속적 학습 방법론들이 제안되었음. 이들은 크게 데이터 보존(Replay), 파라미터 제약(Regularization), 그리고 구조적 확장(Architecture & Prompt) 방식으로 분류됨

**Replay-based Approaches**

- Replay 기반 접근법은 과거 태스크의 데이터를 저장하거나 생성하여 재학습에 활용함
  - CADIC [Yang et al., arXiv 2025]은 대표적인 정상 샘플을 코어셋으로 저장하여 이전 지식을 보존하고자 함
  - ReplayCAD [Hu et al., IJCAI 2025]와 CRD [Li et al., CVPR 2025]는 생성 모델을 활용하여 과거 데이터를 합성함으로써 저장 공간의 제약을 완화하고자 하였음
- 그러나 이러한 접근법들은 근본적인 한계를 가짐. 생성 모델이 만들어내는 샘플의 '충실도 환각'이나 코어셋의 제한된 용량은 정상 데이터의 꼬리 분포(Tail Distribution)를 정밀하게 복원하지 못하며, 이는 밀도 추정 기반 이상 탐지의 정확도를 저하시킴

**Regularization & Distillation Approaches (Weight Constraints)**

- Regularization 및 지식 증류 기반 접근법은 과거 데이터 대신 모델의 중요 파라미터나 출력 분포를 보존하고자 함
  - DNE [Li et al., 2022]는 이전 태스크의 통계량만을 저장하여 메모리 효율성을 높였으며, CFRDC [4]는 문맥 인지 제약(Context-aware Constraint)을 통해 태스크 간 간섭을 완화함
- 그러나 이들 방법은 데이터 분포를 단순한 가우시안으로 가정하거나 파라미터 유연성을 과도하게 제약하는 경향이 있음. 이러한 특성은 복잡한 비선형 변환이 필수적인 NF 구조에 적합하지 않으며, 특히 다양한 텍스처와 구조를 가진 산업 제품의 정상 분포를 정밀하게 모델링하는 데 한계를 보임

**Dynamic Architecture & Prompting (Structural Expansion)**

- 동적 아키텍처 기반 접근법은 태스크마다 별도의 모델 구성요소를 할당하여 간섭을 원천 차단하고자 함
  - SurpriseNet [arXiv 2023]은 완전 격리 방식을 통해 태스크 간 간섭을 완벽히 방지하였으나, 태스크 수에 비례하여 모델 크기가 선형적으로 증가하는 비효율성을 수반함
- 이와 달리 MTRMB [Zhou, You, et al., Arxiv 2025]나 UCAD [Liu et al., AAAI 2024] 등의 프롬프트 기반 방식은 입력 특징에 태스크별 오프셋(Offset)을 추가하여 파라미터 효율성을 확보하였다. 그러나 이는 원본 데이터의 특징 공간(Feature Space)에 인위적인 섭동(Perturbation)을 가하는 것으로, 데이터의 고유한 위상학적 구조(Topological Structure)를 보존해야 하는 NF의 정밀한 밀도 추정을 방해할 수 있음
- 결론적으로, 기존의 Continual AD 연구들은 데이터 프라이버시(Replay 불가), 밀도 추정의 정밀성(Regularization 불가), 파라미터 효율성(Isolation 불가), 세 가지 핵심 요구사항을 동시에 충족하지 못하고 있음
- 특히 NF의 민감한 가역적 구조(Bijective Structure)를 보존하면서도, 지속적으로 유입되는 다중 클래스 분포를 효율적으로 격리 및 확장할 수 있는 새로운 방법론이 절실히 요구됨

---

## 3. Method: DeCoFlow

### 3.1. Problem Formulation

- 연속 이상 탐지 (Continual Anomaly Detection, CAD)은 순차적으로 도착하는 $T$개의 Task Sequence $\mathcal{D} = \{D_0, D_1, \ldots, D_{T-1}\}$를 가정한다
- 각 Task $D_t = \{X_i^{(t)}, y_i^{(t)}\}$는 비지도 학습 설정에 따라 오직 정상 샘플($y_i = 0$)만을 포함하며, 모델은 시점 $t$에서 $D_t$를 학습하며, 이전 데이터 $D_{0:t-1}$에는 접근할 수 없다
- 학습의 최종 목표는 현재 Task $t$를 학습한 모델 $f_{\theta^{(t')}}$가 과거의 모든 task에 대해서도 초기 학습 시점과 유사한 성능을 유지해야 한다. 즉 새로운 지식을 학습함과 동시에 파멸적 망각(Catastrophic forgetting)을 방지해야 한다
- 본 연구는 각 학습 단계마다 새로운 제품 클래스가 입력되고, 추론 시에는 태스크 식별자(Task ID)가 제공되지 않는 클래스 증분 학습(Class-Incremental Learning, CIL) 시나리오를 따른다
- 테스트 시점에는 샘플 $x$만 주어지며 해당 샘플이 어떤 Task에 속하는지(Task-ID)는 제공되지 않는다. 모델이 스스로 적절한 전문가(expert)를 선택해야 한다

### 3.2. Architecture Overview

- DeCoFlow는 섹션 1에서 제기된 분리-효율성 딜레마(Isolation-Efficiency Dilemma)를 해결하기 위해, 기능별로 특화된 모듈들이 유기적으로 결합된 프레임워크를 제안한다
- 본 연구는 거대한 단일 모델을 학습하는 대신, 공유된 지식과 태스크별 적응을 구조적으로 분리하여, 메모리 효율성을 유지 하면서도 파라미터 간섭을 원천 차단하여 완전한 망각 방지를 달성한다

**Key Components & Pipeline Flow**

전체 프레임워크는 데이터 처리 흐름에 따라 다음 네 가지 핵심 단계로 구성된다:

1. **Feature Extractor**: 이미지로부터 다중 스케일 특징을 추출하고 위치 정보(Positional Encoding)를 주입하는 백본 네트워크
2. **Preprocessing Adapters**: 입력 데이터의 분포를 정렬하는 Task-specific Alignment(TSA)와 국소적 문맥을 강화하는 Spatial Context Mixer(SCM)로 구성된 전처리 모듈
3. **Decomposed Normalizing Flow**: 동결된 베이스와 학습 가능한 LoRA 어댑터가 결합되어 밀도 추정을 수행하는 핵심 엔진
4. **Post-processing & Routing**: 비선형 다양체를 보정하는 Affine Coupling Layer(ACL)와 추론 시 적절한 전문가를 선택하는 Prototype Router

전체 데이터 흐름은 다음과 같으며, 각 단계는 공간 정보를 보존하는 텐서 형태를 유지한다.

### 3.3. Design Principle: Invertibility-Independence Decomposition

DeCoFlow의 핵심 통찰은 Normalizing Flow의 커플링 레이어(Coupling Layer)가 파라미터 효율적 연속 학습을 위한 구조적 기반을 제공한다는 점이다. 특히 Affine Coupling Transformation 내의 Scale 및 Shift 네트워크는, 전체 모델의 가역성(Reversibility)이나 야코비안(Jacobian) 계산의 용이성(Tractability)을 저해하지 않으면서도 임의의 함수(Arbitrary Function)로 구현될 수 있다는 유연성을 가진다.

$$
y_1 = x_1, \quad y_2 = x_2 \odot \exp(s(x_1)) + t(x_1), \quad s, t: \mathbb{R}^d \rightarrow \mathbb{R}^d
$$

핵심적으로, 이 유연성은 분해된 서브넷으로 확장된다: $s(x) = s_{base}(x) + \Delta s_t(x)$인 경우, 합계는 여전히 "하나의 함수"이므로 모든 NF 보장을 보존한다. 우리는 태스크별 보정 $\Delta s_t, \Delta t_t$를 저랭크 어댑터 [Hu et al., 2022]로 구현하여, 부분선형 메모리 증가로 완전한 파라미터 격리를 달성한다. 우리는 NF의 임의 함수 속성(Arbitrary Function Property)을 기존의 '표현력의 자유도' 관점이 아닌 '효율적 격리를 위한 구조적 기반'으로 재해석했다.

**The Core Mechanism: Decomposed NF with Frozen Base**

위의 설계 원칙을 바탕으로, 우리는 커플링 레이어의 서브넷을 다음과 같이 '공유된 지식(Shared Knowledge)'과 '태스크별 적응(Task-specific Adaptation)'으로 물리적으로 분해한다:

$$
\text{DecomposedSubnet}(x_{in}) = \text{MLP}_{Base}(x_{in}, \theta_{base}) + \frac{\alpha}{r} B_t(A_t x_{in})
$$

여기서 $\theta_{base}$는 Task 0 학습 이후 영구적으로 동결되어 모든 Task의 Anchor 역할을 수행하며, $A_t, B_t$는 각 Task의 고유한 분포를 학습하는 LoRA Adapter이다. 이 구조는 task 수 $T$에 대해 메모리 증가를 억제하면서도, 기존 지식의 훼손을 완벽하게 방지한다.

**Compensating for Rigidity: The Integral Components**

그러나 베이스 네트워크 동결은 필연적으로 구조적 경직성(Structural Rigidity)를 야기한다. 즉 초기 Task에 고정된 특징 공간은 새로운 Task의 이질적인 분포를 수용하기 어려울 수 있다. DeCoFlow는 이를 해결하기 위해 앞서 언급한 전처리 및 후처리 모듈을 통해 네 가지 기능을 보완한다:

1. **Task-specific Alignment**: 입력 데이터의 통계적 분포를 동결된 베이스의 최적 동작 범위로 강제 정렬하여 입력 인터페이스 불일치를 해소한다
2. **Spatial Context Mixer (SCM)**: NF가 독립적 패치 처리로 인해 놓치는 국소적 상관관계를 보완하여 구조적 사각지대를 해결한다
3. **Task-specific Affine Coupling Layer (ACL)**: 선형 LoRA만으로는 표현하기 힘든 복잡한 분포 차이를 보정하기 위해, 출력단에서 비선형 다양체(Manifold) 적응을 수행한다
4. **Tail-Aware Loss (TAL)**: 쉬운 샘플(Bulk)에 의한 그래디언트 지배를 방지하고, 상위 고손실 패치에 가중치를 부여하여 이상 탐지의 핵심인 결정 경계를 정밀하게 최적화한다

### 3.4. Feature Extraction (Architecture Overview)

**Feature Extractor & Patch Embedding**

- 입력 이미지로부터 유의미한 표현을 추출하기 위해, ImageNet으로 사전 학습된 WideResNet-50을 특징 추출기(Feature Extractor)로 사용한다. 이때, 단일 레이어의 출력만을 사용하는 대신 중간 레이어(intermediate layers)의 출력을 계층적으로 활용하여 다중 스케일 정보(multi-scale information)를 확보한다
- PatchCore 방법론과 동일하게, 추출된 특징 맵은 입력 이미지의 지역적(Local) 특성을 보존하기 위해 패치 단위로 분할된다. 이후 풀링(Pooling) 과정을 거쳐 패치 임베딩을 생성하며, 최종적으로 $F \in \mathbb{R}^{B \times H \times W \times D}$ 형태의 특징 텐서를 얻는다. 여기서 $B$는 배치 크기, $H \times W$는 패치 그리드의 공간 해상도, $D$는 특징 차원을 나타낸다

**Positional Encoding**

- Normalizing Flow는 그 구조적 특성상 입력된 패치 임베딩 각각을 독립적으로 처리하는 순열 불변성(Permutation Invariance)을 가진다. 이로 인해 2차원 이미지의 중요한 정보인 패치 간의 상대적 위치 정보가 소실될 수 있다
- 각 패치의 공간적 위치 정보를 보존하기 위해, 우리는 2D 사인파(Sinusoidal) 위치 인코딩 $P \in \mathbb{R}^{H \times W \times D}$를 도입하여 패치 임베딩에 더한다: $F' = F + P$
- 여기서 $F'$은 위치 정보가 주입된 최종 입력 텐서이며, 이후 Normalizing Flow의 입력으로 사용된다

### 3.5. Integral Components: Compensating for Frozen Base Rigidity

- 동결된 베이스(Frozen Base) 설계는 파멸적 망각을 원천 차단하지만, 필연적으로 모델의 구조적 경직성(Structural Rigidity)을 야기한다. 즉, 초기 태스크(Task 0)에 고정된 모델은 새로운 태스크의 이질적인 분포나 미세한 국소적 변동을 포착하는 데 한계를 보인다
- 이를 해결하기 위해, 우리는 동결된 베이스 패러다임의 특정 한계를 보완하는 두 가지 필수 구성요소인 Task-Specific Alignment (TSA)와 Spatial Context Mixer (SCM)를 제안한다

#### 3.5.1. Task-specific Alignment (TSA)

- 첫 번째 작업(Task 0) 학습 이후 Base Flow의 파라미터를 동결(Freeze)하여 지식을 보존한다. 이는 Base Flow가 Task 0 데이터의 분포 통계량에 피팅되어 있음을 의미하며, 새로운 작업의 입력 데이터 분포가 이와 크게 다를 경우 공변량 변화(Covariate Shift) 문제가 발생한다
- 고정된 Base Flow의 가중치는 새로운 분포에 대해 최적의 활성화를 일으키지 못하며, 제한된 용량(Low-Rank)을 가진 LoRA 모듈이 전역적인 분포 차이까지 보정해야 하는 과도한 부담을 안게 된다. 이는 학습 초기 수렴을 지연시키고 최적화 난이도를 증가시킨다. 이를 해결하기 위해 본 연구에서는 Task-specific Alignment를 도입하여 입력 데이터를 강제로 정렬(Alignment)하고, 고정된 Base Flow가 처리하기 가장 효율적인 형태로 데이터를 재조정(Recalibration)한다
- Task-Specific Alignment (TSA) 모듈은 입력 분포를 동결된 베이스의 최적 동작 범위(Active Region) 내로 강제 정렬하는 역할을 수행하며, 다음 두 단계로 구성된다:

**Task-Agnostic Standardization (표준화)**

학습 파라미터가 없는 LayerNorm을 적용하여, 입력 특징 $F^{(1)}$을 평균 0, 분산 1의 표준 정규 분포로 강제 변환한다:

$$
f_{std} = \frac{F^{(1)} - \mathbb{E}[F^{(1)}]}{\sqrt{\text{Var}[F^{(1)}] + \epsilon}}
$$

이 과정은 새로운 작업의 데이터가 들어왔을 때 기존 데이터와의 스케일 및 분포 격차를 사전에 제거하는 역할을 한다. 이를 통해 Normalizing Flow의 최적화 난이도를 낮추고, 모든 데이터가 동일한 통계적 기준선(Baseline)에서 시작하도록 하여 학습 안정성을 확보한다.

**Task-Adaptive Recalibration (재조정)**

표준화된 특징에 대해 태스크별 학습 가능한 파라미터 $\gamma_t$ (Scale)와 $\beta_t$ (Shift)를 적용하여 Affine Transformation을 수행한다. 이는 단순한 복원이 아니라, 고정된 베이스 모델이 해당 태스크를 처리하기 가장 효율적인 통계적 위치로 데이터를 재조정(Recalibration)하는 과정이다:

$$
\gamma_t = 0.5 + 1.5 \cdot \sigma(\gamma_{raw})
$$

$$
\beta_t = 2.0 \cdot \tanh(\beta_{raw})
$$

$$
\hat{F}_t = \gamma_t \odot x + \beta_t
$$

여기서 $\sigma(\cdot)$는 시그모이드 함수이다. $\gamma_t$는 채널별 특징 중요도(Feature Importance)를 조절하는 역할을 수행한다. $\gamma_t$ (Scale)와 $\beta_t$ (Shift)는 안정적인 학습을 위해 각각 [0.5, 2.0]과 [-2.0, 2.0] 범위 내로 제약(Constraint)된다.

**Why Affine Transformation?**

TSA에서 비선형 변환 대신 아핀 변환을 사용하는 이유는 데이터의 본질적인 기하학적 구조(Shape/Pattern)를 훼손하지 않으면서 1차 및 2차 모멘트(평균, 분산)만을 조절하여 통계적 정합성(Statistical Consistency)을 확보하기 위함이다.

#### 3.5.2. Spatial Context Mixer (Structural Blind Spot Correction)

- Normalizing Flow는 기본적으로 입력을 독립적(i.i.d.)인 패치 단위로 처리하여 우도 $p(X) \approx \prod p(x_{u,v})$를 계산한다. 그러나 이러한 접근 방식은 이미지 데이터에 내재된 강한 공간적 상관관계(Spatial Correlation)를 간과한다는 구조적 한계를 지닌다
- 이러한 독립적 처리는 긁힘(Scratch)이나 얼룩(Stain)과 같이 주변 이웃과의 국소적 불연속성(Local Discontinuity)으로 정의되는 미세 결함을 인지하지 못하는 구조적 한계를 가진다. 또한, 확률론적 관점에서도 정상 데이터의 분포는 주변 정보에 의존적이므로, 이를 정확히 파악하기 위해서는 $p(x_{u,v} | \text{Neighbors})$와 같은 조건부 확률 분포를 모델링할 수 있어야 한다
- 이러한 한계를 극복하기 위해 NF 입력 직전에 Spatial Context Mixer (SCM)를 도입한다. SCM은 Depthwise Convolution을 통해 채널 간 간섭 없이 주변 이웃의 정보를 집계하고, 학습 가능한 게이팅 파라미터 $\alpha$를 통해 원본 정보와 문맥 정보의 혼합 비율을 동적으로 조절한다:

$$
C_{u,v} = \sum_{i,j} W_{i,j} \cdot f_{u+i,v+j}^{(2)}, \quad F_{u,v}^{(3)} = (1-\alpha) \cdot F_{u,v}^{(2)} + \alpha \cdot C_{u,v}
$$

최종 출력 $F^{(3)}$는 NF에 독립적인 개체로 입력되지만, 정보적으로는 이미 이웃의 상태를 내포하고 있다. 결과적으로 NF는 아키텍처 변경 없이도 조건부 확률 $p(X_{u,v} | N_{u,v})$을 간접적으로 학습하는 의사 의존성 모델링(Pseudo-Dependency Modeling) 효과를 얻게 된다.

### 3.6. DeCoFlow: Structural Decomposition of Normalizing Flows

본 프레임워크의 핵심 엔진인 DeCoFlow는 전처리된 특징 텐서 $F^{(3)}$를 입력받아, 이를 잠재 공간(Latent Space)의 가우시안 분포로 매핑하여 정상 데이터의 확률 밀도를 추정한다. RealNVP 기반의 가역적 구조를 따르는 DeCoFlow는 연속 학습 환경에서 파멸적 망각을 방지하고 이상 탐지 성능을 극대화하기 위해, Decomposed Coupling Subnet (DCS)과 Affine Coupling Layer 두 가지 핵심 모듈로 구성된다.

#### 3.6.1. Decomposed Coupling Subnet (DCS): Context-Aware & Task-Adaptive

**설계 의도 (Design Philosophy)**

Decomposed Coupling Subnet (DCS)은 각 커플링 레이어 내부에서 변환 파라미터(scale $s$, shift $t$)를 생성하는 핵심 모듈이다. 일반적인 커플링 레이어가 단순한 MLP를 사용하는 것과 달리, DCS는 공간적 문맥 인지(Spatial Context Awareness)와 태스크 적응성(Task Adaptivity)을 동시에 갖추도록 설계되었다. 특히, 이상 탐지의 핵심인 '국소적 대조(Local Contrast)' 정보를 스케일링 변환에 반영하기 위해 비대칭적인 네트워크 구조를 채택한다.

**구조 및 수식 (Structure & Formulation)**

DCS는 다음 세 가지 단계로 구성된다:

1. **Spatial Context Conditioning**: 입력을 1차원 벡터로 평탄화하는 기존 방식과 달리, 2차원 이미지 형태로 재구성하여 공간 구조를 복원한 후 $N \times N$ Depthwise Convolution으로 지역적 문맥을 추출한다. 추출된 문맥은 학습 가능한 게이팅을 통해 반영 비율이 조절된다:

   $$
   \text{ctx} = \alpha \cdot c(x), \quad \text{where} \quad \alpha = \alpha_{max} \cdot \sigma(\theta_{scale})
   $$
2. **Context-Aware s-network**: 이상(Anomaly)은 주로 주변과의 불연속성(Contrast)으로 나타나므로, 스케일 파라미터 $s$는 원본 특징과 문맥 정보를 결합하여 예측한다:

   $$
   s = \text{Linear}_2^{(s)}(\text{ReLU}(\text{Linear}_1^{(s)}([x; \text{ctx}])))
   $$
3. **Context-Free t-network**: 분포의 위치 이동(Shift) $t$는 패치 고유의 특성에 종속되므로, 문맥 없이 원본 특징만으로 예측한다:

   $$
   t = \text{Linear}_2^{(t)}(\text{ReLU}(\text{Linear}_1^{(t)}([x])))
   $$

**LoRA-based Decomposition**

위 수식의 각 Linear 레이어는 LoRALinear로 구현되어 파라미터 분해를 수행한다. Task 0 학습 후 베이스 가중치($\mathbf{W}_{base}, \mathbf{b}_{base}$)는 동결되어 앵커(Anchor) 역할을 하며, 이후 태스크는 저랭크 어댑터($\mathbf{A}_t, \mathbf{B}_t$)만을 학습한다.

이 구조는 새로운 태스크 학습 시 기존 파라미터를 전혀 수정하지 않으므로 파멸적 망각을 원천적으로 방지한다. 또한, LoRA는 서브넷 내부의 선형 레이어에만 적용되므로 전체 NF의 가역성 및 야코비안 계산에 영향을 주지 않는다.

앞선 섹션(3.5.2)의 SCM과 DCS 내부의 Context Conditioning은 모두 공간 정보를 활용하지만, 그 역할은 명확히 구분된다. SCM은 입력 자체에 이웃 정보를 인코딩하는 반면, DCS는 변환 과정에서 이웃 관계를 고려한 스케일링을 가이드한다. 실험 결과는 두 모듈이 상호 보완적으로 작용하여 단일 모듈 대비 유의미한 시너지 효과를 냄을 보여준다.

#### 3.6.2. Task-Specific Affine Coupling Layer

동결된 베이스와 선형 LoRA의 조합만으로는 해소하기 어려운 태스크 간의 미세한 비선형 매니폴드 불일치를 보정하기 위해, NF 출력단에 Affine Coupling Layer (ACL)를 추가로 배치한다. 이 레이어는 앞서 정의한 표준 커플링 구조와 동일하나, 파라미터 공유 없이 태스크별로 완전히 독립된 가중치를 사용하여 학습한다.

ACL은 잠재 변수 $z_{base}$에 추가적인 가역 변환을 적용하여, 선형 레이어가 포착하지 못한 고차 모멘트(첨도, 왜도 등)를 보정한다. 이를 통해 최종 잠재 분포를 목표인 표준 정규 분포 $\mathcal{N}(0,1)$에 정밀하게 정렬시키며, 최종 우도 계산 시 ACL의 야코비안 행렬식을 추가하여 밀도 추정의 정확도를 극대화한다:

$$
z_{final} = f_{ACL}^{(t)}(z_{base})
$$

### 3.7. Training Objective

DeCoFlow의 학습은 정규화 흐름의 기본 원리인 우도 최대화(Likelihood Maximization)를 기반으로 하되, 이상 탐지의 특성을 고려한 꼬리 인식 손실을 결합하여 최적화를 수행한다.

#### 3.7.1. Likelihood Calculation

모델의 학습과 이상 탐지는 입력 데이터가 잠재 공간의 정규 분포로 얼마나 잘 매핑되는지를 측정하는 로그 우도(Log-Likelihood) 계산을 통해 이루어진다.

입력 $x$는 Task-Specific Alignment (TSA), Spatial Context Mixer (SCM), DCS (Decomposed Coupling Subnet), 그리고 Task-Specific Affine Coupling Layer (ACL)를 순차적으로 통과하며 최종 잠재 벡터 $z_{final}$로 변환된다. 이 과정에서 각 가역 변환 단계의 부피 변화율인 야코비안 행렬식(Jacobian Determinant)의 로그 값이 누적된다:

$$
\log |det J_{total}| = \sum \log |det J_{DCS}| + \sum \log |det J_{ACL}|
$$

Decomposed Coupling Subnet(DCS)과 Task-specific Affine Coupling Layer는 모두 아핀 커플링 구조를 사용하므로 동일한 방식으로 로그 행렬식을 계산한다. 각 블록은 입력을 $x_1, x_2$로 분할하고, 수치 안정성을 위해 constraint가 적용된 스케일 파라미터 $\tilde{s}$를 생성한다:

$$
\tilde{s} = \alpha_{clamp} \cdot \tanh(s / \alpha_{clamp})
$$

야코비안 행렬은 블록 하삼각 형태를 가지므로, 로그 행렬식은 스케일 파라미터의 합으로 간단히 계산된다:

$$
\log |det J_{layer}| = \sum_{i=1}^{D/2} \tilde{s}_i
$$

전체 변환에 대한 로그 행렬식은 $M$개의 DCS 블록과 $N$개의 ACL 블록에서 계산된 값을 모두 합산하여 구한다.

최종 잠재 변수 $z_{final}$이 다변량 표준 정규 분포 $\mathcal{N}(0,1)$를 따른다고 가정하면, 잠재 공간에서의 로그 확률 밀도는 다음과 같다:

$$
\log p(z_{final}) = -\frac{1}{2} \|z_{final}\|_2^2 - \frac{D}{2} \log(2\pi)
$$

변수 변환 공식(Change of Variable Formula)에 따라, 입력 공간 $\mathbf{x}$에서의 최종 로그 우도는 다음과 같이 계산된다:

$$
\log p(x) = \log p(z_{final}) + \log |det J_{total}|
$$

#### 3.7.2. Tail-Aware Loss

일반적인 음의 로그 우도(NLL) 손실 함수를 사용할 경우, 모델은 대다수의 학습하기 쉬운 '정상' 패치(Bulk)에 집중하게 된다. $\nabla_\theta L \propto \nabla_\theta \log \sum p(x_i)$ 식에서 알 수 있듯이, 쉬운 샘플들의 그래디언트 합이 전체 학습 방향을 지배하게 되며, 정작 이상 탐지의 결정 경계(Decision Boundary) 형성에 중요한 분포의 꼬리(Tail) 영역에 위치한 '어려운' 패치들은 과소 최적화된다.

이 문제는 동결된 베이스 환경에서 더욱 악화된다. LoRA의 제한된 용량은 벌크 데이터를 완벽하게 맞추는 것보다, 결정 경계를 정밀하게 조정하는 데 집중되어야 하기 때문이다.

이를 해결하기 위해, 우리는 패치별 음의 로그 우도(NLL)를 기준으로 상위 $K\%$에 해당하는 고손실 패치에 가중치를 부여하는 Tail-Aware Loss (TAL)를 도입한다:

$$
L_{total} = (1 - \lambda_{tail}) \cdot \mathbb{E}[L_{NLL}] + \lambda_{tail} \cdot \mathbb{E}_{top-k}[L_{NLL}]
$$

여기서 $L_{NLL} = -\log p(x_{u,v})$이며, $\lambda_{tail}$은 꼬리 영역의 중요도를 조절하는 하이퍼파라미터이다. 이 손실 함수는 제한된 용량의 어댑터가 이상 탐지에 가장 결정적인 영역을 집중적으로 학습하도록 유도한다.

### 3.8. Task Routing & Adapter Activation

본 프레임워크는 Class-Incremental Learning (CIL) 환경을 가정하므로, 추론 시 입력 이미지가 어느 태스크에 속하는지에 대한 사전 정보(Task ID)가 제공되지 않는다. 따라서 모델은 입력 데이터만을 보고 자동으로 적절한 태스크를 식별하고, 해당 태스크에 특화된 어댑터들(LoRA, TSA, ACL)을 활성화해야 한다. 잘못된 태스크 선택은 부적절한 파라미터 활성화로 이어져 탐지 성능을 크게 저하시키므로, 정확한 라우팅은 시스템의 핵심 과제이다.

#### 3.8.1. Prototype-Based Task Routing

우리는 Mahalanobis 거리 기반의 Prototype Router [Lee et al., 2018]를 사용하여 태스크를 자동으로 인식한다.

각 태스크 $t$의 학습 과정에서, 정상 샘플들의 특징 벡터를 수집하여 태스크별 프로토타입을 구축한다. 프로토타입은 백본의 최종 레이어 출력을 평균한 벡터 $\mu_t \in \mathbb{R}^D$와 공분산 행렬 $\Sigma_t \in \mathbb{R}^{D \times D}$로 구성된다.

여기서 $f^t$는 백본의 최종 출력 벡터(Global Average Pooled Feature)이며, $N_t$는 샘플 수, $\lambda_{reg} = 10^{-5}$는 수치 안정성을 위한 정규화 항이다. 추론 속도를 높이기 위해 정밀도 행렬(Precision Matrix) $\Sigma_t^{-1}$은 미리 계산하여 저장한다.

추론 시, 입력 이미지 $x_{img}$로부터 추출된 특징 벡터와 모든 학습된 태스크 프로토타입 간의 마할라노비스 거리(Mahalanobis Distance)를 계산한다.

가장 거리가 가까운 태스크 $t^*$를 선택하고, 해당 태스크에 종속된 파라미터들을 활성화한다.

**단순 유클리드 거리 대신 마할라노비스 거리를 사용하는 이유:**

- **분포 형태 고려**: 각 태스크 특징 분포의 분산(Spread)과 상관관계를 고려하므로, 분포 간 중첩이 있더라도 더 정확한 구분이 가능하다
- **스케일 불변성**: 정밀도 행렬 $\Sigma^{-1}$이 특징 차원 간의 서로 다른 스케일을 자동으로 정규화한다. 실험적으로, 마할라노비스 거리는 유클리드 거리 대비 평균 3~5% 높은 라우팅 정확도를 달성하였다

추론을 위해 저장해야 할 파라미터는 태스크당 평균 벡터와 정밀도 행렬뿐이다. $T=15$ 태스크, $D=768$ 차원 기준으로 총 메모리 사용량은 약 35MB에 불과하며, 이는 전체 모델 크기에 비해 무시할 수 있는 수준이다.

### 3.9. Anomaly Scoring & Inference

전체 추론 파이프라인은 입력 이미지를 정상 분포로 학습된 잠재 공간으로 변환하고, 그 확률 밀도를 측정하여 이상 여부를 판단하는 과정이다. 핵심 아이디어는 정상 샘플은 높은 확률 밀도를, 이상 샘플은 낮은 확률 밀도를 갖는다는 것이다.

**Step 1: Feature Extraction**

입력 이미지를 사전 학습된 CNN 백본(WideResNet-50)에 통과시켜 다층 특징 맵을 추출한다. 이를 패치 단위로 분할하고 적응적 평균 풀링(Adaptive Average Pooling)을 적용하여 $14 \times 14$ 공간 해상도의 패치 임베딩을 생성한다. 이후 2D 사인파 위치 인코딩을 더하여 위치 정보를 보존한다.

**Step 2: Normalizing Flow Transformation**

선택된 태스크 $t$의 어댑터들이 활성화된 상태에서, 입력 $x$는 다음 순서로 변환된다:

1. **Task-Specific Alignment (TSA)**: 분포를 동결된 베이스의 통계량에 맞춰 정렬
2. **Spatial Context Mixer (SCM)**: Depthwise Convolution으로 인접 패치 간 정보 교환
3. **Decomposed Coupling Subnet (DCS)**: 8개의 LoRA 장착 아핀 커플링 레이어를 통과하여 잠재 변수 $z$로 변환
4. **Task-Specific Affine Coupling Layer (ACL)**: 2개의 블록으로 태스크별 비선형 매니폴드 적응 수행

이 과정에서 각 단계의 로그 행렬식을 누적하여 전체 변환의 부피 변화율 $\log |det J_{total}|$을 계산한다.

**Step 3: Anomaly Score Computation**

최종 잠재 변수 $z_{final} \in \mathbb{R}^{H \times W \times D}$가 표준 정규 분포 $\mathcal{N}(0,1)$를 따른다고 가정하고, 각 패치 위치 $(h, w)$에서의 로그 우도를 계산한다.

Anomaly Score는 음의 로그 우도(Negative Log-Likelihood)로 정의된다:

$$
s_{h,w} = -\log p(x_{h,w})
$$

최종 이미지 레벨의 이상 점수는 모든 패치 점수의 평균, 또는 상위 $K$개 패치 점수의 평균으로 집계하여 산출한다.

---

## 4. Experiments

### 4.1. Experiments Setup

(To be continued...)
