## **배경 (Continual AD의 필요성)**

- 딥러닝 기반 이상 탐지(AD)는 정상 데이터만으로 학습하는 One-Class Classification 패러다임을 통해 발전해왔음.
- **산업 현장의 현실:**
    - 제조 현장에서는 신제품 출시, 라인 확장으로 새로운 제품 클래스가 순차적으로 등장
    - 과거 데이터를 모두 축적하는 것은 저장 비용과 프라이버시 규제(GDPR, 의료영상 등)로 인해 불가능
- **파멸적 망각의 위협:** 새로운 클래스 학습 시 이전 지식을 급격히 소실하는 Catastrophic Forgetting 문제에 직면

---

## **기존 방법론의 한계**

### **1) AD에서 망각이 치명적인 이유**

- 분류 모델은 결정 경계(Decision Boundary)만 유지하면 되므로 파라미터 drift에 비교적 강건
- 반면 AD(특히 NF)는 정상 분포의 **확률 밀도(Density)**를 정밀하게 추정해야 함
- 미세한 파라미터 간섭만으로도 우도(Likelihood) 매니폴드가 붕괴 → 치명적 성능 저하
- **결론:** 망각을 '완화(Mitigate)'가 아닌 **'원천 차단(Eliminate)'**해야 함

### **2) 기존 Continual AD 방법론들의 공통 한계**

- 일반적인 지속 학습 기법의 부적합성을 해결하기 위해 이상 탐지에 특화된 다양한 방법론들이 제안됨
- 현재 SOTA 성능을 기록하는 대다수의 연구(CADIC, ReplayCAD 등)는 망각을 방지하기 위해 과거 태스크의 정보를 직접 저장하거나 생성하는 **리플레이(Replay)** 방식에 의존
    - 이 방식은 성능 유지에는 효과적이나, 작업 수에 비례하여 **데이터 메모리 비용(Data Cost)**이 증가
    - 현실적인 메모리/프라이버시 비용뿐만 아니라, 통계적 관점에서도 Replay는 한계를 가짐. 제한된 버퍼 크기(Limited Buffer)에 저장된 소수의 샘플만으로는 고차원 데이터의 복잡한 꼬리 분포(Tail Distribution)를 커버할 수 없으며, 이는 결국 **"편향된 밀도 추정(Biased Density Estimation)"**으로 이어져 이상 탐지의 정밀도를 저하

### 3) Hard Isolation의 한계

- 이러한 데이터 의존성 문제를 해결하기 위해, 최근 연구들은 모델 파라미터를 물리적으로 분리하여 간섭을 원천 차단하는 **'파라미터 격리(Parameter Isolation)'**를 대안으로 주목 (예: SurpriseNet, Continual-MEGA).
    - 그러나 이 접근법은 데이터 비용을 해결하는 대신 필연적으로 **모델 비용(Model Cost)** 문제를 야기
    - 태스크별 독립 네트워크 → **모델 크기 선형 증가 (Linear Growth)**
    - PackNet식 용량 분할 → **용량 포화 (Capacity Saturation)**

### **4) Efficient Isolation의 한계**

- 이러한 모델 비용 문제를 피하기 위해, 최근에는 고정된 모델의 입력단이나 특징 공간(Feature Space)에 프롬프트(Prompt)나 경량 어댑터를 적용하는 **'효율적 격리(Efficient Isolation)'** 방식을 시도
    - 기존의 연구는 Contrastive learning으로 학습 된 Prompt를 저장하거나 Task별 어댑터느 Instance-aware prompt를 통해 고정된 백본의 특징을 새로운 Task에 맞게 조정하는 방식을 채택하고 있음
    - 그러나 이러한 접근법들은 표현력의 한계 또는 추가적인 비용 문제가 여전히 존재
        - 사전학습된 Backbone 표현력에 크게 의존 → **표현력 병목**
        - Prompt는 특징 공간 내 이동(shift)에는 효과적이나, 이질적인 기하학적 구조 표현에 한계
        - 여전히 **메모리 뱅크, 복잡한 라우팅 모듈** 필요 → 완전한 효율성 미달성

---

## **문제 정의 : Isolation-Efficiency Dilemma**

결국 현재의 지속적 이상 탐지 연구들은 이상적인 해결책을 찾지 못한 채, 다음과 같은 **삼중고(Trilemma)** 형태의 **'격리-효율성 딜레마(Isolation-Efficiency Dilemma)'**에 직면

1. **성능(Performance):** 밀도 추정의 정밀도를 위해 **완전 격리(Hard Isolation)**를 택하면 **모델 비용**이 폭발하거나 용량이 포화됩니다.
2. **효율성(Efficiency):** 모델 비용을 줄이려 **효율적 격리(Prompt/Adapter)**를 택하면 **백본 의존성**으로 인해 정밀도가 떨어지거나, 여전히 **보조 메모리/라우팅 비용**이 발생합니다.
3. **확장성(Scalability):** 이를 피해 **리플레이(Replay)**를 택하면 **데이터 메모리 비용** 증가와 **통계적 편향** 문제가 발생합니다.

본 연구는 이러한 딜레마를 타파하기 위해, 데이터 저장 없이(No Replay) 그리고 모델의 폭발적 증가 없이(Parameter Efficient) NF의 구조적 특성을 활용한 새로운 해법을 제안

---

## **Motivation : NF의 구조적 특성이 딜레마를 해결**

- Requirement :
    - **Isolation의 요구사항**: 태스크 간 파라미터 간섭이 없어야 함
    - **Efficiency의 요구사항**: 태스크마다 전체 모델을 복사하면 안 됨
- **핵심 아이디어:** "Full copy가 필요한 게 아니다. **변화가 필요한 부분만 분리**하면 된다."
    - 파라미터를 `W_task = W_shared + ΔW_task`로 분해 (공유: frozen, 태스크별: isolated, small)
- **핵심 질문:** 어떤 모델에서 이러한 분해가 안전한가?
    - **Embedding/Memory Bank (PatchCore 등):** 학습 파라미터 없음, Memory = Replay 문제
    - **Reconstruction (AE, VAE):** Encoder-Decoder 결합 → 분해 시 잠재공간 정합성 붕괴
    - **Teacher-Student (STPM 등):** T-S 정합성 필요 → 분해 시 불안정
    - **Density - NF Coupling:** Subnet 구현과 무관하게 가역성 보장 → **분해 시에도 이론적 보장 유지**

### Design Principle : Invertibility-Independence Decomposition

- NF의 Coupling Layer는 "Arbitrary Function Property"를 가짐 [RealNVP)
    - Subnet이 어떤 함수이든 가역성과 tractable likelihood가 보장된다. 이 특성 덕분에 Subnet을 [Base + LoRA]로 자유롭게 분해할 수 있음
    - 아핀 결합 레이어에서 가역성 보장은 결합 구조에만 의존하며, 서브넷의 내부 매개변수화 방식에는 의존하지 않습니다. 이러한 구조적 특성은 다음을 가능하게 함
        - (i) 매개변수 격리: 작업별 매개변수(LoRA)를 다른 작업에 영향을 주지 않고 서브넷에 국한시킬 수 있음
        - (ii) 유효성 보존: 유효한 서브넷 분해에 대해 흐름 bijectivity와 계산 가능한 likelihood function이 유지됨
        - (iii) 효율적 확장성: 메모리 사용량이 O(N×매개변수)가 아닌 O(N×Rank)로 증가함
    - 

### **Formal Statement :  파라미터 분해를 적용한 아핀 커플링 (Formal Statement: Affine Coupling with Parameter Decomposition)**

- **아핀 커플링 레이어(Affine Coupling Layer)의 수학적 구조:**
    - **순방향 (Forward):** $y_1 = x_1, \quad y_2 = x_2 \odot \exp(s(x_1)) + t(x_1)$
    **역방향 (Inverse):** $x_1 = y_1, \quad x_2 = (y_2 - t(y_1)) \odot \exp(-s(y_1))$
    - Proposition 1 (임의 함수 속성, Arbitrary Function Property):
        - $s, t : \mathbb{R}^{d/2} \to \mathbb{R}^{d/2}$를 임의의 measurable function라 하자. 아핀 커플링 변환 $T(x) = [x_1; x_2 \odot \exp(s(x_1)) + t(x_1)]$는 $s$와 $t$가 내부적으로 어떻게 파라미터화되었는지와 무관하게 전단사(bijective)이며, 다음과 같은 계산 가능한 야코비안 행렬식(tractable Jacobian determinant)을 갖는다.
        
        $$
        \log|\det \nabla T| = \sum_i s_i(x_1)
        $$
        
        - Proof Sketch:
            - 야코비안 행렬은 다음과 같은 블록 삼각 구조(block triangular structure)를 갖는다.
            
            $$
            \frac{\partial T}{\partial x} = \begin{bmatrix} I & 0 \\ \frac{\partial T_2}{\partial x_1} & \text{diag}(\exp(s(x_1))) \end{bmatrix}
            $$
            
            - 따라서 행렬식은 $\prod_i \exp(s_i(x_1))$와 같으며, 이는 $s$의 내부 파라미터화 방식이 아닌 오직 $s$의 출력값에만 의존한다.
- **Structural Connection:**
    - NF의 **Arbitrary Function Property**는 잘 알려진 사실 (RealNVP, 2017)
    - 우리는 이 특성과 **CL의 파라미터 분해 요구사항** 사이의 구조적 연결을 확립
    - 기존 인식: "Subnet이 어떤 함수든 가능" = 표현력의 자유도
    - **우리의 재해석:** "Subnet이 어떤 함수든 가능" = **효율적 격리를 가능케 하는 구조적 기반**
    - → Subnet을 `Base + LoRA`로 분해해도 가역성과 tractable likelihood 유지
    - → **Isolation-Efficiency Dilemma의 구조적 해결**
    - **Feature-level Adapter와의 핵심 차이:**
        - Feature-level: adapter가 density manifold의 input을 교란 → 밀도 추정 왜곡
        - **NF Coupling-level: adapter가 subnet 내부의 transformation만 변경** → Arbitrary Function Property로 가역성 보장

---

## **MoLE-Flow 설계 : Base + LoRA Decomposition**

- **핵심 설계:** Coupling Subnet = Frozen Base + Task-specific LoRA
    - Task 0: Base + LoRA_0 함께 학습 → 이후 Base 동결
    - Task 1+: LoRA_t만 학습 (Base 재사용)
    - **결과:** Task당 8% 파라미터로 완전한 격리 → 딜레마 해결
    - **왜 Low-rank로 충분한가:**
        - "정상이란 무엇인가"의 개념은 task 간 공유 (e.g., texture uniformity, structural integrity)
        - domain-specific appearance만 다름 (e.g., leather vs. transistor의 시각적 특성)
        - → low-rank adjustment로 충분 (Sec. 4.4 rank sensitivity 실험으로 검증)
- **Base Freeze의 부작용과 Integral Components:**
    - Base freeze는 Zero Forgetting을 보장하지만 세 가지 structural side-effect 야기
    - 이를 보상하기 위한 세 가지 구성요소 설계:
        - **Whitening Adapter (WA):** Task 1+ 학습 시 feature distribution shift 보상 - frozen base가 Task 0 분포에 최적화되어 발생하는 입력 분포 불일치 해결
        - **Tail-Aware Loss (TAL):** LoRA의 제한된 capacity로 인한 tail region 정밀도 저하 문제 해결 - anomaly detection에 critical한 low-density 영역 학습 강화
        - **Dense Intermediate Attention (DIA):** LoRA의 low-rank constraint로 인해 부족해진 local transformation capacity 보충 - fine-grained spatial pattern 모델링 복원
    - **핵심:** Interaction Effect Analysis로 이들이 "generic boosters"가 아닌, **Base Freeze 조건에서만 필수적인 integral components**임을 검증
- **Task-Agnostic Inference:**
    - **Challenge:** CL 환경에서 inference 시 task ID가 주어지지 않음 - 어떤 expert를 사용할지 자동 결정 필요
    - **Solution:** Prototype 기반 라우팅 - 각 task의 feature distribution을 prototype으로 저장, Mahalanobis distance로 가장 가까운 task 선택
    - **결과:** 100% 라우팅 정확도 (MVTec AD) - AD task 간 feature space가 well-separated

---

## **Contributions**

1. **Structural Connection:** NF coupling layer의 Arbitrary Function Property와 CL의 파라미터 분해 요구사항 사이의 구조적 연결을 확립. 이 연결을 통해 zero-forgetting CL을 O(rank) 효율로 달성 가능.
2. **Parameter-Efficient Isolation Framework:** MoLE-Flow 제안 - coupling subnet을 frozen shared base와 task-specific LoRA로 분해하여, task당 8% 파라미터로 완전한 격리 달성.
3. **Integral Components with Rigorous Validation:** Base Freeze의 부작용을 보상하는 세 가지 구성요소를 제안하고, Interaction Effect Analysis를 통해 구조적 필수요소임을 입증.
4. **State-of-the-Art Results:** MVTec-AD (15 classes, 5 runs)에서 98.05+/-0.12% I-AUC, 55.80+/-0.35% P-AP를 Zero Forgetting으로 달성.