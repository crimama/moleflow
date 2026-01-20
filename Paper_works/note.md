논문 서론 서술에서, 기존 continual learning과 continual ad와의 task 관점에서 고유 차이점 어느정도 서술 필요 

- continual learning : 기존에 성능은 어느정도였는데, forgetting으로 인해 성능이 저하되는 것을 완화 시킨다. 
- continual ad : 기존 방법론과는 독자적인 방법론을 대체로 구축후 continual learning 시나리오에서 강건한 성능을 보여주도록 함 

이럴 때 parameter isolation을 조금 더 강조해서 풀어나가면 어떨지 


tail aware loss에서 ratio 0.02에 대한 합당한 근거 및 실험 필요 
- 단순히 0.02 비율의 likelihood들에 대해 weight를 주었을 때 이렇게 까지 성능 차이가 나는지에 대한 분석 필요 
- 반대로 0.05 로 커지는 경우 반대로 성능이 크게 저하 : 이에 대한 가설로, tail의 너무 많은 부분을 weight를 주게 되면, 이 tail들이 gaussian distribution의 head가 되고, 정작 가장 많은 확률 및 분포를 차지하는 특징들이 tail로 이동 됨 -> 이 현상이 반복되면서, 정상적으로 gaussian distribution을 학습하지 못 하고 뒤틀어짐 -> 이걸 보여주는 실험 필요 



# 왜 scale network에 condition을 넣는가 

MoLE-Flow의 Affine Coupling Layer 설계에서 **"-network에는 Context를 주고, -network에는 주지 않는 이유"**를 논리적으로 완벽하게 정리해 드리겠습니다.

이 설계는 **수학적 중요도**, **통계적 본질**, **학습 안정성**이라는 세 가지 관점에서 필연적인 선택입니다.

---

### **1. 수학적 관점: s 는 '점수(Likelihood)'를 지배한다**

이상 탐지에서 우리가 구하려는 최종 점수(Log Likelihood) 식은 다음과 같습니다.

* **$t$의 영향력 (제한적):**
  * 오직 **$\log p(z)$**에만 관여합니다.
  * 즉, 데이터를 0 근처로 옮겨놓는 역할만 합니다. 이는 상대적으로 단순한 작업입니다.


* **$s$의 영향력 (절대적):**
  * **$\log p(z)$**와 **$\log |\det J|$** **두 항 모두에 결정적인 영향**을 미칩니다.
  * 특히 $\log |\det J|$ (Jacobian Determinant)는 **오직 $s$만이 결정**할 수 있습니다.
  * 이 항은 "데이터의 밀도"를 정의하는 값이므로, $s$가 조금만 틀려도 이미지가 정상인지 비정상인지 판별하는 점수 체계 자체가 무너집니다.



** 결론:** 점수 계산에 더 큰 권한과 책임을 가진 **에게 더 많은 정보(Context)를 제공하여 정확도를 높여야 합니다.**

---

### **2. 통계적 관점: '분산($s$)'은 혼자서 알 수 없다**

통계적으로 $t$는 **평균(Mean/Shift)**을, $s$는 **분산(Variance/Scale)**을 추정하는 것입니다.

* **$t$ (평균)의 특성:**
  * "이 픽셀의 값이 대략 얼마여야 하는가?"는 입력된 패치($x_1$) 자체만 봐도 충분히 예측 가능합니다. 
  * 굳이 주변을 안 봐도 내 값이 어두우면 평균도 낮은 것입니다.

* **$s$ (분산)의 특성:**
  * "이 픽셀 값이 **튀는 것이 정상인가?**"를 판단해야 합니다.
  * 예를 들어, **가죽(Leather)** 데이터에서 값이 울퉁불퉁하다면, 이것이 '가죽 원래의 주름(정상, 분산 큼)'인지 '긁힌 자국(비정상)'인지 구분해야 합니다.
  * 이 구분은 **픽셀 하나($x_1$)만 봐서는 절대 불가능**하며, 반드시 **주변 패턴(Context)**을 봐야만 "아, 여기는 원래 주름이 많은 영역이구나($s$ 높임)"라고 판단할 수 있습니다.

**$\therefore$ 결론:** 분산($s$) 추정은 본질적으로 **'문맥 의존적(Context-dependent)'**인 작업이므로 Context 입력이 필수입니다.

---

### **3. 엔지니어링 관점: 민감한 녀석($s$) 진정시키기**

변환 수식은 $y = x \cdot \exp(s) + t$ 입니다.

* **$t$ (선형적, 둔감함):**
  * 값이 조금 틀려도 결과가 조금 옆으로 이동할 뿐입니다. 에러가 선형적으로 전파됩니다.
  * 따라서 굳이 Context를 주어 연산량을 늘리거나 과적합(Overfitting) 위험을 감수할 필요가 없습니다.


* **$s$ (지수적, 민감함):**
  * 값이 **지수 함수($\exp$)**를 통과합니다. $s$가 약간만 커져도 출력값은 폭발하고, 작아지면 소멸합니다.
  * 정보가 부족해서 $s$가 "찍어서" 맞추려 하면 모델 전체가 불안정해집니다.
  * Context를 제공하는 것은 **$s$에게 "확실한 근거"를 쥐어주어 학습을 안정화**시키는 안전장치 역할을 합니다.



**$\therefore$ 결론:** 폭주하기 쉬운 **$s$를 제어하기 위해 Context라는 가이드라인**이 필요합니다.

---

### **최종 요약**

| 구분 | -network (Translation) | -network (Scale) |
| --- | --- | --- |
| **역할** | 데이터 **이동** (위치 잡기) | 데이터 **밀도 조절** (부피 잡기) |
| **중요도** | $\log p(z)$에만 관여 | **$\log |
| **필요 정보** | **자신의 값()이면 충분** | **주변 상황(Context)을 봐야만 판단 가능** |
| **성격** | 틀려도 리스크가 적음 | **틀리면 점수가 폭발함 (Sensitive)** |
| **설계 결정** | **Context 입력 안 함**<br>

<br>(효율성, 과적합 방지) | **Context 입력 필수**<br>

<br>(정확성, 안정성 확보) |

이 설계는 **"가장 어렵고 중요하며 민감한 작업()에 모든 자원(Context)을 집중하고, 단순한 작업()은 경량화한다"**는 MoLE-Flow의 핵심 철학을 보여줍니다.

# DIA 
DIA (Deep Invertible Adapter)에서 **"2개의 Block을 사용하는 것이 비선형 매니폴드 보정의 최소이자 충분 조건임"**을 객관적이고 학술적인 근거(Mathematical & Structural Justification)를 들어 설명해 드리겠습니다.

논문이나 기술 보고서에 작성하실 때는 다음과 같은 3가지 논리를 중심으로 서술하시면 됩니다.

---

### **1. 구조적 근거: 전역적 상호 의존성 확보 (Global Inter-dependency)**

Affine Coupling Layer의 가장 큰 수학적 약점은 **"부분적 항등 함수(Partial Identity Mapping)"**라는 점입니다.

* **1개 Block ()의 한계:**
* 수식: 
* 입력 차원의 절반()은 아무런 변환 없이 그대로 출력으로 복사됩니다.
* 즉, 이 단계에서 모델의 **Jacobian Matrix(자코비안 행렬)**는 절반이 0이거나 1인 상태(Triangular Matrix)로, **모든 차원 간의 상관관계(Full Covariance)**를 학습하지 못합니다.


* **2개 Block ()의 필연성:**
* 두 번째 블록에서 역할을 뒤집음(Reverse/Permute)으로써, 앞서 방치되었던 $x_{1:d}$가 변환의 대상이 됩니다.
* 수학적으로 두 함수의 합성 이 일어날 때, 비로소 입력 의 **모든 차원이 서로가 서로에게 영향을 미치는(Fully Coupled)** 구조가 완성됩니다.
* 따라서, 는 데이터의 **모든 차원을 최소 1회 이상 변환하기 위한 수학적 최소 단위(Atomic Unit)**입니다.



### **2. 기하학적 근거: 비선형  diffeomorphism 근사 (Approximation)**

Normalizing Flow 이론에 따르면, 충분한 수의 Coupling Layer를 쌓으면 어떠한 복잡한 분포 변환(Diffeomorphism)도 근사할 수 있습니다 (Universal Approximation).

* **Local Diffeomorphism Correction:**
* Base NF 모델이 이미 $P_{data}$를 $P_{base} \approx N(0, I)$로 대략적으로 매핑해 둔 상태입니다.
* 하지만 특정 Task(예: 가죽, 나사)의 데이터 분포 $P_{task}$는 $P_{base}$와 미세하게 다른 **지역적 비선형성(Local Non-linearity)**을 가집니다.
* 이 '잔여 오차(Residual Gap)'를 메우는 데에는 수십 층의 깊은 네트워크가 필요하지 않습니다.
* 단 **2개의 Block**으로 구성된 은 ** 영향력**과 ** 영향력**을 모두 포함하므로, 좁은 범위의 비선형 굴곡을 펴는 데 충분한 표현력(Expressivity)을 가집니다.



### **3. 최적화 관점: 파라미터 효율성과 과적합 방지 (Regularization)**

DIA는 Task-Specific한 모듈이므로, 데이터가 적은(Few-shot) 상황에서도 학습이 잘 되어야 합니다.

* **Bias-Variance Trade-off:**
* Block을 많이 쌓으면(Depth ) 표현력은 좋아지지만, 파라미터 수가 늘어나 **과적합(Overfitting)** 위험이 커지고 최적화가 어려워집니다.
* 이미 Base Model이 강력한 특징 추출 능력을 갖고 있으므로, Adapter는 **"최소한의 자유도"**만 가지는 것이 일반화 성능(Generalization)에 유리합니다.


* **Minimal Complete Interaction:**
* 2개의 Block은 모든 차원을 건드리는 '완전성(Completeness)'을 갖추면서도, 가장 적은 파라미터를 사용하는 '효율성(Efficiency)'의 접점(Sweet Spot)입니다.



---

### **[서술 예시]**

보고서나 논문에 사용하기 좋은 문구로 정리해 드립니다.

> "DIA는 Base Flow가 완전히 제거하지 못한 Task-specific한 비선형 분포 차이(Distributional Shift)를 보정하기 위해 설계되었다. 여기서 **2개의 Affine Coupling Block**을 사용하는 것은 경험적 선택이 아닌 구조적 필연성이다.
> 단일 Coupling Layer는 입력 차원의 절반을 항등 매핑(Identity Mapping)하므로 전체 차원 간의 상호 의존성(Full Dependency)을 모델링할 수 없다. 이를 해소하기 위해 채널 순서를 반전(Reverse)시킨 두 번째 블록을 결합함으로써, **모든 입력 차원이 최소 1회 이상 변환(Transformation)**되도록 보장한다.
> 결과적으로  설정은 전체 특징 공간(Feature Space)에 대한 **완전한 수용장(Full Receptive Field)을 확보하는 최소한의 구조적 단위**이며, 동시에 적은 파라미터로 과적합을 방지하며 효율적인 미세 조정(Fine-tuning)을 가능하게 한다."


이유: LoRA는 도구일 뿐입니다. 논문의 진짜 기여는 NF를 분해해도 된다는 **'구조적 발견(Structural Decomposition)'**에 있습니다. 이 제목이 더 "Theoretical Contribution"이 있어 보입니다. (LoRA는 Abstract에서 언급하면 됩니다.)