	Related works 
	Unified Multi-class Anomaly Detection 
(Paradigm Shift to Unified Models) 
	초기의 이상 탐지 연구들은 제품 클래스(예: 나사, 병)마다 개별 모델을 학습시키는 One-Class Setting에 집중했으나, 이는 다품종 생산이 이루어지는 실제 현장에서 관리 비용을 급증시킴. 
	이에 따라 UniAD [NeurIPS 2022]와 OmniAL [CVPR 2023]를 기점으로, 단일 모델로 여러 클래스를 동시에 학습하는 Unified (Multi-class) AD 패러다임으로 전환됨. 
	최근에는 MambaAD [NeurIPS 2024] (State Space Model), DiAD [AAAI 2024], LafitE [2023] (Diffusion) 등 다양한 아키텍처가 시도되고 있음.
(Challenges : Interference & Identity Mapping ) 
	그러나 단일 모델이 상이한 데이터 분포를 동시에 학습할 때, 클래스 간 특징이 충돌하는 'Inter-class Interference' [Arxiv 2024] 문제가 발생함. 
	특히 Reconstruction 기반 모델들(DecAD [ICCV 2025], Revitalizing Reconstruction [2024])은 이상(Anomaly)까지 그대로 복원해버리는 Identity Mapping 문제를 해결하기 위해 복잡한 대조 학습이나 Latent Disentanglement를 필요로 함.
(Normalizing Flow)
	이러한 문제에서 비교적 자유로운 것이 **Normalizing Flow (NF)**임. 
	NF는 데이터의 우도(Likelihood)를 직접 최적화하므로, Dinomaly [CVPR 2025]의 "Less is More" 철학처럼 복잡한 보조 태스크 없이도 명확한 이상 탐지가 가능함.
	초기 NF 모델들은 다중 클래스를 단일 정규 분포(N(0,1)로 강제 매핑하여 분포가 뭉개지는 한계가 있었으나, 최신 연구들은 구조적 고도화를 통해 이를 해결함
	HGMNF [ECCV 2024]: 잠재 공간을 계층적 GMM으로 모델링하여 클래스별 분포를 확률적으로 분리, 간섭을 구조적으로 방지함
	VQ-Flow [IEEE TMM 2024]: 계층적 벡터 양자화(Hierarchical Vector Quantization)를 도입하여 멀티 모달 특성을 이산적 코드북(Discrete Codebook)에 정교하게 매핑함으로써 NF의 효용성을 입증함.
	하지만 GMNF(GMM)나 VQ-Flow(Codebook)는 정적 환경을 가정하므로, 초기 태스크 분포에 과적합(Overfitting)된 고정된 잠재 구조를 형성함. 새로운 태스크 유입 시, 고착화된 기존 구조와 새로운 데이터 분포 간에 '구조적 불일치(Structural Mismatch)'가 발생함. 이 불일치를 해소하기 위해 공유 구조를 강제로 업데이트하면, 이전 태스크를 지지하던 핵심 파라미터가 덮어씌워져(Overwritten) 필연적으로 파멸적 망각이 발생함.
	Continual Learning 
	앞서 언급한 Unified AD 모델들의 구조적 경직성(Structural Rigidity)과 재학습의 비효율성을 극복하기 위해, 학계의 관심은 지속적 학습(Continual Learning)으로 이동하고 있음. 이는 데이터가 순차적으로 유입되는 비정상(Non-stationary) 환경에서, 새로운 지식을 학습하는 가소성(Plasticity)과 이전 지식을 보존하는 안정성(Stability) 사이의 본질적인 딜레마(Stability-Plasticity Dilemma)를 다룸.
(기존 연구의 한계)
	초기 연구들은 이 딜레마를 “균형”의 관점에서 접근하여 불완전한 Trade-off를 시도함 
	Replay-based 접근의 경우 과거 데이터를 저장하여 안정성을 확보하려 했으나, 메모리 제약 및 프라이버시 문제로 인해 새로운 데이터를 수용할 버퍼가 부족해져 가소성(Plasticity)이 제한됨
	Regularization-based 접근 [EWC] 등의 경우 중요 파라미터의 변화를 억제하여 안정성을 높였으나, 이는 모델의 학습 능력을 저해하여 Plasticity를 희생시키는 결과를 낳음 
(Structural Isolation & PEFT)
	최근에는 거대 모델의 등장과 함께 파라미터 효율적 미세조정(PEFT)를 통해 딜레마를 타협이 아닌 구조적 분리로 해결하려는 시도가 주류를 이룸 
	Modular Experts 접근 [GainLoRA, MINGLE]은 태스크별 LoRA 모듈(Plasticity 담당)과 게이팅 네트워크를 결합한 전문가 혼합(MoE) 구조를 통해 간섭 없는 지식 확장을 구현함.
	Geometric & Causal Constraints 접근 [AnaCP, PAID]은 기하학적 구조(상대적 각도 등) 보존을 통해 학습 간 간섭을 수학적으로 차단하며, CaLoRA는 인과적 추론을 통해 과거 지식을 강화하는 후방 전이(Backward Transfer)까지 달성함.
	Dynamic Subspace 접근[CoSO]는 학습 중 중요한 부분공간(Subspace)을 동적으로 할당하여 효율적인 가소성을 확보함.
	이러한 PEFT 및 구조적 격리 기법들은 분류나 언어 모델에서는 딜레마를 효과적으로 해결했음. 그러나 이들은 대략적인 결정 경계(Decision Boundary)를 긋는 데 최적화되어 있어, 정상 데이터의 정밀한 확률 밀도(Probability Density) 추정이 필수적인 이상 탐지(Anomaly Detection) 태스크, 특히 민감한 Normalizing Flow 구조에 그대로 적용하기에는 한계가 있음.
(Gap : Decision Boundary vs Dnsity Manifold)
	분류 모델은 판별적(Discriminative) 특성을 가지므로, 클래스 간의 결정 경계(Decision Boundary)만 유지된다면 파라미터의 미세한 변화(Drift)에도 예측 결과가 강건하게 유지됨. 즉, 일반적인 Adapter가 유발하는 특징 공간의 섭동(Perturbation)을 어느 정도 허용함.
	반면, NF는 정상 데이터의 확률 밀도(Probability Density) 자체를 정밀하게 모델링하는 생성적(Generative) 특성을 가짐. NF의 핵심인 가역적 매핑(Bijective Mapping)은 파라미터 변화에 매우 민감하여, 단순한 Adapter 삽입이나 특징 공간의 미세한 왜곡만으로도 우도 매니폴드(Likelihood Manifold) 전체가 붕괴되거나 정상 샘플을 이상으로 오판하는 결과를 초래함. 따라서, 기존의 분류 중심 PEFT 기법을 넘어, NF의 구조적 가역성을 보존하면서도 효과적으로 파라미터를 격리할 수 있는 새로운 방법론이 요구됨.



	Continual Learning in Anomaly Detection 
	앞서 언급한 일반적인 CL 기법들의 한계를 극복하기 위해, 이상 탐지(AD) 도메인에 특화된 다양한 지속적 학습 방법론들이 제안되었음. 이들은 크게 데이터 보존(Replay), 파라미터 제약(Regularization), 그리고 구조적 확장(Architecture & Prompt) 방식으로 분류됨.
(Replay-based Approaches)
	앞서 언급한 일반적인 CL 기법들의 한계를 극복하기 위해, 이상 탐지(AD) 도메인에 특화된 다양한 지속적 학습 방법론들이 제안되었음. 이들은 크게 데이터 보존(Replay), 파라미터 제약(Regularization), 그리고 구조적 확장(Architecture & Prompt) 방식으로 분류됨.
	Experience Replay: CADIC [3]은 통합 메모리 뱅크와 점진적 코어셋(Incremental Coreset) 선택을 통해 핵심 임베딩을 저장하며, ONER [7]은 온라인 경험 리플레이를 통해 스트림 환경에 대응함.
	Generative & Compressed Replay: 메모리 제약을 극복하기 위해 ReplayCAD [8]와 CRD [5]는 확산 모델(Diffusion Model)을 이용하여 과거 데이터를 생성하며, SCALE [20]은 초해상도(SR) 기술을 활용해 이미지를 극도로 압축하여 저장함.
	그러나 이러한 방식은 여전히 데이터 프라이버시(Privacy) 이슈에서 자유롭지 못함. 또한, 생성 모델의 '충실도 환각(Hallucination)' [5]이나 코어셋의 제한된 용량은 정상 데이터의 꼬리 분포(Tail Distribution)를 정밀하게 복원하지 못해, 밀도 추정의 정확도를 저하시키는 원인이 됨.
(Regularization & Distillation Approaches (Weight Constraints)) 
	데이터 저장 없이 파라미터 업데이트를 제어하여 지식을 보존하려는 시도임.
	Statistical Preservation: DNE [21]는 과거 데이터의 통계량(평균, 공분산)만을 저장하여 가우시안 분포를 유지하려 시도함.
	Knowledge Distillation: Reverse Distillation [11]은 교사-학생 구조를 역방향으로 설계하여 특징 복원 능력을 전이하며, CFRDC [4]는 문맥 인지(Context-aware) 제약을 통해 특징 공간의 변화를 억제함.
	이들은 주로 DNE와 같이 데이터 분포를 단순한 가우시안으로 가정하거나, 파라미터의 유연성을 과도하게 제약하여 새로운 태스크에 대한 가소성(Plasticity)을 저해함. 이는 복잡한 비선형 변환이 필요한 NF 구조에 적합하지 않음.
(Dynamic Architecture & Prompting (Structural Expansion)
	최근에는 모델 구조를 동적으로 확장하거나 입력을 변형하여 간섭을 피하는 연구가 활발함.
	Parameter Isolation: SurpriseNet [18]은 리플레이 없이 각 태스크마다 전용 전문가(Expert) 모듈을 할당하여 간섭을 원천 차단하는 'No-Replay' 방식을 제안함. Continual-MEGA [6] 역시 MoE 구조를 도입하여 일반화 성능을 높임.
	Prompt-based Learning: MTRMB [1]와 UCAD [13]은 멀티모달 프롬프트나 대조 학습된 프롬프트를 입력에 추가하여 태스크 정보를 주입함.
	SurpriseNet과 같은 완전 격리(Full Isolation) 방식은 태스크 수에 비례하여 모델 크기가 선형적으로 증가하는 비효율성(Inefficiency)을 가짐.
	프롬프트 방식은 입력 특징을 인위적으로 변형시킴. 이는 PAID [38]가 지적한 기하학적 구조 보존의 중요성을 위배하며, NF가 추정해야 할 정상 데이터의 위상학적 구조(Topological Structure)를 왜곡시켜 정밀한 밀도 추정을 방해함.
	결국, 현재의 Continual AD 연구들은 데이터 프라이버시(Replay 불가), 밀도 추정의 정밀성(Regularization 불가), 파라미터 효율성(Isolation 불가), 그리고 특징 공간의 보존(Prompting 불가)이라는 상충되는 요구사항들을 동시에 만족시키지 못하고 있음. 따라서 NF의 가역적 구조를 보존하면서도 파라미터를 효율적으로 격리할 수 있는 새로운 방법론이 절실히 요구됨.
