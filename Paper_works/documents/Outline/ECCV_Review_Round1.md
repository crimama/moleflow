# ECCV 2026 Paper Review

**Paper Title**: DeCoFlow: Structural Decomposition of Normalizing Flows for Continual Anomaly Detection

**Authors**: Hun Im, Jungi Lee, Subeen Cha, Pilsung Kang

**Reviewer**: Area Chair / Senior Reviewer

**Review Date**: 2026-01-21

---

## 1. Summary (2-3 sentences)

This paper addresses catastrophic forgetting in continual anomaly detection by proposing DeCoFlow, which leverages the Arbitrary Function Property (AFP) of Normalizing Flows to enable safe parameter decomposition. The key insight is that coupling layer subnets can be decomposed into frozen base weights plus task-specific LoRA adapters without compromising NF's invertibility or likelihood tractability. The method achieves 98.05% I-AUC on MVTec-AD with mathematically guaranteed zero forgetting (FM=0.0, BWT=0.0) and 100% routing accuracy, using only 8% additional parameters per task.

---

## 2. Strengths

### S1. Novel and Well-Motivated Theoretical Foundation
The reinterpretation of AFP as a "structural foundation for efficient isolation" rather than mere "expressiveness freedom" is genuinely novel. This provides a principled answer to why NF-based approaches are uniquely suited for parameter decomposition in continual learning, which other architectures (VAE, AE, Teacher-Student) cannot achieve. The architecture comparison experiment (Table 2) validates this claim convincingly with Teacher-Student showing +24.08% forgetting while NF maintains FM=0.0%.

### S2. Mathematically Guaranteed Zero Forgetting
Unlike most continual learning methods that merely "mitigate" forgetting, DeCoFlow provides mathematical guarantees: base parameters are frozen after Task 0, and task-specific adapters are completely isolated. The BWT verification (Table 5) empirically confirms this with p-values of 1.000 across all 15 tasks, demonstrating perfect retention.

### S3. Comprehensive Experimental Design
The experiments are well-structured with:
- Standard benchmarks (MVTec-AD, ViSA)
- Multiple evaluation metrics (I-AUC, P-AUC, P-AP, FM, BWT, Routing Accuracy)
- Extensive ablation studies (component, rank, architecture depth)
- Statistical rigor (2x2 factorial ANOVA with Bonferroni correction for interaction effects)
- Clear distinction between "integral" and "general" components

### S4. Strong Empirical Results
DeCoFlow outperforms all baselines on MVTec-AD:
- +3.35%p I-AUC over CADIC (98.05% vs 94.7%)
- +1.85%p I-AUC over ReplayCAD while requiring NO replay
- 100% routing accuracy
- 3x memory reduction and 6x faster inference vs replay methods

### S5. Honest Discussion of Limitations
The paper explicitly acknowledges the P-AP trade-off (55.8% vs CADIC's 58.4%, -2.6%p) and frames it as a principled design choice. The failure case analysis (routing confusion for similar tasks, extreme distribution shift, small anomalies) demonstrates scientific honesty.

---

## 3. Weaknesses

### W1. Incomplete Experiment: Coupling-level vs Feature-level Adaptation (Critical)
Section 4.4.2 describes an experiment comparing coupling-level LoRA (DeCoFlow) vs feature-level adapters (L2P/DualPrompt style), which is critical for validating the AFP claim (C1). However, this experiment is marked as "TODO" with only "Expected Outcome" provided. This is a significant gap because:
- The claim that "feature-level adapters disrupt the density manifold" is central to the paper's novelty
- Without this comparison, the advantage of coupling-level adaptation over simpler feature-level approaches remains unproven
- PEFT-based CL methods like L2P and DualPrompt are mentioned in baselines but not actually compared in Table 3

**Impact**: Weakens the core contribution (C1) substantially.

### W2. SVD Analysis Interpretation Requires Revision
The SVD analysis (Section 4.4.3) reveals that effective rank for 95% energy is ~504, and rank-64 captures only ~28.5% of total energy. The paper claims this supports "selective learning of critical singular directions." However:
- This contradicts the initial "low-rank adaptation hypothesis" framing
- The argument shift from "intrinsically low-rank" to "LoRA selects important directions" is post-hoc rationalization
- No evidence is provided that the top 64 singular directions are indeed "most critical for anomaly detection"
- The rank ablation showing <0.1%p variance (Table 8) could equally support that LoRA acts as a regularizer rather than capturing critical directions

**Suggestion**: Either provide evidence that top singular directions correlate with detection performance, or reframe the contribution as "implicit regularization through low-rank constraint."

### W3. Missing Comparisons with PEFT-based Continual Learning Methods
The baseline comparison lacks modern PEFT-based continual learning methods:
- L2P (Learning to Prompt) - mentioned in Related Works but not compared
- DualPrompt - mentioned but not compared
- LAE (Learnable Adapter Expert) - mentioned in 4.1.4 but absent from results
- S-Prompts, CODA-Prompt, etc.

While these methods are designed for classification, adapting them to AD would provide a fairer comparison for the "parameter efficiency" claim.

### W4. ViSA Results Show Performance Gap
On ViSA dataset (Table 4):
- DeCoFlow achieves 90.0% I-AUC, competitive with ReplayCAD (90.3%)
- But P-AP drops significantly: 26.6% vs ReplayCAD's 41.5% (-14.9%p)
- Routing accuracy drops to 95.8% (from 100% on MVTec)

This substantial P-AP gap on ViSA is not adequately explained. The paper mentions "PCB1-PCB4 confusion" but doesn't address why localization suffers more severely on this dataset.

### W5. Computational Overhead Higher Than Claimed
The paper claims "only 8% additional parameters per task" in the abstract, but Table 3 shows "22-42%" per task. The full configuration (with DIA) requires 41.7% of NF base per task. While still more efficient than full model replication, the abstract claim is misleading.

---

## 4. Questions for Authors

### Q1. [Critical] Coupling vs Feature-level Experiment
Can you provide preliminary results or at least a pilot study for the coupling-level vs feature-level adaptation comparison (Section 4.4.2)? Even results on a subset of classes would substantially strengthen the AFP claim.

### Q2. SVD Analysis Interpretation
Given that rank-64 captures only 28.5% of total energy but achieves near-optimal performance, how do you explain this? Have you verified that the top 64 singular directions indeed capture task-discriminative information rather than noise?

### Q3. ViSA P-AP Gap
The P-AP gap on ViSA (26.6% vs 41.5%) is much larger than on MVTec. Is this due to:
(a) Different anomaly characteristics (finer-grained defects)?
(b) Routing errors cascading to localization errors?
(c) Frozen base being optimized for MVTec-like distributions?

### Q4. Task 0 Sensitivity
Section 4.3.5 mentions Task 0 selection sensitivity analysis as TODO. Given that the frozen base is determined by Task 0, this seems important. Do you have preliminary findings on whether a "bad" Task 0 choice can lead to systematic failures?

---

## 5. Scoring (1-10 scale)

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | 7.5 | AFP reinterpretation is novel; coupling-level LoRA for NF is new. However, individual components (LoRA, prototype routing, tail-aware loss) are not novel. |
| **Technical Quality** | 7.0 | Solid theoretical foundation and methodology. Weakened by incomplete coupling vs feature-level experiment and post-hoc SVD interpretation. |
| **Experimental Validation** | 7.5 | Comprehensive ablations, statistical rigor (ANOVA), multiple datasets. Missing PEFT-CL baselines and one critical experiment. |
| **Presentation** | 8.0 | Well-written, clear structure, honest about limitations. Tables and figures are informative. Some notation inconsistencies (MoLE-Flow vs DeCoFlow). |
| **Significance** | 7.5 | Important problem (continual AD), practical solution with real deployment potential. Zero forgetting guarantee is valuable for manufacturing. |
| **Overall Score** | **7.3** | Good paper with novel theoretical contribution and strong empirical results, but critical experiment incomplete. |

---

## 6. Decision

**Weak Accept** (Borderline Positive)

### Justification

This paper presents a well-motivated approach to continual anomaly detection with a novel theoretical contribution (AFP reinterpretation) and strong empirical results (98.05% I-AUC, zero forgetting). The mathematical guarantee of zero forgetting is particularly valuable for industrial applications where reliability is paramount.

However, the paper has two significant issues that prevent a stronger recommendation:

1. **Critical Missing Experiment**: The coupling-level vs feature-level adaptation comparison (Section 4.4.2) is central to validating the core AFP claim but remains unexecuted. This leaves a gap in the paper's main contribution.

2. **Interpretation Concerns**: The SVD analysis interpretation shifted from "intrinsically low-rank" to "selective learning of critical directions" without sufficient evidence, weakening Claim C2.

If the authors can provide results for the coupling vs feature-level experiment and strengthen the SVD interpretation, this paper would merit a clear Accept.

---

## 7. Specific Suggestions for Improvement

### To Raise Score to Accept (7.5-8.0):

1. **Complete Section 4.4.2 Experiment**
   - Implement feature-level adapter baseline (add learnable prompts/adapters before NF input)
   - Compare I-AUC, P-AP, FM, and likelihood consistency
   - Even 6-class pilot results would suffice

2. **Strengthen SVD Analysis**
   - Option A: Show correlation between top singular directions and anomaly detection performance
   - Option B: Reframe as "implicit regularization" and cite relevant literature on over-parameterization

3. **Add PEFT-CL Baselines**
   - Adapt L2P or DualPrompt for NF-based AD
   - Even if they perform poorly, showing why validates your approach

4. **Clarify Parameter Overhead**
   - Update abstract to reflect actual overhead (22-42% not 8%)
   - Or report DIA-free configuration (21.8%) as primary

### To Raise Score to Strong Accept (8.5+):

5. **ViSA Analysis**
   - Diagnose root cause of P-AP gap (routing vs intrinsic limitation)
   - Propose or implement mitigation

6. **Complete TODO Experiments**
   - Task 0 sensitivity (4.3.5)
   - Long-sequence scalability (4.3.6)
   - Backbone sensitivity (4.3.7)

7. **Theoretical Analysis**
   - Formal bound on LoRA's approximation error for coupling subnets
   - Connection to neural network compression theory

---

## 8. Minor Issues

### Presentation
- **Terminology inconsistency**: The paper alternates between "DeCoFlow" (Outline_v2) and "MoLE-Flow" (Tables, Method_kr). Unify to one name.
- **Abstract claim**: "8% additional parameters per task" should be "22-42%" or clarify this is for DIA-free configuration.
- **Table 3**: Missing L2P, DualPrompt mentioned in Section 4.1.4 baselines.

### Technical
- **Equation numbering**: Method section lacks equation numbers, making cross-referencing difficult.
- **Hyperparameter sensitivity**: Section 3.3.1 mentions WA constraint ranges from ablation but Table S2 is referenced as supplementary (not provided).

### Missing Details
- **Training time breakdown**: Per-component training time not provided.
- **Failure case frequency**: "8.3% routing confusion" for carpet-grid but total failure rate not aggregated.

---

## 9. Confidence

**Confidence Level**: 4/5 (Confident but not absolutely certain)

I have expertise in normalizing flows, continual learning, and anomaly detection. I have carefully read the paper and supporting materials. My uncertainty stems from:
- Not having access to the actual code implementation
- Some experiments marked as TODO, limiting full assessment
- Cannot verify statistical claims without raw data

---

## 10. Summary for Authors

Your paper presents a promising approach to continual anomaly detection with strong theoretical motivation and impressive empirical results. The key insight that NF's AFP enables safe parameter decomposition is both novel and practically useful.

**Must-fix before camera-ready**:
1. Complete the coupling vs feature-level experiment (Section 4.4.2) or remove the claim
2. Clarify the SVD analysis interpretation
3. Fix terminology inconsistency (DeCoFlow vs MoLE-Flow)
4. Correct the "8%" parameter claim in abstract

**Strongly recommended**:
1. Add PEFT-CL baselines (L2P, DualPrompt)
2. Analyze ViSA P-AP gap root cause
3. Complete Task 0 sensitivity analysis

I look forward to seeing the revised version address these concerns. The core idea is strong and the paper has potential for significant impact in the industrial anomaly detection community.

---

*Review completed following ECCV 2026 guidelines. All scores are preliminary and subject to discussion during the Area Chair meeting.*
