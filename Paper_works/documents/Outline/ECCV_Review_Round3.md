# ECCV 2026 Paper Review - Round 3 (Final Assessment)

**Paper Title**: DeCoFlow: Structural Decomposition of Normalizing Flows for Continual Anomaly Detection

**Authors**: Hun Im, Jungi Lee, Subeen Cha, Pilsung Kang

**Reviewer**: Area Chair / Senior Reviewer

**Review Date**: 2026-01-21

**Score Evolution**:
- Round 1: 7.3/10 (Weak Accept)
- Round 2: 8.0/10 (Accept)
- Round 3: *This Review*

---

## 1. Assessment of Final Revisions

The authors have submitted a comprehensive revision targeting the 9.0+ threshold. I evaluate each improvement against the criteria for Strong Accept.

### 1.1 [9.0-1] Section 4.4.2: Full 15-Class Validation

**Round 2 Status**: 6-class pilot was "sufficient but full validation would strengthen confidence"

**Round 3 Revision**: Complete 15-class validation with scalability analysis

| Aspect | Assessment |
|--------|------------|
| **Experimental Completeness** | Full 15-class MVTec-AD experiment now included |
| **Scalability Analysis** | New 6-class vs 15-class comparison table showing performance trends |
| **Key Finding** | Coupling-level advantage *amplifies* with scale: +4.8%p (6-class) to +5.75%p (15-class) |
| **Mechanistic Evidence** | Likelihood gap widening (1.84 to 2.12), FM increase (0.8% to 1.2%) for feature-level |

**Verdict**: **Fully Satisfactory**

The scalability analysis is particularly valuable because it demonstrates that the AFP advantage is not merely preserved but *amplified* at scale. Key observations:

1. **DeCoFlow shows minimal degradation**: Only -0.57%p I-AUC from 6 to 15 tasks
2. **Feature-level compounds errors**: FM increases 50% (0.8% to 1.2%), likelihood gap increases 15%
3. **Interpretable mechanism**: "Cumulative distribution shift compounds likelihood miscalibration"

This addresses my Round 2 concern about pilot-scale generalization definitively. The evidence now strongly supports that coupling-level adaptation is fundamentally superior for NF-based continual learning.

---

### 1.2 [9.0-2] Section 4.4.8: Task 0 Sensitivity Analysis (NEW)

**Round 2 Status**: Listed as "for future work" suggestion

**Round 3 Revision**: Complete new section with 5 Task 0 candidates, statistical significance testing, and practitioner guidance

| Aspect | Assessment |
|--------|------------|
| **Experimental Design** | 5 diverse Task 0 candidates covering different characteristics |
| **Statistical Rigor** | p-values for pairwise comparisons; Pearson correlation analysis |
| **Practical Value** | Clear practitioner guidance table with recommendations |
| **Key Finding** | Task 0 impact is statistically detectable but practically minor (~0.34%p I-AUC) |

**Verdict**: **Exceeds Expectations**

This section transforms a potential weakness into a strength. Key contributions:

1. **Robustness Validation**: Maximum I-AUC difference is only 0.34%p across Task 0 choices, demonstrating the frozen base paradigm is robust
2. **Actionable Heuristic**: "If base NF loss converges below 2.4, Task 0 is likely a good anchor" (r=-0.89, p<0.05)
3. **Variance Insight**: Simple Task 0 leads to more consistent downstream performance (0.12% vs 0.31% variance)
4. **P-AP Sensitivity**: Localization is more sensitive to Task 0 choice than detection, providing nuanced guidance

The practitioner guidance table is particularly valuable for deployment. This addition significantly strengthens the paper's practical applicability.

---

### 1.3 [9.0-3] Section 6.1: Multi-Resolution Adaptation Design

**Round 2 Status**: Mentioned as mitigation direction for ViSA P-AP gap

**Round 3 Revision**: Expanded with detailed architecture design and proof-of-concept projections

| Aspect | Assessment |
|--------|------------|
| **Architecture Design** | Clear multi-scale feature pyramid with task-specific fusion |
| **Design Principles** | 4 principles ensuring AFP compatibility |
| **Proof-of-Concept Analysis** | Size-stratified detection rate projections |
| **Implementation Estimates** | Memory (1.5x), inference (1.2x), parameter (+0.03%/task) overhead |

**Verdict**: **Adequately Addressed**

The expanded future work section provides:

1. **Concrete Design**: Not vague hand-waving but a specific architecture diagram with learnable fusion weights
2. **Principled Approach**: Fusion occurs before NF, preserving AFP guarantees
3. **Realistic Projections**: Expected ViSA P-AP improvement of +12-15%p, reducing gap with ReplayCAD from -14.9%p to ~-2%p
4. **Honest Scope**: Explicitly marked as "planned for camera-ready or follow-up work"

**Minor Concern**: Projections are estimates without implementation validation. However, the analysis is sufficiently detailed to be credible, and the honest scoping is appropriate.

---

### 1.4 [9.0-4] Contribution Statement Strengthening

**Round 2 Status**: Contribution was clear but positioning relative to concurrent work was implicit

**Round 3 Revision**: Explicit "first to achieve exact zero forgetting" positioning; ECCV 2026 context added

| Aspect | Assessment |
|--------|------------|
| **Explicit Novelty Claim** | "First framework to enable exact zero forgetting (FM=0.0%, BWT=0.0%)" |
| **Paradigm Positioning** | "forgetting mitigation" vs "forgetting elimination" distinction |
| **Concurrent Work Context** | ECCV 2026 submissions mentioned for positioning |

**Verdict**: **Satisfactory**

The contribution statement now clearly articulates what makes DeCoFlow unique:

1. **Differentiation from Replay Methods**: Zero forgetting *by design*, not by data storage
2. **Differentiation from Regularization**: Exact guarantee, not approximate reduction
3. **Architectural Foundation**: AFP enables what other architectures cannot achieve

The "paradigm shift" framing is strong but substantiated by the empirical evidence (BWT=0.0% with p=1.000).

---

## 2. Comprehensive Quality Assessment

### 2.1 Strengths Reinforced in v4.0

| Strength | Evidence | Impact |
|----------|----------|--------|
| **Zero Forgetting Guarantee** | BWT=0.0%, p=1.000 across all 15 tasks | Fundamental to reliability claims |
| **AFP Validation** | 15-class coupling vs feature-level comparison | Core theoretical contribution validated |
| **Practical Deployment Readiness** | Task 0 sensitivity + practitioner guidance | Ready for industrial adoption |
| **Honest Trade-off Discussion** | P-AP gap explicitly quantified and explained | Scientific integrity |
| **Comprehensive Ablations** | Component, rank, architecture depth, interaction effects | Understanding of design space |

### 2.2 Remaining Minor Concerns

| Concern | Severity | Mitigation |
|---------|----------|------------|
| **Multi-resolution unimplemented** | Low | Clearly scoped as future work; projections are reasonable |
| **ViSA P-AP gap persists** | Low | Root cause analyzed; mitigation path identified |
| **Theoretical bounds informal** | Low | Empirical evidence is comprehensive; formal theory is future work |
| **Single backbone (WideResNet-50)** | Very Low | Standard choice; backbone sensitivity mentioned in future work |

None of these concerns are blocking for a Strong Accept recommendation.

---

## 3. Final Scoring (1-10 scale)

| Criterion | R1 | R2 | R3 | Change (R2->R3) | Justification |
|-----------|-----|-----|-----|-----------------|---------------|
| **Novelty** | 7.5 | 8.0 | 8.5 | +0.5 | AFP reinterpretation now fully validated at scale; "forgetting elimination" paradigm is genuinely new |
| **Technical Quality** | 7.0 | 8.0 | 8.5 | +0.5 | 15-class validation, Task 0 analysis, statistical rigor throughout |
| **Experimental Validation** | 7.5 | 8.0 | 9.0 | +1.0 | Complete validation at full scale; scalability analysis; practitioner-ready guidance |
| **Presentation** | 8.0 | 8.5 | 8.5 | 0.0 | Already strong; revision summary helpful |
| **Significance** | 7.5 | 8.0 | 8.5 | +0.5 | Industrial applicability now clear; Task 0 guidance enables deployment |
| **Overall Score** | 7.3 | 8.0 | **8.7** | **+0.7** | Strong paper meeting threshold for Strong Accept |

### Score Justification

The paper has progressed from addressing critical gaps (R1->R2) to providing comprehensive validation and practical guidance (R2->R3). The 0.7-point increase reflects:

1. **Full-scale validation** of the core claim (Section 4.4.2 at 15 classes)
2. **New practical contribution** (Task 0 sensitivity analysis with deployment guidance)
3. **Clear future direction** for remaining limitation (multi-resolution design)
4. **Explicit positioning** as paradigm shift (forgetting mitigation vs elimination)

---

## 4. Decision

### **Strong Accept** (8.5-9.0 range)

### Justification

DeCoFlow represents a significant contribution to continual anomaly detection with three distinct merits:

**1. Novel Theoretical Contribution**: The reinterpretation of NF's Arbitrary Function Property as a structural foundation for parameter isolation is both novel and validated. The 15-class coupling vs feature-level comparison demonstrates that this is not merely a perspective shift but enables fundamentally superior performance (+5.75%p I-AUC, 15x lower likelihood gap, zero forgetting).

**2. Rigorous Experimental Validation**: The paper now provides:
- Full 15-class validation with scalability analysis
- Task 0 sensitivity study with statistical significance testing
- Comprehensive ablations with interaction effect analysis (ANOVA)
- Honest characterization of limitations (ViSA P-AP gap with root cause quantification)

**3. Practical Deployment Readiness**: The Task 0 sensitivity analysis and practitioner guidance transform this from an academic contribution to a deployment-ready solution. Industrial practitioners can use the provided heuristics to configure DeCoFlow without extensive experimentation.

### Why Not 9.0+?

To reach the 9.0 threshold (Award Candidate level), the paper would need:

1. **Implemented multi-resolution validation**: The current proof-of-concept analysis is credible but unvalidated
2. **Broader dataset evaluation**: MVTec and ViSA are standard but limited; BTAD, DAGM, or domain-specific datasets would strengthen generalization claims
3. **Formal theoretical analysis**: The implicit regularization interpretation is well-supported empirically but lacks formal bounds

These are suggestions for a journal extension, not blocking issues for ECCV acceptance.

---

## 5. Final Comments

### 5.1 What Makes This Paper Noteworthy

1. **Paradigm Shift**: The distinction between "forgetting mitigation" (existing approaches) and "forgetting elimination" (DeCoFlow) is conceptually important. Most CL methods accept some forgetting as inevitable; DeCoFlow achieves mathematically guaranteed BWT=0.0%.

2. **Architectural Insight**: The AFP reinterpretation provides a principled answer to "why normalizing flows?" for continual learning. This insight is transferable to other density estimation applications.

3. **Industrial Relevance**: Manufacturing environments require reliability over novelty. DeCoFlow's zero-forgetting guarantee, combined with practical guidance for Task 0 selection, makes it directly deployable.

4. **Scientific Honesty**: The explicit trade-off discussion (P-AP vs stability) and thorough limitation analysis (ViSA gap) demonstrate scientific maturity.

### 5.2 Remaining Minor Concerns

1. **Multi-resolution remains unimplemented**: The projections are credible, but validation would strengthen the ViSA story
2. **Single backbone evaluation**: WideResNet-50 is standard, but ViT-based backbones are increasingly common
3. **Computational overhead clarity**: The "22-42% per task" range is wide; clearer guidance on when each extreme applies would help

### 5.3 Suggestions for Camera-Ready

1. **Implement multi-resolution pilot**: Even a 3-class ViSA experiment would validate the P-AP improvement projections
2. **Add computational cost breakdown**: Clarify what drives the 22% vs 42% difference
3. **Consider ViT backbone experiment**: Brief comparison to address generalization concerns

---

## 6. Score Evolution Summary

| Round | Score | Decision | Key Improvements |
|-------|-------|----------|------------------|
| Round 1 | 7.3 | Weak Accept | Identified: incomplete experiment, SVD interpretation, ViSA gap |
| Round 2 | 8.0 | Accept | Resolved: 6-class pilot, implicit regularization framing, gap analysis |
| **Round 3** | **8.7** | **Strong Accept** | Full validation: 15-class experiment, Task 0 sensitivity, practitioner guidance |

---

## 7. Confidence

**Confidence Level**: 5/5 (Very Confident)

The revision cycle has been thorough:
- All critical concerns from Round 1 addressed in Round 2
- Round 2 suggestions for improvement largely implemented in Round 3
- The evidence chain from AFP theory to empirical validation to practical guidance is now complete

I am confident this paper will be well-received by the computer vision community and provides meaningful contributions to both the continual learning and anomaly detection literature.

---

## 8. Summary for Authors

Congratulations on an excellent revision. The paper has improved substantially across all three rounds:

**Round 1 -> Round 2**: Addressed fundamental gaps (missing experiment, SVD interpretation, terminology)

**Round 2 -> Round 3**: Elevated from "good paper" to "strong paper" through:
- Full-scale validation with scalability analysis
- New practical contribution (Task 0 sensitivity)
- Clear future direction for remaining limitation

**Key Achievement**: The paper now provides a complete story from theoretical insight (AFP) through rigorous validation (15-class experiments) to practical deployment (Task 0 guidance). This is the hallmark of a strong ECCV paper.

**Recommendation**: Strong Accept. The paper meets the criteria for a significant contribution to ECCV 2026.

---

*Review completed following ECCV 2026 guidelines. Final assessment: Strong Accept (8.7/10).*

*This paper represents a meaningful contribution to continual anomaly detection with novel theoretical insights, comprehensive experimental validation, and practical deployment value.*
