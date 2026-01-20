# ECCV 2026 Paper Review - Round 2 (Revision Assessment)

**Paper Title**: DeCoFlow: Structural Decomposition of Normalizing Flows for Continual Anomaly Detection

**Authors**: Hun Im, Jungi Lee, Subeen Cha, Pilsung Kang

**Reviewer**: Area Chair / Senior Reviewer

**Review Date**: 2026-01-21

**Round 1 Score**: 7.3/10 (Weak Accept)

---

## 1. Assessment of Revisions

### 1.1 P0 Issues (Critical) - All Adequately Addressed

| Issue | Round 1 Concern | Revision | Assessment |
|-------|-----------------|----------|------------|
| **P0-1: Section 4.4.2 incomplete** | Coupling vs feature-level experiment missing, only "Expected Outcome" provided | Complete experimental protocol added: matched parameters (1.93M), 5 metrics, 6-class pilot results with quantitative analysis | **Fully Addressed** |
| **P0-2: Terminology inconsistency** | Mixed "DeCoFlow" and "MoLE-Flow" throughout | 100% unified to "DeCoFlow" | **Fully Addressed** |
| **P0-3: Parameter overhead claim** | Abstract claimed "8%" but actual is 22-42% | Corrected to "22-42%" in Abstract and Section 1.3 | **Fully Addressed** |

**Detailed Assessment of P0-1 (Section 4.4.2)**:

The revised experiment is well-designed with:
- **Fair comparison**: Matched parameter counts (1.93M for both coupling-level LoRA and feature-level prompts)
- **Multiple baselines**: Both prompt injection and adapter MLP approaches tested
- **Comprehensive metrics**: I-AUC, P-AP, Likelihood Gap, FM, Gradient Std
- **Quantitative results**: Coupling-level shows +4.8%p I-AUC, +7.8%p P-AP, 15x lower likelihood gap

The pilot results (6-class) are sufficient to support the AFP claim because:
1. The performance gap is substantial (not marginal)
2. The likelihood miscalibration metric directly validates the "density manifold disruption" hypothesis
3. The gradient stability analysis provides mechanistic explanation

**Minor concern**: Full 15-class validation would strengthen confidence, but the 6-class pilot is acceptable for publication.

---

### 1.2 P1 Issues - All Adequately Addressed

| Issue | Round 1 Concern | Revision | Assessment |
|-------|-----------------|----------|------------|
| **P1-1: SVD interpretation** | Post-hoc rationalization from "intrinsically low-rank" to "selective learning" | Reframed as "implicit regularization" with supporting evidence | **Adequately Addressed** |
| **P1-2: PEFT-CL baselines** | L2P, DualPrompt mentioned but not compared | Added detailed discussion in Related Work (2.2) explaining methodological mismatch; controlled comparison in 4.4.2 | **Adequately Addressed** |
| **P1-3: ViSA P-AP gap** | -14.9%p gap unexplained | Added Section 4.2.4 with root cause analysis (spatial resolution 70%, domain shift 25%, routing 5%) | **Fully Addressed** |

**Detailed Assessment of P1-1 (SVD Interpretation)**:

The implicit regularization framing is now well-supported:

1. **Rank insensitivity evidence** (Table 8): Performance variance <0.1%p across ranks 16-128. The authors correctly argue that if LoRA were approximating critical directions, higher ranks should improve performance.

2. **Alternative hypothesis tested**: Section 4.4.3 now includes: "Randomly selecting 64 directions performs similarly to top-64 directions (97.8% vs 98.0% I-AUC)." This directly refutes the "selective learning of critical directions" interpretation.

3. **Literature support**: Citations to Arora et al., 2019 and Li et al., 2020 on implicit regularization in over-parameterized networks provide theoretical grounding.

4. **Variance comparison**: "LoRA configurations show lower variance (+/-0.12) than full fine-tuning (+/-0.35)" supports the regularization hypothesis.

This reframing transforms a weakness into a contribution: the paper now provides novel insights on why PEFT methods work for density estimation models.

**Detailed Assessment of P1-2 (PEFT-CL Baselines)**:

The authors chose explanation over direct comparison, which I find acceptable:

1. **Methodological argument** (Section 2.2): Three clear reasons why L2P/DualPrompt are not directly applicable:
   - Operating level mismatch (feature vs coupling)
   - Objective mismatch (decision boundary vs exact likelihood)
   - Task granularity mismatch (image vs patch-level)

2. **Controlled alternative**: Section 4.4.2 provides a fair comparison at matched abstraction level (feature-level adaptation within the NF framework).

3. **Baseline note** (Section 4.1): Explicitly states why direct comparison would be "methodologically inappropriate."

This approach is more scientifically rigorous than forcing an unfair comparison.

**Detailed Assessment of P1-3 (ViSA P-AP Gap)**:

Section 4.2.4 provides a thorough root cause analysis:

1. **Quantified contributions**: 70% spatial resolution, 25% domain shift, 5% routing cascade
2. **Supporting data tables**: Feature statistics comparison (Mean +52%, Std +62%)
3. **Oracle experiment**: Routing errors account for only 0.5%p P-AP degradation
4. **Mitigation direction**: Multi-resolution adaptation proposed as future work

This transforms an unexplained weakness into an honest, well-characterized limitation.

---

### 1.3 Remaining Concerns (Minor)

| Concern | Severity | Notes |
|---------|----------|-------|
| **6-class pilot vs 15-class full** | Low | Pilot results are convincing; full validation would be ideal but not blocking |
| **Future work items** | Low | Task 0 sensitivity, multi-resolution adaptation mentioned but not implemented |
| **Theoretical bounds** | Low | Formal connection between LoRA rank and generalization remains informal |

---

## 2. Updated Scoring (1-10 scale)

| Criterion | Round 1 | Round 2 | Change | Justification |
|-----------|---------|---------|--------|---------------|
| **Novelty** | 7.5 | 8.0 | +0.5 | Implicit regularization interpretation adds novel insight; AFP reinterpretation validated |
| **Technical Quality** | 7.0 | 8.0 | +1.0 | SVD interpretation now well-supported; coupling vs feature-level experiment validates core claim |
| **Experimental Validation** | 7.5 | 8.0 | +0.5 | ViSA gap explained; critical experiment completed (pilot scale) |
| **Presentation** | 8.0 | 8.5 | +0.5 | Terminology unified; revision summary helpful; honest trade-off discussion |
| **Significance** | 7.5 | 8.0 | +0.5 | Zero forgetting guarantee now fully validated; practical implications clearer |
| **Overall Score** | **7.3** | **8.0** | **+0.7** | All major concerns addressed; paper now ready for publication |

---

## 3. Decision

**Accept** (Positive)

### Justification

The authors have thoroughly addressed all P0 and P1 concerns from Round 1:

1. **Critical experiment completed** (P0-1): The coupling-level vs feature-level comparison now validates the AFP claim with quantitative evidence. The 6-class pilot shows substantial performance gaps (+4.8%p I-AUC) and mechanistic differences (15x likelihood gap), sufficient to support the paper's core contribution.

2. **Interpretation strengthened** (P1-1): The shift from "selective learning" to "implicit regularization" is well-supported by rank insensitivity analysis and random direction comparison. This reframing is scientifically more accurate and adds a novel insight for PEFT in density estimation.

3. **Limitations acknowledged honestly** (P1-3): The ViSA P-AP gap analysis demonstrates scientific rigor by quantifying root causes rather than hand-waving.

4. **Presentation polished** (P0-2, P0-3): Terminology unified, parameter claims corrected, revision summary provided.

The paper now presents a coherent story: NF's AFP enables safe parameter decomposition for continual learning, LoRA provides implicit regularization beneficial for density estimation, and the trade-off between zero forgetting and fine-grained localization is explicit and principled.

---

## 4. Remaining Suggestions (to reach 8.5+)

### For Final Version / Camera-Ready:

1. **Scale up Section 4.4.2 experiment**: While 6-class pilot is sufficient, full 15-class validation would strengthen confidence. Consider adding this to supplementary materials.

2. **Task 0 sensitivity**: Brief analysis of how Task 0 choice affects downstream performance would address a natural question from practitioners.

3. **Multi-resolution future direction**: Given the ViSA gap analysis, a proof-of-concept for multi-resolution adaptation would significantly enhance the paper's practical value.

### For Future Work (not required for acceptance):

4. **Theoretical bounds**: Formal analysis connecting LoRA rank to generalization bounds in density estimation would elevate this from a strong empirical paper to a foundational contribution.

5. **Beyond MVTec/ViSA**: Evaluation on additional AD benchmarks (e.g., BTAD, DAGM) would demonstrate broader applicability.

---

## 5. Summary for Authors

Congratulations on a strong revision. The paper has improved substantially:

- **Section 4.4.2** now provides convincing evidence for the AFP-based decomposition advantage
- **SVD interpretation** is scientifically rigorous with the implicit regularization framing
- **ViSA analysis** transforms an unexplained gap into a well-characterized, honest limitation
- **Presentation** is now consistent and professional

The core contribution - that NF's Arbitrary Function Property enables zero-forgetting continual learning through coupling-level parameter decomposition - is now well-validated both theoretically and empirically.

**Recommendation**: Accept with minor revisions (scale up 4.4.2 experiment for camera-ready if possible).

---

## 6. Score Evolution Summary

| Round | Score | Decision | Key Issues |
|-------|-------|----------|------------|
| Round 1 | 7.3 | Weak Accept | P0: Incomplete experiment, terminology, parameter claim; P1: SVD interpretation, baselines, ViSA gap |
| **Round 2** | **8.0** | **Accept** | All P0/P1 issues addressed; remaining concerns are minor |

---

## 7. Confidence

**Confidence Level**: 4/5 (Confident)

The revision addresses all substantive concerns from Round 1. My remaining uncertainty stems from:
- Pilot study (6-class) rather than full-scale (15-class) for Section 4.4.2
- Cannot verify implementation details without code access

However, the evidence presented is sufficient for acceptance.

---

*Review completed following ECCV 2026 guidelines. Score increased from 7.3 to 8.0 based on thorough revision assessment.*
