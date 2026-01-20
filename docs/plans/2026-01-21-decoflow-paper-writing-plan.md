# DeCoFlow Paper Writing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the DeCoFlow (MoLE-Flow) research paper for ECCV 2026 submission, converting existing Korean/English bullet point outlines into publication-quality full paragraph prose with proper citations.

**Architecture:** Two-stage writing process: (1) organize existing content into structured outlines with key points and citations, (2) convert to flowing academic prose following IMRAD structure. All sections must be written in complete paragraphs - never bullet points in final manuscript.

**Tech Stack:** LaTeX (ECCV format), scientific-writing skill, research-lookup for citations, semantic-scholar MCP for paper retrieval.

---

## Current State Assessment

### Completed Experiments
- ✅ 4.2.2: Cross-dataset generalization (ViSA)
- ✅ 4.2.3: Zero forgetting verification (BWT=0)
- ✅ 4.3.1-4: Component ablation, Interaction effect, LoRA rank, Architecture depth
- ✅ 4.4.1: Architecture comparison (NF vs VAE/AE/T-S)
- ✅ 4.4.3: SVD analysis

### Remaining Experiments (TODO)
- 🔴 4.4.2: Coupling-level vs Feature-level adaptation
- 🟡 4.3.5: Task 0 selection sensitivity
- 🟡 4.4.5: DIA transformation analysis
- 🟢 4.3.6: Long-sequence scalability (30+ tasks)

### Existing Documents
- `Paper_works/documents/Outline/Outline.md`: Full Korean/English outline with bullets
- `Paper_works/main.md`: Complete English draft with detailed structure
- `Paper_works/documents/Outline/storyline.md`: Key storyline and arguments
- `Paper_works/documents/Experiments/Experiments_main.md`: Experiment results
- `Paper_works/documents/Outline/4.Experiments_Redesigned.md`: Experiment design v4.0

---

## Task 1: Abstract Writing

**Files:**
- Read: `Paper_works/main.md:9-23` (existing abstract outline)
- Create: `Paper_works/latex/sections/00_abstract.tex`
- Reference: `Paper_works/documents/Outline/Outline.md:1-93` (introduction for context)

**Step 1: Extract key points from existing outline**

The abstract must cover:
- Problem: Continual AD faces isolation-efficiency dilemma
- Insight: NF's Arbitrary Function Property enables safe parameter decomposition
- Method: MoLE-Flow/DeCoFlow with frozen base + task-specific LoRA
- Components: WA, TAL, DIA as integral components
- Results: 98.05% I-AUC, 55.80% P-AP with FM=0.0 on MVTec-AD 15 classes

**Step 2: Write structured abstract (200-250 words)**

```latex
% Structure: Background (1 sent) → Problem (1-2 sent) → Insight (1-2 sent)
% → Method (2-3 sent) → Results (1-2 sent) → Impact (1 sent)
```

**Step 3: Review against ECCV requirements**

Run: Verify word count (max 300 words for ECCV)
Expected: 200-250 words, structured flow

**Step 4: Commit**

```bash
git add Paper_works/latex/sections/00_abstract.tex
git commit -m "docs: draft abstract for DeCoFlow paper"
```

---

## Task 2: Introduction Section (Section 1)

**Files:**
- Read: `Paper_works/main.md:27-101` (existing intro outline)
- Read: `Paper_works/documents/Outline/Outline.md:18-93` (Korean detailed outline)
- Create: `Paper_works/latex/sections/01_introduction.tex`

**Step 1: Identify key subsections and citations needed**

Subsections:
1.1 Background (Continual AD necessity)
1.2 Why forgetting is critical in AD (vs classification)
1.3 Limitations of existing approaches (Replay, Hard isolation, Efficient adaptation)
1.4 Isolation-Efficiency Dilemma (key problem formulation)
1.5 Key Insight (NF's Arbitrary Function Property)
1.6 MoLE-Flow Overview
1.7 Contributions (4 points)

**Step 2: Research-lookup for missing citations**

Citations needed:
- Catastrophic forgetting: [McCloskey & Cohen 1989], [French 1999]
- Continual AD methods: [CADIC], [ReplayCAD], [UCAD], [DNE]
- NF for AD: [FastFlow], [CFLOW-AD], [MSFlow], [HGAD]
- LoRA: [Hu et al., ICLR 2022]
- Unified AD: [UniAD], [OmniAL], [MambaAD]

**Step 3: Convert bullets to full paragraphs**

For each subsection:
- Transform bullet points into complete sentences
- Add transitions between paragraphs
- Integrate citations naturally within prose
- Ensure logical flow building to the key insight

**Step 4: Write contributions in prose form**

```latex
Our contributions are fourfold. First, we establish a structural connection
between NF's arbitrary function property and continual learning's parameter
decomposition requirements, showing that NF uniquely enables \emph{exact
zero forgetting} (FM=0.0)...
```

**Step 5: Review for clarity and length**

Run: Word count check (target: ~1200-1500 words for intro)
Expected: Clear problem statement → insight → solution → contributions

**Step 6: Commit**

```bash
git add Paper_works/latex/sections/01_introduction.tex
git commit -m "docs: draft introduction section with full prose"
```

---

## Task 3: Related Work Section (Section 2)

**Files:**
- Read: `Paper_works/main.md:105-132` (existing related work outline)
- Read: `Paper_works/documents/Outline/Outline.md:96-175` (detailed Korean outline)
- Read: `Paper_works/documents/Outline/2.Related_works.md` (if exists)
- Create: `Paper_works/latex/sections/02_related_work.tex`

**Step 1: Structure subsections**

2.1 Unified Multi-class Anomaly Detection
- One-class → Multi-class paradigm shift
- UniAD, OmniAL, MambaAD, DiAD
- NF methods: FastFlow, MSFlow, HGAD, VQ-Flow
- Limitation: static capacity, structural rigidity

2.2 Continual Learning: From Trade-off to Structural Separation
- Stability-plasticity dilemma
- PEFT emergence: LoRA and variants
- Gap: Decision boundary vs density manifold

2.3 Continual Learning in Anomaly Detection
- Replay-based: CADIC, ReplayCAD
- Regularization: DNE, CFRDC
- Architecture: SurpriseNet, UCAD
- Gap summary: No method satisfies all 4 requirements

**Step 2: Use semantic-scholar MCP for citation verification**

Verify and retrieve:
- UniAD [You et al., NeurIPS 2022]
- HGAD [Yao et al., ECCV 2024]
- MambaAD [He et al., NeurIPS 2024]
- GainLoRA [Liang et al., arXiv 2025]
- CoSO [Cheng et al., NeurIPS 2025]

**Step 3: Write each subsection in full paragraphs**

Key: End related work with clear gap statement leading to our method:
```latex
In summary, existing CAD methods fail to simultaneously satisfy:
(1) no replay, (2) precise density modeling, (3) parameter efficiency,
and (4) task-agnostic routing. MoLE-Flow addresses all four through
NF-native parameter decomposition with integrated routing.
```

**Step 4: Commit**

```bash
git add Paper_works/latex/sections/02_related_work.tex
git commit -m "docs: draft related work section with comprehensive literature review"
```

---

## Task 4: Method Section (Section 3)

**Files:**
- Read: `Paper_works/main.md:136-345` (existing method outline)
- Read: `Paper_works/documents/Outline/3.Method.md` (detailed method)
- Read: `Paper_works/documents/Outline/Outline.md:178-425` (Korean method outline)
- Create: `Paper_works/latex/sections/03_method.tex`

**Step 1: Organize method structure**

3.1 Problem Formulation & Architecture Overview
- CAD setting definition
- Pipeline flow diagram reference
- Training strategy (Task 0 vs Task 1+)

3.2 Why Normalizing Flow: Arbitrary Function Property
- Mathematical formulation
- AFP definition and proof sketch
- Comparison table with other architectures

3.3 MoLE Block: LoRA-Integrated Coupling Layer
- Subnet decomposition equation
- Context-aware subnet design (s-network, t-network)
- Parameter overhead calculation

3.4 Integral Components: Compensating for Frozen Base Rigidity
3.4.1 Whitening Adapter (WA) - Distribution alignment
3.4.2 Tail-Aware Loss (TAL) - Gradient redistribution
3.4.3 Deep Invertible Adapter (DIA) - Nonlinear manifold correction

3.5 Prototype-Based Task Routing
- Mahalanobis distance formulation
- Prototype construction during training

**Step 2: Write mathematical formulations clearly**

Ensure equations are:
- Numbered for reference
- Accompanied by prose explanation
- Consistent notation (use notation table)

**Step 3: Write each subsection with proper derivations**

For 3.2 AFP section:
```latex
\begin{proposition}[Arbitrary Function Property]
Let $s, t: \mathbb{R}^{d/2} \to \mathbb{R}^{d/2}$ be arbitrary measurable
functions. The affine coupling transformation
$T(x) = [x_1; x_2 \odot \exp(s(x_1)) + t(x_1)]$ is bijective with tractable
Jacobian determinant $\log|\det \nabla T| = \sum_i s_i(x_1)$,
independent of the internal parameterization of $s$ and $t$.
\end{proposition}
```

**Step 4: Create method figure reference**

Reference Figure 1 (architecture overview) with:
- Left: Full pipeline
- Center: MoLE block detail
- Right: Prototype routing

**Step 5: Commit**

```bash
git add Paper_works/latex/sections/03_method.tex
git commit -m "docs: draft method section with mathematical formulations"
```

---

## Task 5: Experiments Section (Section 4)

**Files:**
- Read: `Paper_works/documents/Outline/4.Experiments_Redesigned.md` (complete experiment design)
- Read: `Paper_works/documents/Experiments/Experiments_main.md` (results data)
- Read: `Paper_works/main.md:349-605` (existing experiment outline)
- Create: `Paper_works/latex/sections/04_experiments.tex`

**Step 1: Write 4.1 Experimental Setup**

- Dataset descriptions (MVTec-AD, ViSA)
- CL protocol (1×1 scenario)
- Metrics (I-AUC, P-AP, FM, BWT)
- Baselines (categorized by approach)
- Implementation details

**Step 2: Write 4.2 Main Results**

4.2.1 MVTec-AD comparison table (Table 1)
- Insert results from Experiments_main.md
- Write prose analysis highlighting:
  - SOTA I-AUC (98.05%) with FM=0.0
  - Only method achieving true zero forgetting
  - Comparison with replay methods (CADIC, ReplayCAD)

4.2.2 ViSA results with routing accuracy
- I-AUC: 90.0%, FM=0.0, Routing: 95.8-100%
- Analysis of P-AP gap vs ReplayCAD

4.2.3 Zero Forgetting Verification
- BWT=0.0 verification with statistical test
- Task retention curve figure reference

**Step 3: Write 4.3 Ablation Studies**

4.3.1 Component ablation table
4.3.2 Interaction effect analysis (2×2 factorial ANOVA)
4.3.3 LoRA rank sensitivity (16-128 all perform similarly)
4.3.4 Architecture depth analysis

**Step 4: Write 4.4 Analysis**

4.4.1 Architecture comparison (NF vs VAE/AE/T-S) - completed experiment
4.4.3 SVD analysis with interpretation revision
4.4.6 Routing mechanism analysis
4.4.7 Computational cost comparison

**Step 5: Format all tables properly**

```latex
\begin{table}[t]
\centering
\caption{Comparison with state-of-the-art methods on MVTec-AD...}
\begin{tabular}{lcccc}
\toprule
Method & I-AUC↑ & P-AP↑ & FM↓ & Routing \\
\midrule
...
\bottomrule
\end{tabular}
\end{table}
```

**Step 6: Commit**

```bash
git add Paper_works/latex/sections/04_experiments.tex
git commit -m "docs: draft experiments section with results tables"
```

---

## Task 6: Conclusion Section (Section 5)

**Files:**
- Read: `Paper_works/main.md:857-873` (existing conclusion outline)
- Create: `Paper_works/latex/sections/05_conclusion.tex`

**Step 1: Write conclusion structure**

- Summary of contributions (1 paragraph)
- Key results recap (1 paragraph)
- Limitations (1 paragraph, 4 points)
- Future work directions (1 paragraph, 4 points)

**Step 2: Write in full prose**

```latex
We presented MoLE-Flow, a continual learning framework for anomaly detection
that resolves the isolation-efficiency dilemma through structural decomposition
of normalizing flows. By identifying the arbitrary function property as a
theoretical foundation for safe parameter decomposition, we achieved
\emph{exact zero forgetting} (FM=0.0) with only 22-42\% parameter overhead
per task...
```

**Step 3: Commit**

```bash
git add Paper_works/latex/sections/05_conclusion.tex
git commit -m "docs: draft conclusion section"
```

---

## Task 7: References and BibTeX

**Files:**
- Create: `Paper_works/latex/references.bib`
- Reference: All citations from sections

**Step 1: Compile citation list from all sections**

Categories:
- Continual Learning (EWC, PackNet, LoRA, etc.)
- Anomaly Detection (FastFlow, PatchCore, UniAD, etc.)
- Continual AD (CADIC, ReplayCAD, UCAD, DNE, etc.)
- Datasets (MVTec-AD, ViSA)
- Foundation (RealNVP, NeurIPS proceedings)

**Step 2: Use semantic-scholar MCP to retrieve proper BibTeX**

For each citation:
```bash
# Use mcp__semantic-scholar__get_paper to retrieve metadata
# Convert to BibTeX format
```

**Step 3: Verify all citations are referenced in text**

Run: Check for unreferenced bib entries
Expected: All entries have at least one \cite{}

**Step 4: Commit**

```bash
git add Paper_works/latex/references.bib
git commit -m "docs: add comprehensive BibTeX references"
```

---

## Task 8: Figures and Tables

**Files:**
- Read: `Paper_works/figures/` directory
- Create: `Paper_works/latex/figures/` (copy/organize)
- Update: All section files with proper figure references

**Step 1: Inventory existing figures**

From `Paper_works/figures/`:
- architecture_comparison.png
- svd_spectrum.png
- effective_rank_histogram.png
- fig1_gradient_concentration.png
- fig5_ablation_heatmap.png

**Step 2: Create missing figures list**

Required:
- Figure 1: Architecture overview (pipeline + MoLE block + routing)
- Figure 2: Interaction effect analysis (3 panels)
- Figure 3: Learning curves comparison
- Figure 7: Task retention curve (BWT verification)

**Step 3: Generate schematics using scientific-schematics skill**

```bash
python scripts/generate_schematic.py "MoLE-Flow architecture overview showing
feature extraction, preprocessing adapters, decomposed NF with frozen base and
LoRA, DIA blocks, and prototype router" -o figures/fig1_architecture.png
```

**Step 4: Insert figure references in LaTeX**

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig1_architecture.png}
\caption{Overview of MoLE-Flow architecture...}
\label{fig:architecture}
\end{figure}
```

**Step 5: Commit**

```bash
git add Paper_works/latex/figures/ Paper_works/latex/sections/*.tex
git commit -m "docs: add figures and references"
```

---

## Task 9: Supplementary Material

**Files:**
- Create: `Paper_works/latex/supplementary.tex`
- Reference: `Paper_works/main.md:906-1107` (appendix content)

**Step 1: Structure supplementary sections**

A. Implementation Details
B. Task 0 Selection Analysis
C. LoRA Scaling Factor Ablation
D. WA Constraint Bounds Ablation
E. TAL Top-k Ratio Ablation
F. Extended Ablation Results
G. Task Order Sensitivity
H. Per-class Results
I. SVD Analysis Details
J. Resolution Analysis for ViSA
K. Statistical Analysis Protocol
L. Failure Case Analysis

**Step 2: Write each section in full prose**

**Step 3: Commit**

```bash
git add Paper_works/latex/supplementary.tex
git commit -m "docs: draft supplementary material"
```

---

## Task 10: Final Integration and Review

**Files:**
- Read: All section files
- Update: `Paper_works/latex/main.tex`
- Create: Final compiled PDF

**Step 1: Integrate all sections into main.tex**

```latex
\documentclass{eccv}
\input{sections/00_abstract}
\begin{document}
\maketitle
\input{sections/01_introduction}
\input{sections/02_related_work}
\input{sections/03_method}
\input{sections/04_experiments}
\input{sections/05_conclusion}
\bibliography{references}
\end{document}
```

**Step 2: Check page limit (8 pages + references + supplementary)**

Run: `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
Expected: 8-page main paper, properly formatted

**Step 3: Proofread for consistency**

Check:
- Terminology consistency (MoLE-Flow vs DeCoFlow)
- Notation consistency (check notation table)
- Figure/table references
- Citation format

**Step 4: Final commit**

```bash
git add Paper_works/latex/
git commit -m "docs: complete DeCoFlow paper draft for ECCV 2026"
```

---

## Timeline Summary

| Task | Description | Priority | Est. Complexity |
|------|-------------|----------|-----------------|
| 1 | Abstract | P0 | Low |
| 2 | Introduction | P0 | High |
| 3 | Related Work | P0 | Medium |
| 4 | Method | P0 | High |
| 5 | Experiments | P0 | High |
| 6 | Conclusion | P1 | Low |
| 7 | References | P1 | Medium |
| 8 | Figures | P1 | Medium |
| 9 | Supplementary | P2 | Medium |
| 10 | Integration | P0 | Medium |

---

## Key Writing Guidelines

1. **Never use bullet points in final manuscript** - all content must be flowing prose
2. **Use two-stage process**: outline with bullets → convert to paragraphs
3. **Integrate citations naturally** within sentences, not as lists
4. **Maintain consistent terminology** throughout (prefer DeCoFlow or MoLE-Flow, not both)
5. **Use active voice** where appropriate for clarity
6. **Quantify claims** with specific numbers from experiments
7. **Connect each section** to the core insight (AFP enables zero forgetting)

---

## Dependencies

- Tasks 1-6 can proceed in parallel (independent sections)
- Task 7 (references) depends on completion of Tasks 1-6
- Task 8 (figures) can proceed in parallel
- Task 9 (supplementary) can proceed after main sections
- Task 10 (integration) requires all previous tasks

---

*Plan created: 2026-01-21*
*Target venue: ECCV 2026*
*Estimated total effort: 10 major writing tasks*
