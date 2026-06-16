# VLM vs Human Analysis: Implementation Status

## ✅ FULLY IMPLEMENTED (Publication-Ready)

### Statistical Rigor
1. **Bootstrap 95% CIs** - Per-video AND overall for:
   - Agreement (human-VLM)
   - Human accuracy (vs ground truth)
   - VLM accuracy (vs ground truth)
   
2. **Significance Tests**:
   - Agreement vs chance (p=0.33): Binomial test
   - Human vs VLM accuracy difference: Wilcoxon signed-rank test (paired, per-video)

3. **Calibration Metrics**:
   - **ECE** (Expected Calibration Error) for both human & VLM
   - **MCE** (Maximum Calibration Error) for both human & VLM
   - Reliability curves (visual)

### Performance Metrics
4. **Standard ML Metrics**:
   - Precision/Recall/F1-score for Goal A and Goal B (excluding uncertain "C")
   - Cohen's Kappa for A/B inter-rater reliability
   - Confusion matrices

5. **Per-Video Analysis**:
   - Agreement, accuracy, and deltas with bootstrap CIs
   - **Human vs VLM delta** (Human_Acc - VLM_Acc) per video

### Robustness Checks
6. **Sensitivity to "C" (Uncertain Choice)**:
   - Variant A: Treat C as incorrect (conservative)
   - Variant B: Drop C entirely (optimistic)
   - Compares agreement/accuracy under both treatments

### Error Analysis
7. **Hard Cases Drill-Down**:
   - Identifies worst-performing video (lowest VLM accuracy)
   - Frame-by-frame inspection: choice, confidence, pA/pB, correctness, cues
   - Probability margin analysis (|pA - pB|) to detect ambiguity-driven errors

### Temporal Patterns
8. **Time-to-Threshold** (0.8 confidence):
   - Per-video first timepoint where human/VLM confidence ≥80%
   - Delta (VLM - Human) timing difference

9. **Agreement/Accuracy Over Time**:
   - Plots showing how metrics evolve across video frames

### Trajectory Effects
10. **Ambiguous vs Legible**:
    - Agreement and accuracy by trajectory type
    - Chi-square test for significant differences

### Visualizations (5 plots)
11. **Confidence correlation** - Scatter plot (human vs VLM)
12. **Agreement/accuracy over time** - Line plots with chance level
13. **Pause time comparison** - Bar chart (human pause vs VLM first high-conf)
14. **Per-video triple bars** - Agreement, Human Acc, VLM Acc side-by-side
15. **Calibration curves** - Reliability diagrams for human & VLM

### Outputs
- `analysis_results.json` - All metrics with statistical tests
- `aligned_human_vlm_data.csv` - Full aligned dataset
- 5 high-resolution plots (300 dpi PNG)

---

## ❌ NOT IMPLEMENTED (Lower Priority)

### Medium Priority (Nice to Have)
1. **Classwise confusion matrices** - A-only vs B-only (excluding C)
   - *Rationale: Existing confusion matrix already shows A/B/C breakdown; excluding C is covered by sensitivity analysis*

2. **Confidence vs difficulty bins** - Accuracy by confidence deciles
   - *Rationale: Calibration curves already show this relationship visually*

### Lower Priority (Requires New Data)
3. **Cross-model sanity check** - Compare with GPT-4V or Gemini Pro
   - *Rationale: Requires running additional models; out of scope for single-model analysis*

---

## 🎯 Priority Assessment (Senior Researcher Perspective)

### What Reviewers Will Ask For:
✅ Statistical significance tests → **DONE**  
✅ Bootstrap confidence intervals → **DONE**  
✅ Standard ML metrics (P/R/F1) → **DONE**  
✅ Calibration analysis (ECE/MCE) → **DONE**  
✅ Robustness checks → **DONE (sensitivity to C)**  
✅ Error analysis → **DONE (hard cases drill-down)**

### Publication Readiness:
The current implementation covers **all critical requirements** for an HRI/AI conference/journal submission:
- Rigorous statistical testing
- Comprehensive performance metrics
- Transparent error analysis
- Reproducible visualizations

### What's Skipped (Justification):
- **Classwise confusion**: Redundant with existing analysis
- **Confidence bins**: Covered by calibration curves
- **Cross-model**: Out of scope (requires new experiments)

---

## 📊 Usage

```bash
# Make sure human participant data path is correct in script
# Line 25: PARTICIPANT_DATA_DIR = Path(r"C:\Users\...\participant_data")

python scripts/analyze_vlm_vs_human.py
```

### Expected Runtime:
- ~30-60 seconds (depends on dataset size)
- Bootstrap resampling (2000 iterations) is the bottleneck

### Outputs Location:
```
analysis_results/
├── analysis_results.json              # All metrics & stats
├── aligned_human_vlm_data.csv        # Full aligned dataset
├── confidence_correlation.png         # Human-VLM confidence scatter
├── agreement_over_time.png            # Temporal dynamics
├── pause_time_comparison.png          # Legibility timing
├── agreement_accuracy_by_video.png    # Per-video triple bars
└── calibration_curves.png             # Reliability diagrams
```

---

## 🔬 Key Insights the Analysis Reveals

1. **Statistical Significance**: Are human-VLM agreement rates better than chance?
2. **Calibration Quality**: Do confidence scores reflect actual accuracy?
3. **Hard Case Failures**: Why does VLM fail on ambiguous trajectories?
4. **Robustness**: How sensitive are findings to treatment of uncertain ("C") responses?
5. **Temporal Dynamics**: When do humans vs VLM reach high confidence?
6. **Per-Video Heterogeneity**: Which videos are hardest for VLM vs humans?

---

## 📝 Suggested Paper Sections

### Methods
- Bootstrap resampling (2000 iterations, 95% CI)
- Binomial test (agreement vs chance)
- Wilcoxon signed-rank (paired human vs VLM)
- ECE/MCE calibration metrics

### Results
- Table 1: Overall agreement, accuracy, CIs, p-values
- Table 2: Per-video breakdown with deltas
- Figure 1: Calibration curves (ECE/MCE)
- Figure 2: Agreement/accuracy over time
- Figure 3: Per-video comparison (triple bars)
- Figure 4: Error analysis (hard case frame-by-frame)

### Discussion
- Sensitivity analysis justifies treating C as incorrect
- Error analysis reveals ambiguity-driven failures
- Calibration shows VLM is [well/poorly] calibrated vs humans
- Temporal patterns suggest VLM [lags/leads] human inference
