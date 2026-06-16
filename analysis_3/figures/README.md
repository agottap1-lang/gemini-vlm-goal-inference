# Presentation Figures - Multi-Model VLM Evaluation

**All figures are available in both PNG (300 DPI) and PDF formats for maximum flexibility.**

---

## 📊 Figure 1: Model Comparison
**File:** `figure1_model_comparison.png/pdf`

**What it shows:**
- 3-panel comparison of all models
- Panel A: VLM Accuracy with human baseline (98.25%)
- Panel B: IoU (agreement with human judgments)
- Panel C: Number of anomalies (reliability metric)

**Key findings:**
- **gemini-2.5-pro** is the clear winner: 93.75% accuracy
- 12% improvement over Flash baseline
- 56% fewer anomalies than Flash

**When to use:** 
- Opening slide to show overall results
- When you need to justify model selection

---

## 🔥 Figure 2: Per-Video Accuracy Heatmap
**File:** `figure2_per_video_heatmap.png/pdf`

**What it shows:**
- Color-coded heatmap: Green = high accuracy, Red = low accuracy
- Each row = one model
- Each column = one video
- Numbers show exact accuracy percentages

**Key findings:**
- All models achieve 100% on: amb_l_block, amb_r_block, le_d_drawer_close, le_l_block
- All models struggle with: amb_d_drawer_close, amb_to_drawer_close, le_r_block, le_t_drawer_close
- Pro model has the most green cells (consistent high performance)

**When to use:**
- To show which videos are hardest
- To demonstrate consistency across different scenarios
- To highlight problem areas

---

## 🎯 Figure 3: Worst-Performing Videos
**File:** `figure3_worst_videos.png/pdf`

**What it shows:**
- Grouped bar chart ranking videos from worst to best (left to right)
- Red shading highlights the 4 worst performers
- Shows all 3 models' performance on each video

**Key findings:**
- 4 videos consistently problematic across all models:
  - **amb_d_drawer_close**: Confusion between top/bottom drawers
  - **amb_to_drawer_close**: Early wrong high-confidence predictions
  - **le_r_block**: Spatial reasoning failure (left/right confusion)
  - **le_t_drawer_close**: Drawer discrimination difficulty

**When to use:**
- To justify why certain videos are harder
- To discuss model limitations
- To defend against criticism ("even the best model struggles here")

---

## 🚨 Figure 4: Anomaly Breakdown
**File:** `figure4_anomaly_breakdown.png/pdf`

**What it shows:**
- Panel A: Stacked bar chart showing anomaly types by model
- Panel B: Pie chart showing Flash's anomaly distribution (most problematic)

**Anomaly types:**
- **Choice Flip**: Changed answer after high confidence (temporal inconsistency)
- **High Conf Wrong**: Wrong prediction with >85% confidence (overconfidence)
- **Final Uncertain**: Still uncertain at video end (indecisiveness)
- **Low Conf Legible**: Low confidence on legible trajectory (underconfidence)

**Key findings:**
- Flash has 9 anomalies (most unreliable)
- Pro has only 4 anomalies (most reliable)
- Choice flips are the most common issue across all models

**When to use:**
- To discuss reliability and trustworthiness
- To explain failure modes
- To show why Pro is preferred over Flash despite speed

---

## ⚖️ Figure 5: Model Trade-offs
**File:** `figure5_model_tradeoffs.png/pdf`

**What it shows:**
- Panel A: Accuracy vs Speed (inference time)
  - Top-left quadrant = best (high accuracy, fast)
- Panel B: Accuracy vs Reliability (anomalies)
  - Top-left quadrant = best (high accuracy, few anomalies)

**Key findings:**
- **Pro is fastest AND most accurate** (unexpected win!)
- Flash is slowest despite being "lightweight"
- Pro has best trade-off: highest accuracy, fewest anomalies, fastest speed
- Gemini-3-pro-preview is middle-ground (likely preview limitations)

**When to use:**
- To answer "Why not just use Flash if it's cheaper?"
- To justify computational cost
- To show that Pro dominates on all metrics

---

## 🏆 Figure 6: Summary Infographic
**File:** `figure6_summary_infographic.png/pdf`

**What it shows:**
- One-page executive summary with:
  - Winner announcement banner
  - Model accuracy comparison (horizontal bars)
  - Anomaly count comparison (horizontal bars)
  - Worst-performing videos identification
  - Key findings summary box
  - Clear recommendation

**Key findings (all in one place):**
- gemini-2.5-pro: 93.75% accuracy (best)
- 12% improvement over Flash, 56% fewer anomalies
- Gemini-3-pro underperforms (likely preview issues)
- 4 videos consistently difficult (spatial reasoning)
- Clear recommendation: Deploy Pro for production

**When to use:**
- **First or last slide** of presentation
- Executive summary for non-technical audiences
- Quick reference for decision-makers
- When you have limited time

---

## 📋 Quick Reference Guide

### For Technical Presentations:
1. **Figure 6** (Summary) - Overview
2. **Figure 1** (Model Comparison) - Justify selection
3. **Figure 4** (Anomaly Breakdown) - Explain reliability
4. **Figure 5** (Trade-offs) - Show comprehensive dominance
5. **Figure 3** (Worst Videos) - Discuss limitations

### For Executive Presentations:
1. **Figure 6** (Summary) - All you need!
2. **Figure 1** (Model Comparison) - If they want more detail
3. **Figure 5** (Trade-offs) - If they ask about cost/speed

### For Academic/Conference Papers:
1. **Figure 1** (Model Comparison) - Core results
2. **Figure 2** (Heatmap) - Per-video breakdown
3. **Figure 4** (Anomaly Breakdown) - Failure analysis
4. **Figure 3** (Worst Videos) - Difficulty analysis
5. **Figure 5** (Trade-offs) - Multi-objective evaluation

---

## 💡 Presentation Tips

### Color Coding:
- **Green** (gemini-2.5-pro) = Winner
- **Orange** (gemini-2.5-flash) = Baseline
- **Blue** (gemini-3-pro-preview) = Experimental

### Key Messages:
1. "Pro model achieves 93.75% accuracy - closest to human (98.25%)"
2. "12% improvement over baseline with 56% fewer errors"
3. "4 videos reveal fundamental challenges in spatial reasoning"
4. "Pro dominates on ALL metrics: accuracy, speed, reliability"
5. "Clear recommendation: Deploy Pro for production"

### Common Questions & Answers:
**Q:** "Why not use Flash if it's faster?"
**A:** Actually, Pro IS faster! See Figure 5.

**Q:** "What about the newest model (Gemini-3)?"
**A:** Preview version underperforms. See Figure 1.

**Q:** "What are the hardest scenarios?"
**A:** Spatial reasoning and drawer discrimination. See Figure 3.

**Q:** "How reliable is the best model?"
**A:** Only 4 anomalies in 31 predictions (87% reliability). See Figure 4.

---

## 📁 File Formats

- **PNG**: High-resolution (300 DPI), use for PowerPoint, Google Slides, Keynote
- **PDF**: Vector format, use for LaTeX, publications, scalable displays

Both formats are presentation-ready and publication-quality.

---

**Generated:** April 2, 2026  
**Analysis Folder:** analysis_3/  
**Total Figures:** 6 (12 files counting both formats)
