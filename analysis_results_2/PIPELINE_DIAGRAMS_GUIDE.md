# Pipeline Diagrams - Usage Guide for ICRA Paper/Thesis

## 📊 Generated Diagrams

### 1. Main Pipeline Diagram (`pipeline_diagram.png/pdf`)
**Purpose:** Comprehensive system overview showing end-to-end pipeline  
**Best for:** Methods section, system overview, first figure in paper

**Key Features:**
- ✅ Complete data flow from input to evaluation
- ✅ Shows parallel VLM and Human baseline paths
- ✅ Clearly indicates fair comparison methodology
- ✅ Includes evaluation metrics and results
- ✅ Modular component grouping with dashed boxes
- ✅ Legend distinguishing VLM pipeline (solid) vs Human baseline (dashed)

**Components Shown:**
1. **Input Module:** Robot motion videos (8 videos, 4 ambiguous, 4 legible)
2. **Preprocessing Module:** Frame extraction and timepoint selection
3. **VLM Module:** Cumulative frame sequence → Vision-Language Model → Goal inference prompt
4. **Output Module:** Goal prediction {A, B, C} with confidence scores
5. **Human Baseline Module:** Parallel path with 8 participants using same timepoints
6. **Evaluation Module:** Accuracy vs ground truth, Human-VLM agreement (IoU), inference timing

**Results Highlighted:**
- Human: 66.7% | VLM: 64.5% | IoU: 74.2%
- Fair comparison note emphasized

---

### 2. Detailed VLM Architecture (`vlm_architecture_detailed.png/pdf`)
**Purpose:** Technical deep-dive into VLM processing  
**Best for:** Methods section (VLM subsection), technical appendix

**Key Features:**
- ✅ Shows internal VLM components
- ✅ Illustrates multi-modal fusion (vision + language)
- ✅ Details the vision encoder and language model stages
- ✅ Includes goal options explanation
- ✅ Notes trajectory types (legible vs ambiguous)

**Components Shown:**
1. **Input:** Frame sequence f₀, f₁, ..., fₜ (cumulative)
2. **Vision Encoder:** Image embedding → Visual features
3. **Language Model:**
   - Task prompt ("What is the robot's goal?")
   - Multi-modal fusion (Vision + Language)
   - Self-attention layers
4. **Output:** Classifier → Goal prediction {A, B, C}
5. **Goal Options:**
   - A: Move to left block / Close bottom drawer
   - B: Move to right block / Close top drawer
   - C: Uncertain / Insufficient information

---

## 🎓 How to Use in Your Paper

### For ICRA Submission

**Figure 1: System Pipeline** (`pipeline_diagram.pdf`)
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{pipeline_diagram.pdf}
\caption{Overview of the VLM-based goal inference pipeline. The system processes robot motion videos through frame extraction and timepoint selection before feeding cumulative frame sequences to a vision-language model (Gemini). A human baseline with 8 participants evaluates the same timepoints for fair comparison. Evaluation includes accuracy against ground truth, human-VLM agreement (IoU), and inference timing. Solid arrows indicate the VLM pipeline; dashed arrows show the human baseline path.}
\label{fig:pipeline}
\end{figure*}
```

**Figure 2: VLM Architecture** (`vlm_architecture_detailed.pdf`)
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{vlm_architecture_detailed.pdf}
\caption{Detailed architecture of the vision-language model for goal inference. The model processes cumulative frame sequences through a vision encoder to extract visual features, which are fused with a task-specific prompt in the language model. Multi-modal fusion and self-attention layers enable reasoning about robot intentions, producing goal classifications (A: left block/bottom drawer, B: right block/top drawer, C: uncertain).}
\label{fig:vlm_architecture}
\end{figure}
```

### For Thesis

**Chapter 3: Methodology**

Use both diagrams in sequence:
1. **Section 3.1 System Overview:** `pipeline_diagram.pdf` as full-page figure
2. **Section 3.2 VLM Architecture:** `vlm_architecture_detailed.pdf` as detailed subsection

**Recommended Caption Structure:**

**Figure 3.1: Complete System Pipeline**
> "Figure 3.1 illustrates the complete VLM-based goal inference system. The pipeline begins with 8 robot motion videos (4 ambiguous, 4 legible trajectories) captured from manipulation tasks. Videos are preprocessed through frame extraction and timepoint selection (t = 0, ..., T seconds), generating cumulative frame sequences that provide increasing temporal context. These sequences are input to a vision-language model (Gemini) along with a goal inference prompt. The VLM outputs predicted goals (A, B, or C) with confidence scores.
>
> To ensure rigorous evaluation, we implement a parallel human baseline where 8 participants view the same videos at identical timepoints through a user study interface, producing human goal predictions. Both VLM and human predictions are compared against ground truth labels from the robot's motion planner, enabling calculation of accuracy metrics and human-VLM agreement (IoU). The fair comparison methodology—evaluating the VLM only at timepoints shown to humans—ensures valid performance comparison. Final results show comparable performance: Human 66.7%, VLM 64.5%, with 74.2% agreement (IoU)."

**Figure 3.2: VLM Architecture Details**
> "Figure 3.2 details the internal architecture of the vision-language model for goal inference. The model processes cumulative frame sequences f₀, f₁, ..., fₜ, where each frame is embedded through a vision encoder to extract visual features representing robot pose, gripper configuration, and object states. These visual features are combined with a natural language task prompt ('What is the robot's goal?') in a multi-modal fusion module.
>
> The language model component uses self-attention layers to reason about temporal patterns and spatial relationships, enabling inference of the robot's intended goal. The output classifier produces one of three goal predictions: A (move to left block or close bottom drawer), B (move to right block or close top drawer), or C (uncertain/insufficient information). The model's ability to output 'uncertain' (C) is critical for handling ambiguous trajectories where motion patterns do not clearly indicate a specific goal."

---

## 📐 Design Specifications

### Style Features (ICRA-appropriate)
- **Typography:** Serif font (Times New Roman style)
- **Color scheme:** Muted academic colors (light blue, yellow, green, orange, purple)
- **Border color:** Dark gray (#424242)
- **Arrow color:** Medium gray (#616161)
- **Line weights:** 1.5 for module boxes, 1.0 for groups
- **Resolution:** 300 DPI (publication quality)

### Module Color Coding
- **Light blue-gray (#E8EAF6):** Input data
- **Light yellow (#FFF9C4):** Preprocessing/feature extraction
- **Light green (#C8E6C9):** Core model/VLM components
- **Light orange (#FFCCBC):** Output/predictions
- **Light purple (#F3E5F5):** Human baseline components

### Consistency Guidelines
- ✅ All boxes use rounded corners
- ✅ Modular sections indicated with dashed borders
- ✅ Solid arrows = main VLM pipeline
- ✅ Dashed arrows = human baseline path
- ✅ Consistent spacing and alignment
- ✅ Key metrics highlighted in yellow boxes

---

## ✏️ Customization Options

If you need to modify the diagrams, edit: `scripts/create_pipeline_diagram.py`

### Common Modifications:

**Change colors:**
```python
COLOR_INPUT = '#E8EAF6'      # Modify these hex codes
COLOR_PREPROCESS = '#FFF9C4'
COLOR_MODEL = '#C8E6C9'
# ... etc
```

**Adjust layout:**
```python
ax.set_xlim(0, 14)  # Change canvas width
ax.set_ylim(0, 10)  # Change canvas height
```

**Modify text labels:**
```python
create_box(ax, (x, y), width, height, 
           'Your Label Here', color)
```

**Add new modules:**
```python
create_module_group(ax, (x, y), width, height, 'Module Name')
```

**Regenerate after changes:**
```bash
python scripts/create_pipeline_diagram.py
```

---

## 📊 Figure Placement Recommendations

### ICRA Paper (6-8 pages)
- **Figure 1:** Pipeline diagram (full width, spans both columns)
- **Figure 2:** VLM architecture (single column, in Methods section)
- **Figure 3-5:** Results figures (accuracy, IoU, etc.)

### Extended Abstract/Workshop
- Use **only** pipeline diagram as Figure 1
- Refer to VLM architecture in text, include as supplementary material

### Thesis Chapter
- **Figure 3.1:** Pipeline diagram (full page width)
- **Figure 3.2:** VLM architecture (3/4 page width)
- Include both in Chapter 3: Methodology
- Follow with experimental results figures in Chapter 4

---

## 🎯 Key Messages to Convey

When referencing these diagrams in your text:

1. **Fair Comparison Methodology**
   - "As shown in Figure X, both VLM and human participants evaluated videos at identical timepoints, ensuring fair performance comparison."

2. **Modular System Design**
   - "The system comprises five main modules (Figure X): input processing, preprocessing, VLM inference, output generation, and evaluation against ground truth and human baseline."

3. **Multi-Modal Processing**
   - "Figure X illustrates how the VLM combines visual features from frame sequences with natural language prompts through multi-modal fusion, enabling reasoning about robot goals."

4. **End-to-End Pipeline**
   - "The complete pipeline (Figure X) processes raw robot motion videos through to goal predictions and evaluation metrics, with parallel human baseline for validation."

---

## 🔍 Review Checklist

Before submitting, verify:
- [ ] Figures render correctly at print size (usually 3.5" width for single column, 7" for double)
- [ ] All text is legible when printed
- [ ] Colors are distinguishable in grayscale (for black & white printing)
- [ ] Arrows clearly indicate data flow direction
- [ ] Module labels are descriptive and consistent with text
- [ ] Figure captions are self-contained and informative
- [ ] Resolution is 300 DPI or higher
- [ ] File format is accepted (PDF recommended for LaTeX, PNG for Word)

---

## 📁 File Locations

```
analysis_results_2/
├── pipeline_diagram.png          # Full system pipeline (PNG)
├── pipeline_diagram.pdf          # Full system pipeline (PDF, vector)
├── vlm_architecture_detailed.png # VLM architecture (PNG)
└── vlm_architecture_detailed.pdf # VLM architecture (PDF, vector)
```

**Recommendation:** Use PDF versions for LaTeX submissions (scalable vector graphics)  
**Alternative:** Use PNG versions for PowerPoint/Word documents

---

## 💡 Tips for Academic Figures

### DO:
✅ Use consistent visual language across all figures  
✅ Include scale/context when relevant  
✅ Label all axes, components, and data flows  
✅ Provide comprehensive captions  
✅ Reference figures in text before they appear  
✅ Use color sparingly and meaningfully  

### DON'T:
❌ Use decorative elements that don't convey information  
❌ Mix too many font sizes/styles  
❌ Make text smaller than 8pt  
❌ Use more than 5-6 colors  
❌ Overcrowd the figure  
❌ Leave components unlabeled  

---

## 📚 Related Figures in Your Project

You can also use these alongside your analysis figures:

**Results Section Figures:**
- `figure1_main_comparison.pdf` - Overall accuracy and IoU comparison
- `figure2_participant_iou.pdf` - Per-participant variation
- `figure3_video_iou.pdf` - Per-video agreement
- `confusion_matrices.pdf` - Error analysis
- `scatter_human_vs_vlm.pdf` - Human vs VLM performance

**Typical Figure Sequence in Paper:**
1. **Fig 1:** Pipeline diagram (this file)
2. **Fig 2:** Main results (accuracy + IoU)
3. **Fig 3:** Per-video performance
4. **Fig 4:** Confusion matrices
5. **Fig 5:** (Optional) VLM architecture details

---

## 🎓 Citation Format

If your pipeline diagram becomes widely used, here's how others might cite it:

**APA:**
> "The VLM-based goal inference pipeline (Author, 2026) consists of five main components..."

**IEEE:**
> "We adopt the pipeline architecture shown in [1], which processes robot motion videos through..."

**Referencing your own work:**
> "As illustrated in our pipeline diagram (Figure 1), the system..."

---

## ✅ Final Checklist for Submission

Before including in paper/thesis:

### Technical
- [ ] Resolution verified (300 DPI minimum)
- [ ] File size reasonable (< 5MB for PNG, < 1MB for PDF)
- [ ] Format compatible with submission system
- [ ] All fonts embedded (for PDF)

### Content
- [ ] All modules clearly labeled
- [ ] Data flow direction obvious
- [ ] Color scheme appropriate
- [ ] No spelling errors in labels
- [ ] Consistent terminology with text

### Integration
- [ ] Figure referenced in text before appearance
- [ ] Caption written and proofread
- [ ] Figure number assigned correctly
- [ ] Alt text provided (for accessibility)

---

## 🎨 Alternative Styles

If ICRA house style differs, you can modify to:

**IEEE Style:** More technical, detailed annotations  
**Nature/Science Style:** Minimal color, maximum clarity  
**Springer Style:** Grayscale-friendly, high contrast  
**ArXiv Preprint:** More color, less formal  

All styles are achievable by modifying the color scheme in the Python script.

---

**Your diagrams are publication-ready!** 🎉

Use them confidently in your ICRA submission, thesis, or presentations.
