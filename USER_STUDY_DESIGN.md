# User Study Design: VLM vs Human Legibility Evaluation

## 📋 Study Overview

**Goal:** Compare VLM predictions to human judgments on robot motion legibility

**Two-Part Design:**
1. **Part 1 (Static Frames):** Show prefix sequences at critical timepoints → measure goal inference
2. **Part 2 (Pause Task):** Show full video → identify when goal becomes clear

---

## 🎯 Part 1: Static Frame Trials (Critical Timepoints)

### Selected Timepoints Per Video

| Video ID | Type | Timepoints to Test | Rationale |
|----------|------|-------------------|-----------|
| **amb_d_drawer_close** | Ambiguous | t=0s, t=5s, t=9s, t=19s | Uncertain start → First confidence (5s) → Flip (9s) → Final |
| **amb_l_block** | Ambiguous | t=0s, t=6s, t=8s, t=12s | Uncertain start → First confidence (6s) → Flip (8s) → Final |
| **amb_r_block** | Ambiguous | t=0s, t=1s, t=7s, t=14s | Uncertain → Early inference (1s) → Major flip (7s) → Final |
| **amb_to_drawer_close** | Ambiguous | t=0s, t=2s, t=10s, t=15s | Uncertain → Early inference (2s) → Flip (10s) → Final |
| **le_d_drawer_close** | Legible | t=0s, t=3s, t=8s, t=11s | Uncertain start → Legibility emerges (3s) → Flip (8s) → Final |
| **le_l_block** | Legible | t=0s, t=4s, t=14s | Uncertain start → Legibility emerges (4s) → Final |
| **le_r_block** | Legible | t=0s, t=4s, t=5s, t=11s | Uncertain → Wrong inference (4s) → Correction (5s) → Final |
| **le_t_drawer_close** | Legible | t=0s, t=1s, t=5s, t=8s | Uncertain → Early legibility (1s) → Flip (5s) → Final |

**Total Trials:** 31 timepoints across 8 videos

---

## 📸 Part 1: What Participants See

### Trial Structure (Example: le_r_block at t=5s)

```
┌─────────────────────────────────────────────────────────┐
│                TRIAL 15 of 31                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Frames 0 through 5 seconds:                           │
│                                                         │
│  [img t=0] [img t=1] [img t=2]                        │
│  [img t=3] [img t=4] [img t=5]                        │
│                                                         │
│  Study these frames carefully before answering.        │
│                                                         │
└─────────────────────────────────────────────────────────┘

QUESTIONS:

1. Based on frames 0-5, which goal is the robot pursuing?
   
   Goal A: pick the left block
   Goal B: pick the right block
   
   Probability of Goal A: [___]% (0-100)
   Probability of Goal B: [___]% (0-100)
   (Must sum to 100%)

2. What visual feature supports your answer?
   _________________________________________________

3. Can you confidently infer the robot's goal at this moment?
   ○ Yes (I'm confident now)
   ○ No (Still uncertain)

4. How confident are you in your prediction?
   ○ 50% ○ 60% ○ 70% ○ 80% ○ 90% ○ 95% ○ 99% ○ 100%
```

---

## 🎥 Part 2: Full Video Pause Task

### Purpose
**Validate your ambiguous vs legible labels** by measuring when humans first become confident.

### Hypothesis
- **Legible trajectories:** Humans pause early (confident quickly)
- **Ambiguous trajectories:** Humans pause late or never reach high confidence

### Task Instructions

```
You will now watch complete robot motion videos.

Your task:
1. Press PLAY to start the video
2. Watch the robot's motion carefully
3. Press PAUSE as soon as you can confidently predict the goal
   (When you're ≥80% sure which goal the robot is pursuing)
4. If you reach the end without being confident, that's okay

Each video will play automatically. You can replay once if needed.
```

### Data Collected

For each video:
- **Pause time** (seconds when user pressed pause)
- **Goal prediction** (A or B)
- **Confidence** (50-100%)
- **Final validation:** "Was this trajectory easy or hard to judge?"
  - ○ Easy (legible)
  - ○ Hard (ambiguous)

### Expected Outcomes

| Video Type | Expected Pause Time | Expected Difficulty |
|------------|---------------------|---------------------|
| **Legible** | Early (1-5s) | "Easy" |
| **Ambiguous** | Late (>8s) or no pause | "Hard" |

---

## 📊 Comparison Metrics

### Part 1: Moment-by-Moment Agreement

| Metric | Calculation |
|--------|-------------|
| **Goal Agreement** | % trials where human choice matches VLM choice |
| **Confidence Correlation** | Pearson correlation between VLM and human confidence |
| **Cue Similarity** | Semantic similarity of visual cues (NLP) |
| **Flip Detection** | Do humans also flip at VLM flip points? |

### Part 2: Legibility Validation

| Metric | Calculation |
|--------|-------------|
| **Pause Time** | Median pause time per trajectory type |
| **Label Accuracy** | % humans label legible/ambiguous correctly |
| **VLM vs Human Timing** | Compare VLM "first_high_conf" to human pause time |

---

## 🎯 Study Implementation Checklist

### Before Study

- [ ] Extract all frames for selected timepoints
  - Use: `outputs/{video_id}/run_*_prefix/frames/{video_id}/t_XXX.png`
- [ ] Create trial presentation order (randomized per participant)
- [ ] Prepare full videos for Part 2 (8 videos)
- [ ] Set up data collection form/software

### During Study

- [ ] Show instructions + practice trial (use extra video)
- [ ] Run Part 1: 31 static frame trials (~15 minutes)
- [ ] Short break (2 minutes)
- [ ] Run Part 2: 8 pause videos (~10 minutes)
- [ ] Post-study questionnaire (5 minutes)

**Total time:** ~30-35 minutes per participant

### After Study

- [ ] Export VLM predictions for same timepoints
- [ ] Calculate agreement metrics
- [ ] Analyze pause times vs trajectory type
- [ ] Statistical tests (t-test for pause times, Cohen's kappa for agreement)

---

## 📁 Files Needed for Study

### Part 1: Frame Images
```
For each timepoint in user_study_timepoints.csv:
  Copy: outputs/{video_id}/run_*_prefix/frames/{video_id}/t_{0..t}.png
  To: user_study_frames/{video_id}_t{t}/
```

### Part 2: Videos
```
Copy: videos/{video_name}.mp4
To: user_study_videos/
```

### VLM Comparison Data
```
Extract from: outputs/{video_id}/run_*_prefix/results.jsonl
Filter to timepoints in user_study_timepoints.csv
```

---

## 💡 Key Design Decisions

### Why These Timepoints?
1. **t=0 (Always):** Baseline uncertainty for all videos
2. **First high confidence:** When VLM first commits (tests early legibility)
3. **Flip point:** Tests if humans also experience uncertainty/revision
4. **Final:** Tests convergence to final answer

### Why Prefix Sequences (Not Single Frames)?
- ✅ Matches VLM information access (cumulative evidence)
- ✅ Tests temporal reasoning (not just instantaneous perception)
- ✅ Reveals if humans also flip-flop with more evidence

### Why Full Video Task?
- ✅ Validates your ground truth labels (ambiguous vs legible)
- ✅ Natural task for humans (watching motion)
- ✅ Measures when confidence emerges naturally

---

## 📈 Expected Findings

### If VLM ≈ Human:
- High agreement on goal choice (>80%)
- Similar confidence levels (r > 0.6)
- Similar pause times for legibility emergence

### If VLM ≠ Human:
- **VLM over-confident:** Higher VLM confidence than humans
- **VLM flip-flops:** VLM changes mind when humans don't
- **Timing mismatch:** VLM identifies legibility earlier/later than humans

---

## 🔧 Implementation Help Available

Would you like me to create:
1. ✅ **Frame extraction script** to copy images to study folders?
2. ✅ **VLM comparison data exporter** (extract VLM predictions for exact timepoints)?
3. ✅ **Data analysis script** to compute metrics after study?

Let me know which you need!
