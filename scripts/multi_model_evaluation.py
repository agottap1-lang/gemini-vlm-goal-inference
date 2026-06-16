#!/usr/bin/env python3
"""
Multi-Model Evaluation: Run multiple Gemini models and compare results.
Identifies anomalies and calculates accuracy, agreement, and IoU for each model.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# API key is loaded from .env file via GeminiClient

from gemini_vlm_eval.client import GeminiClient
from gemini_vlm_eval.schema import ManifestEntry
from gemini_vlm_eval.video import extract_frames
import pandas as pd
import numpy as np

# Configuration
MANIFEST_FILE = Path("data/manifest.jsonl")
USER_STUDY_TIMEPOINTS_FILE = Path("user_study_timepoints.csv")
PARTICIPANT_DATA_DIR = Path(r"C:\Users\anude\OneDrive\Documents\hri-goal-inference-study\participant_data")
OUTPUTS_DIR = Path("outputs")
ANALYSIS_DIR = Path("analysis_3")

VIDEO_ID_MAPPING = {"amb_d_drawer_cclose": "amb_d_drawer_close"}

# Models to evaluate (from older to newer, including experimental)
MODELS_TO_TEST = [
    "gemini-2.5-flash",         # Current baseline (Flash)
    "gemini-2.5-pro",           # Pro model
    "gemini-3-pro-preview",     # Gemini 3 Pro preview
    "gemini-3.1-pro-preview",   # Gemini 3.1 Pro (LATEST!)
    "gemini-pro-latest"         # Production "latest" model
]

# Ground truth
GROUND_TRUTH = {
    'amb_d_drawer_close': 'A',
    'amb_l_block': 'A',
    'amb_r_block': 'B',
    'amb_to_drawer_close': 'B',
    'le_d_drawer_close': 'A',
    'le_l_block': 'A',
    'le_r_block': 'B',
    'le_t_drawer_close': 'B',
}

TRAJECTORY_TYPES = {
    'amb_d_drawer_close': 'ambiguous',
    'amb_l_block': 'ambiguous',
    'amb_r_block': 'ambiguous',
    'amb_to_drawer_close': 'ambiguous',
    'le_d_drawer_close': 'legible',
    'le_l_block': 'legible',
    'le_r_block': 'legible',
    'le_t_drawer_close': 'legible',
}

def load_user_study_timepoints():
    """Load timepoints where humans were evaluated."""
    df = pd.read_csv(USER_STUDY_TIMEPOINTS_FILE)
    timepoints_map = {}
    for _, row in df.iterrows():
        video_id = row['video_id']
        timepoints_str = row['recommended_timepoints']
        timepoints = [int(t.strip()) for t in timepoints_str.split(',')]
        timepoints_map[video_id] = timepoints
    return timepoints_map

def load_manifest():
    """Load manifest with video metadata."""
    manifest = []
    with open(MANIFEST_FILE, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                manifest.append(ManifestEntry(**data))
    return manifest

def load_human_data():
    """Load human participant data."""
    human_data = []
    
    for json_file in sorted(PARTICIPANT_DATA_DIR.glob("*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for obs in data:
                    if isinstance(obs, dict):
                        vid = obs.get('video_id')
                        vid = VIDEO_ID_MAPPING.get(vid, vid)
                        
                        if obs.get('phase') == 'video_stop':
                            human_data.append({
                                'participant_id': obs.get('participant_id'),
                                'video_id': vid,
                                'decision_time_sec': obs.get('decision_time_sec'),
                                'choice': obs.get('choice', '').upper(),
                                'confidence': obs.get('confidence_0_10')
                            })
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return pd.DataFrame(human_data)

def evaluate_video_at_timepoints(video_entry, timepoints, model_name):
    """Evaluate a video at specific timepoints using given model."""
    print(f"    {video_entry.video_id}: ", end='', flush=True)
    
    # Extract frames
    try:
        frame_data = list(extract_frames(video_entry.video_path, sample_rate_seconds=1.0))
        frame_bytes_dict = {int(t): jpeg_bytes for _, jpeg_bytes, t in frame_data}
    except Exception as e:
        print(f"ERROR - Could not extract frames: {e}")
        return []
    
    # Initialize client
    client = GeminiClient(model=model_name)
    
    results = []
    success_count = 0
    
    for t in timepoints:
        if t not in frame_bytes_dict:
            print('⚠', end='', flush=True)
            continue
        
        # Prepare frames (prefix mode: all from 0 to t)
        frames_to_send = [frame_bytes_dict[i] for i in range(t + 1) if i in frame_bytes_dict]
        
        try:
            result = client.evaluate_frame(
                image_bytes=frames_to_send,
                manifest_entry=video_entry,
                t_sec=t,
                frame_idx=t,
                mode="prefix_frames"
            )
            results.append(result)
            success_count += 1
            print('✓', end='', flush=True)
        except Exception as e:
            print('✗', end='', flush=True)
            print(f"\n      Error at t={t}s: {e}")
    
    print(f" ({success_count}/{len(timepoints)})")
    return results

def calculate_metrics(vlm_results, human_df):
    """Calculate accuracy, agreement, IoU, and temporal stability metrics."""
    
    # VLM Accuracy
    vlm_correct = sum(1 for r in vlm_results if r.choice == r.goal_gt and r.choice in ['A', 'B'])
    vlm_total = len([r for r in vlm_results if r.choice in ['A', 'B']])
    vlm_accuracy = (vlm_correct / vlm_total * 100) if vlm_total > 0 else 0
    
    # Human Accuracy
    human_correct = 0
    human_total = 0
    for _, row in human_df.iterrows():
        if row['choice'] in ['A', 'B']:
            human_total += 1
            if row['choice'] == GROUND_TRUTH.get(row['video_id']):
                human_correct += 1
    human_accuracy = (human_correct / human_total * 100) if human_total > 0 else 0
    
    # IoU: Intersection over Union of choices
    # For each video/timepoint, check if VLM and majority human agree
    iou_matches = 0
    iou_total = 0
    
    for result in vlm_results:
        if result.choice not in ['A', 'B']:
            continue
        
        # Get human choices for this video
        human_choices = human_df[
            (human_df['video_id'] == result.video_id) & 
            (human_df['choice'].isin(['A', 'B']))
        ]['choice'].values
        
        if len(human_choices) > 0:
            # Majority human choice
            from collections import Counter
            human_majority = Counter(human_choices).most_common(1)[0][0]
            
            iou_total += 1
            if result.choice == human_majority:
                iou_matches += 1
    
    iou = (iou_matches / iou_total * 100) if iou_total > 0 else 0
    
    # ===== NEW TEMPORAL STABILITY METRICS =====
    # Group results by video
    video_results = {}
    for result in vlm_results:
        if result.video_id not in video_results:
            video_results[result.video_id] = []
        video_results[result.video_id].append(result)
    
    # Sort by time within each video
    for video_id in video_results:
        video_results[video_id] = sorted(video_results[video_id], key=lambda x: x.t_sec)
    
    # Track temporal stability metrics
    total_choice_flips = 0
    videos_with_flips = 0
    early_correct_count = 0  # Correct at first timepoint
    stable_correct_count = 0  # Correct early AND stayed correct (no flips)
    videos_evaluated = len(video_results)
    
    for video_id, results in video_results.items():
        gt = GROUND_TRUTH.get(video_id)
        if not gt:
            continue
        
        # Count choice flips for this video
        prev_choice = None
        flipped = False
        for result in results:
            if result.choice in ['A', 'B']:
                if prev_choice and prev_choice != result.choice:
                    total_choice_flips += 1
                    flipped = True
                prev_choice = result.choice
        
        if flipped:
            videos_with_flips += 1
        
        # Check early correctness (first decided choice)
        first_decided = next((r for r in results if r.choice in ['A', 'B']), None)
        if first_decided and first_decided.choice == gt:
            early_correct_count += 1
            
            # Check if stayed correct (no flips to wrong answer)
            stayed_correct = True
            for result in results:
                if result.choice in ['A', 'B'] and result.choice != gt:
                    stayed_correct = False
                    break
            
            if stayed_correct:
                stable_correct_count += 1
    
    # Calculate percentages
    no_flip_rate = ((videos_evaluated - videos_with_flips) / videos_evaluated * 100) if videos_evaluated > 0 else 0
    early_correct_rate = (early_correct_count / videos_evaluated * 100) if videos_evaluated > 0 else 0
    stable_correct_rate = (stable_correct_count / videos_evaluated * 100) if videos_evaluated > 0 else 0
    
    return {
        'vlm_accuracy': vlm_accuracy,
        'vlm_correct': vlm_correct,
        'vlm_total': vlm_total,
        'human_accuracy': human_accuracy,
        'human_correct': human_correct,
        'human_total': human_total,
        'iou': iou,
        'iou_matches': iou_matches,
        'iou_total': iou_total,
        # New temporal stability metrics
        'total_choice_flips': total_choice_flips,
        'videos_with_flips': videos_with_flips,
        'no_flip_rate': no_flip_rate,
        'early_correct_count': early_correct_count,
        'early_correct_rate': early_correct_rate,
        'stable_correct_count': stable_correct_count,
        'stable_correct_rate': stable_correct_rate,
        'videos_evaluated': videos_evaluated
    }

def detect_anomalies(model_results, model_name):
    """Detect anomalies in model predictions."""
    anomalies = []
    
    # Group by video
    video_results = {}
    for result in model_results:
        if result.video_id not in video_results:
            video_results[result.video_id] = []
        video_results[result.video_id].append(result)
    
    for video_id, results in video_results.items():
        results = sorted(results, key=lambda x: x.t_sec)
        
        # Anomaly 1: Choice flips late in the video (after being confident)
        high_conf_choice = None
        for i, result in enumerate(results):
            if result.confidence >= 80 and result.choice in ['A', 'B']:
                if high_conf_choice is None:
                    high_conf_choice = result.choice
                elif result.choice != high_conf_choice:
                    anomalies.append({
                        'type': 'choice_flip_after_high_confidence',
                        'video_id': video_id,
                        'model': model_name,
                        'timepoint': result.t_sec,
                        'previous_choice': high_conf_choice,
                        'new_choice': result.choice,
                        'confidence': result.confidence,
                        'description': f"Changed from {high_conf_choice} (high conf) to {result.choice} at t={result.t_sec}s"
                    })
        
        # Anomaly 2: Very low confidence in legible trajectories at late timepoints
        if TRAJECTORY_TYPES.get(video_id) == 'legible':
            late_results = [r for r in results if r.t_sec >= 5]
            for result in late_results:
                if result.confidence < 60:
                    anomalies.append({
                        'type': 'low_confidence_in_legible',
                        'video_id': video_id,
                        'model': model_name,
                        'timepoint': result.t_sec,
                        'confidence': result.confidence,
                        'choice': result.choice,
                        'description': f"Low confidence ({result.confidence}%) in legible video at t={result.t_sec}s"
                    })
        
        # Anomaly 3: Wrong prediction when confidence is high
        for result in results:
            if result.confidence >= 85 and result.choice != result.goal_gt and result.choice in ['A', 'B']:
                anomalies.append({
                    'type': 'high_confidence_wrong',
                    'video_id': video_id,
                    'model': model_name,
                    'timepoint': result.t_sec,
                    'predicted': result.choice,
                    'ground_truth': result.goal_gt,
                    'confidence': result.confidence,
                    'description': f"Predicted {result.choice} with {result.confidence}% confidence, but correct is {result.goal_gt}"
                })
        
        # Anomaly 4: Uncertainty in final timepoints
        if len(results) > 0:
            final_result = results[-1]
            if final_result.choice == 'C':
                anomalies.append({
                    'type': 'final_uncertain',
                    'video_id': video_id,
                    'model': model_name,
                    'timepoint': final_result.t_sec,
                    'description': f"Still uncertain (C) at final timepoint t={final_result.t_sec}s"
                })
    
    return anomalies

def run_multi_model_evaluation():
    """Main evaluation function."""
    
    print("=" * 90)
    print("MULTI-MODEL VLM EVALUATION")
    print("=" * 90)
    print()
    
    # Setup
    ANALYSIS_DIR.mkdir(exist_ok=True)
    
    # Load data
    print("Loading configuration...")
    manifest = load_manifest()
    timepoints_map = load_user_study_timepoints()
    human_df = load_human_data()
    print(f"✓ Loaded {len(manifest)} videos")
    print(f"✓ Loaded {len(human_df)} human observations")
    print()
    
    # Check available models
    print("Checking available models...")
    try:
        from google import genai
        genai_client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
        available_models = []
        for model in genai_client.models.list():
            available_models.append(model.name.replace('models/', ''))
        
        # Filter to models that exist
        models_to_run = []
        for model in MODELS_TO_TEST:
            if model in available_models:
                models_to_run.append(model)
                print(f"  ✓ {model}")
            else:
                print(f"  ✗ {model} (not available)")
        print()
    except Exception as e:
        print(f"Warning: Could not check models: {e}")
        models_to_run = MODELS_TO_TEST
    
    if not models_to_run:
        print("❌ No models available!")
        return
    
    # Results storage
    all_results = {}
    all_metrics = {}
    all_anomalies = {}
    
    # Run evaluation for each model
    for model_name in models_to_run:
        print("=" * 90)
        print(f"EVALUATING MODEL: {model_name}")
        print("=" * 90)
        print()
        
        model_results = []
        start_time = time.time()
        
        for video_entry in manifest:
            video_id = video_entry.video_id
            timepoints = timepoints_map.get(video_id, [])
            
            if not timepoints:
                print(f"  ⚠️  {video_id}: No timepoints, skipping")
                continue
            
            results = evaluate_video_at_timepoints(video_entry, timepoints, model_name)
            model_results.extend(results)
        
        elapsed = time.time() - start_time
        
        # Save results
        model_file = ANALYSIS_DIR / f"results_{model_name.replace('.', '_').replace('-', '_')}.jsonl"
        with open(model_file, 'w') as f:
            for result in model_results:
                f.write(json.dumps(result.model_dump()) + '\n')
        
        print()
        print(f"✓ Completed in {elapsed:.1f}s - Saved to {model_file.name}")
        print()
        
        # Calculate metrics
        metrics = calculate_metrics(model_results, human_df)
        metrics['model'] = model_name
        metrics['total_predictions'] = len(model_results)
        metrics['evaluation_time_sec'] = elapsed
        
        all_results[model_name] = model_results
        all_metrics[model_name] = metrics
        
        # Detect anomalies
        anomalies = detect_anomalies(model_results, model_name)
        all_anomalies[model_name] = anomalies
        
        print(f"📊 Metrics for {model_name}:")
        print(f"  VLM Accuracy:     {metrics['vlm_accuracy']:.2f}% ({metrics['vlm_correct']}/{metrics['vlm_total']})")
        print(f"  Human Accuracy:   {metrics['human_accuracy']:.2f}%")
        print(f"  IoU (Agreement):  {metrics['iou']:.2f}% ({metrics['iou_matches']}/{metrics['iou_total']})")
        print(f"  Anomalies Found:  {len(anomalies)}")
        print()
    
    # Generate comparison report
    print("=" * 90)
    print("GENERATING COMPARISON REPORT")
    print("=" * 90)
    print()
    
    # Overall comparison table
    comparison_df = pd.DataFrame([
        {
            'Model': m,
            'VLM Accuracy (%)': all_metrics[m]['vlm_accuracy'],
            'IoU (%)': all_metrics[m]['iou'],
            'No-Flip Rate (%)': all_metrics[m]['no_flip_rate'],
            'Early Correct (%)': all_metrics[m]['early_correct_rate'],
            'Stable Correct (%)': all_metrics[m]['stable_correct_rate'],
            'Choice Flips': all_metrics[m]['total_choice_flips'],
            'Predictions': all_metrics[m]['total_predictions'],
            'Anomalies': len(all_anomalies[m]),
            'Time (s)': all_metrics[m]['evaluation_time_sec']
        }
        for m in models_to_run if m in all_metrics
    ])
    
    comparison_df = comparison_df.sort_values('VLM Accuracy (%)', ascending=False)
    
    print("MODEL COMPARISON (sorted by accuracy):")
    print()
    print(comparison_df.to_string(index=False))
    print()
    
    comparison_df.to_csv(ANALYSIS_DIR / "model_comparison.csv", index=False)
    print(f"✓ Saved to model_comparison.csv")
    print()
    
    # ===== BEST MODEL SELECTION BASED ON USER CRITERIA =====
    print("=" * 90)
    print("BEST MODEL SELECTION (USER CRITERIA)")
    print("=" * 90)
    print()
    print("Selection Criteria:")
    print("  1. Maximum IoU with humans (agreement)")
    print("  2. Maximum correct answers (accuracy)")
    print("  3. Minimum choice flips (temporal consistency)")
    print("  4. Early correct & stable (robust predictions)")
    print()
    
    # Get human accuracy from first model's metrics (same for all)
    human_accuracy = list(all_metrics.values())[0]['human_accuracy'] if all_metrics else 0
    
    # Create composite score
    comparison_df['composite_score'] = (
        comparison_df['VLM Accuracy (%)'] * 0.35 +
        comparison_df['IoU (%)'] * 0.25 +
        comparison_df['No-Flip Rate (%)'] * 0.20 +
        comparison_df['Stable Correct (%)'] * 0.20
    )
    
    best_overall = comparison_df.iloc[0]
    best_composite = comparison_df.nlargest(1, 'composite_score').iloc[0]
    
    print("🏆 RECOMMENDED MODEL:")
    print(f"   Model: {best_composite['Model']}")
    print(f"   Accuracy: {best_composite['VLM Accuracy (%)']:.2f}% (Human: {human_accuracy:.2f}%)")
    print(f"   IoU: {best_composite['IoU (%)']:.2f}%")
    print(f"   No-Flip Rate: {best_composite['No-Flip Rate (%)']:.2f}%")
    print(f"   Early Correct: {best_composite['Early Correct (%)']:.2f}%")
    print(f"   Stable Correct: {best_composite['Stable Correct (%)']:.2f}%")
    print(f"   Choice Flips: {int(best_composite['Choice Flips'])} total")
    print(f"   Composite Score: {best_composite['composite_score']:.2f}/100")
    print()
    
    print("Top 3 Models by Criteria:")
    print()
    print(f"  Highest Accuracy:      {comparison_df.nlargest(1, 'VLM Accuracy (%)').iloc[0]['Model']} ({comparison_df.nlargest(1, 'VLM Accuracy (%)').iloc[0]['VLM Accuracy (%)']:.2f}%)")
    print(f"  Highest IoU:           {comparison_df.nlargest(1, 'IoU (%)').iloc[0]['Model']} ({comparison_df.nlargest(1, 'IoU (%)').iloc[0]['IoU (%)']:.2f}%)")
    print(f"  Fewest Choice Flips:   {comparison_df.nsmallest(1, 'Choice Flips').iloc[0]['Model']} ({int(comparison_df.nsmallest(1, 'Choice Flips').iloc[0]['Choice Flips'])} flips)")
    print(f"  Most Stable Correct:   {comparison_df.nlargest(1, 'Stable Correct (%)').iloc[0]['Model']} ({comparison_df.nlargest(1, 'Stable Correct (%)').iloc[0]['Stable Correct (%)']:.2f}%)")
    print()
    
    # Per-video comparison
    print("=" * 90)
    print("PER-VIDEO ACCURACY COMPARISON")
    print("=" * 90)
    print()
    
    video_comparison = []
    for video_id in sorted(GROUND_TRUTH.keys()):
        row = {'video_id': video_id, 'ground_truth': GROUND_TRUTH[video_id]}
        
        for model_name in models_to_run:
            if model_name not in all_results:
                continue
            
            vid_results = [r for r in all_results[model_name] if r.video_id == video_id and r.choice in ['A', 'B']]
            correct = sum(1 for r in vid_results if r.choice == r.goal_gt)
            total = len(vid_results)
            accuracy = (correct / total * 100) if total > 0 else 0
            
            row[f'{model_name}_acc'] = accuracy
            row[f'{model_name}_n'] = f"{correct}/{total}"
        
        video_comparison.append(row)
    
    video_df = pd.DataFrame(video_comparison)
    print(video_df.to_string(index=False))
    print()
    
    video_df.to_csv(ANALYSIS_DIR / "video_comparison.csv", index=False)
    print(f"✓ Saved to video_comparison.csv")
    print()
    
    # Anomaly report
    print("=" * 90)
    print("ANOMALY ANALYSIS")
    print("=" * 90)
    print()
    
    anomaly_summary = []
    for model_name in models_to_run:
        if model_name not in all_anomalies:
            continue
        
        anomalies = all_anomalies[model_name]
        
        # Count by type
        from collections import Counter
        type_counts = Counter(a['type'] for a in anomalies)
        
        print(f"{model_name}:")
        print(f"  Total anomalies: {len(anomalies)}")
        for atype, count in type_counts.most_common():
            print(f"    - {atype}: {count}")
        print()
        
        anomaly_summary.extend(anomalies)
    
    # Save all anomalies
    anomaly_df = pd.DataFrame(anomaly_summary)
    if len(anomaly_df) > 0:
        anomaly_df.to_csv(ANALYSIS_DIR / "anomalies.csv", index=False)
        print(f"✓ Saved detailed anomalies to anomalies.csv")
        print()
    
    # Detailed anomaly explanations
    print("=" * 90)
    print("DETAILED ANOMALY EXPLANATIONS")
    print("=" * 90)
    print()
    
    with open(ANALYSIS_DIR / "anomaly_explanations.txt", 'w') as f:
        for model_name in models_to_run:
            if model_name not in all_anomalies:
                continue
            
            anomalies = all_anomalies[model_name]
            if not anomalies:
                continue
            
            f.write(f"{'=' * 80}\n")
            f.write(f"MODEL: {model_name}\n")
            f.write(f"{'=' * 80}\n\n")
            
            print(f"### {model_name} ({len(anomalies)} anomalies)")
            print()
            
            for i, anomaly in enumerate(anomalies[:10], 1):  # Show first 10
                line = f"{i}. {anomaly['description']}\n"
                f.write(line)
                print(f"  {line.strip()}")
                
                # Explanation
                explanation = ""
                if anomaly['type'] == 'choice_flip_after_high_confidence':
                    explanation = "   → Likely due to ambiguous motion or conflicting visual cues as motion progresses."
                elif anomaly['type'] == 'low_confidence_in_legible':
                    explanation = "   → Model may not effectively integrate temporal information despite prefix_frames mode."
                elif anomaly['type'] == 'high_confidence_wrong':
                    explanation = "   → Model misinterpreted visual cues. Possible training biases or limited understanding of this task domain."
                elif anomaly['type'] == 'final_uncertain':
                    explanation = "   → Video may be truly ambiguous, or model failed to accumulate enough evidence."
                
                f.write(explanation + "\n\n")
                print(explanation)
            
            if len(anomalies) > 10:
                remaining = len(anomalies) - 10
                f.write(f"\n... and {remaining} more anomalies\n\n")
                print(f"\n  ... and {remaining} more anomalies\n")
            
            print()
    
    print(f"✓ Saved detailed explanations to anomaly_explanations.txt")
    print()
    
    # Summary
    summary = {
        'evaluation_date': datetime.now().isoformat(),
        'models_evaluated': models_to_run,
        'videos_evaluated': len(manifest),
        'metrics': {m: all_metrics[m] for m in models_to_run if m in all_metrics},
        'best_model': comparison_df.iloc[0]['Model'] if len(comparison_df) > 0 else None,
        'best_accuracy': comparison_df.iloc[0]['VLM Accuracy (%)'] if len(comparison_df) > 0 else None
    }
    
    with open(ANALYSIS_DIR / "evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("=" * 90)
    print("MULTI-MODEL EVALUATION COMPLETE")
    print("=" * 90)
    print()
    print(f"📁 All results saved to: {ANALYSIS_DIR}")
    print()
    print("Generated files:")
    print("  - model_comparison.csv       : Overall model comparison")
    print("  - video_comparison.csv       : Per-video accuracy breakdown")
    print("  - anomalies.csv              : All detected anomalies")
    print("  - anomaly_explanations.txt   : Detailed explanations")
    print("  - evaluation_summary.json    : Complete summary")
    print("  - results_*.jsonl            : Raw results for each model")
    print()
    
    if comparison_df is not None and len(comparison_df) > 0:
        best = comparison_df.iloc[0]
        print(f"🏆 Best Model: {best['Model']}")
        print(f"   Accuracy: {best['VLM Accuracy (%)']:.2f}%")
        print(f"   IoU: {best['IoU (%)']:.2f}%")
    
    return summary

if __name__ == "__main__":
    try:
        summary = run_multi_model_evaluation()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
