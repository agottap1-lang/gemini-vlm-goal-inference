#!/usr/bin/env python3
"""
Identify critical timepoints for user study from VLM evaluation results.

Extracts key moments:
1. First uncertain (C, 50%)
2. First high confidence (≥90%)
3. Highest flip-flop moment (if exists)
4. Final timepoint

Outputs CSV with recommended timepoints for human study.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def load_jsonl_results(jsonl_path):
    """Load evaluation results from JSONL."""
    results = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return sorted(results, key=lambda x: x['t_sec'])

def find_critical_timepoints(results, video_id):
    """
    Identify critical timepoints for user study.
    
    Returns dict with:
    - first_uncertain: First C choice
    - first_high_conf: First confidence ≥ 90%
    - flip_point: Largest flip in choice (optional)
    - final: Last timepoint
    """
    critical = {
        'video_id': video_id,
        'first_uncertain': None,
        'first_high_conf': None,
        'max_flip': None,
        'final': None,
        'duration': len(results)
    }
    
    # First uncertain
    for r in results:
        if r['choice'] == 'C' and r['confidence'] == 50:
            critical['first_uncertain'] = r['t_sec']
            break
    
    # First high confidence
    for r in results:
        if r['confidence'] >= 90:
            critical['first_high_conf'] = r['t_sec']
            break
    
    # Find largest flip (choice change with high confidence)
    prev_choice = None
    max_flip_magnitude = 0
    max_flip_time = None
    
    for i, r in enumerate(results):
        if prev_choice and r['choice'] != prev_choice and r['choice'] != 'C':
            # Calculate flip magnitude (how confident was previous + current)
            if i > 0:
                prev_conf = results[i-1]['confidence']
                curr_conf = r['confidence']
                flip_magnitude = prev_conf + curr_conf
                
                if flip_magnitude > max_flip_magnitude and curr_conf >= 80:
                    max_flip_magnitude = flip_magnitude
                    max_flip_time = r['t_sec']
        
        prev_choice = r['choice']
    
    critical['max_flip'] = max_flip_time
    
    # Final timepoint
    critical['final'] = results[-1]['t_sec']
    
    return critical

def analyze_all_videos(output_dir):
    """Analyze all videos and extract critical timepoints."""
    
    # Find all result directories
    outputs_path = Path(output_dir)
    
    all_critical = []
    
    for video_dir in outputs_path.iterdir():
        if not video_dir.is_dir() or video_dir.name == 'frames':
            continue
        
        video_id = video_dir.name
        
        # Find most recent prefix_frames run
        prefix_runs = sorted(video_dir.glob("run_*_prefix"))
        if not prefix_runs:
            print(f"⚠️  No prefix_frames runs found for {video_id}")
            continue
        
        latest_run = prefix_runs[-1]
        jsonl_path = latest_run / "results.jsonl"
        
        if not jsonl_path.exists():
            print(f"⚠️  No results.jsonl found in {latest_run}")
            continue
        
        print(f"📊 Analyzing {video_id}...")
        results = load_jsonl_results(jsonl_path)
        critical = find_critical_timepoints(results, video_id)
        
        all_critical.append(critical)
        
        # Print summary
        print(f"   Duration: {critical['duration']}s")
        print(f"   First uncertain: t={critical['first_uncertain']}s")
        print(f"   First high conf: t={critical['first_high_conf']}s")
        print(f"   Max flip: t={critical['max_flip']}s" if critical['max_flip'] else "   No significant flips")
        print(f"   Final: t={critical['final']}s")
        print()
    
    return all_critical

def export_to_csv(critical_list, output_path):
    """Export critical timepoints to CSV."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'video_id', 'duration', 'first_uncertain', 'first_high_conf', 
            'max_flip', 'final', 'recommended_timepoints'
        ])
        writer.writeheader()
        
        for c in critical_list:
            # Generate recommended timepoint list
            timepoints = []
            if c['first_uncertain'] is not None:
                timepoints.append(c['first_uncertain'])
            if c['first_high_conf'] is not None:
                timepoints.append(c['first_high_conf'])
            if c['max_flip'] is not None:
                timepoints.append(c['max_flip'])
            timepoints.append(c['final'])
            
            # Remove duplicates and sort
            timepoints = sorted(set(timepoints))
            
            c['recommended_timepoints'] = ','.join(str(t) for t in timepoints)
            writer.writerow(c)
    
    print(f"✅ Exported to {output_path}")

def main():
    output_dir = Path(__file__).parent.parent / "outputs"
    
    print("="*70)
    print("CRITICAL TIMEPOINT IDENTIFICATION FOR USER STUDY")
    print("="*70)
    print()
    
    all_critical = analyze_all_videos(output_dir)
    
    # Export to CSV
    csv_path = Path(__file__).parent.parent / "user_study_timepoints.csv"
    export_to_csv(all_critical, csv_path)
    
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total videos analyzed: {len(all_critical)}")
    print(f"Results saved to: {csv_path}")
    print()
    print("Next steps:")
    print("1. Review user_study_timepoints.csv")
    print("2. Select 3-4 timepoints per video for Part 1 (static frames)")
    print("3. Use full videos for Part 2 (pause task)")
    print()

if __name__ == "__main__":
    main()
