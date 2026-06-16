#!/usr/bin/env python3
"""
Save the VLM prompt and run one inference on legible left block video.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# API key is loaded from .env file via GeminiClient
from gemini_vlm_eval.prompt import get_instruction_prompt
from gemini_vlm_eval.client import GeminiClient
from gemini_vlm_eval.video import extract_frames
from gemini_vlm_eval.schema import ManifestEntry
import cv2

# Video configuration
VIDEO_PATH = "videos/le l block.mp4"
VIDEO_ID = "le_l_block"
GOAL_A = "pick the left block"
GOAL_B = "pick the right block"
TIMEPOINT = 5  # Run inference at t=5 seconds
MODE = "prefix_frames"  # Use prefix_frames mode to show cumulative context

# Output directory
OUTPUT_DIR = Path("analysis_results_final")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    print("=" * 90)
    print("VLM PROMPT EXTRACTION AND INFERENCE TEST")
    print("=" * 90)
    print()
    
    # Generate the prompt
    print(f"Video: {VIDEO_ID}")
    print(f"Goal A: {GOAL_A}")
    print(f"Goal B: {GOAL_B}")
    print(f"Timepoint: t={TIMEPOINT}s")
    print(f"Mode: {MODE}")
    print()
    
    prompt = get_instruction_prompt(
        goal_A=GOAL_A,
        goal_B=GOAL_B,
        t_sec=TIMEPOINT,
        video_id=VIDEO_ID,
        mode=MODE
    )
    
    # Save prompt to file
    prompt_file = OUTPUT_DIR / f"vlm_prompt_{VIDEO_ID}_t{TIMEPOINT}_{MODE}.txt"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    print("=" * 90)
    print("PROMPT SENT TO VLM:")
    print("=" * 90)
    print()
    print(prompt)
    print()
    print("=" * 90)
    print(f"✓ Prompt saved to: {prompt_file}")
    print("=" * 90)
    print()
    
    # Extract frames
    print(f"Extracting frames from {VIDEO_PATH}...")
    frame_data = list(extract_frames(VIDEO_PATH, sample_rate_seconds=1.0))
    
    if TIMEPOINT >= len(frame_data):
        print(f"Error: Video only has {len(frame_data)} frames")
        return 1
    
    print(f"✓ Extracted {len(frame_data)} frames")
    print()
    
    # Extract JPEG bytes (frames are already encoded)
    frame_bytes_list = [jpeg_bytes for _, jpeg_bytes, _ in frame_data]
    print(f"✓ Frame bytes ready: {len(frame_bytes_list)} frames")
    print()
    
    # Prepare frames based on mode
    if MODE == "prefix_frames":
        # Send all frames from 0 to TIMEPOINT
        frames_to_send = frame_bytes_list[:TIMEPOINT + 1]
        print(f"Mode: prefix_frames - Sending {len(frames_to_send)} frames (t=0 to t={TIMEPOINT})")
        print(f"Frame indices: {list(range(len(frames_to_send)))}")
    else:
        # Send only the frame at TIMEPOINT
        frames_to_send = frame_bytes_list[TIMEPOINT]
        print(f"Mode: single_frame - Sending 1 frame (t={TIMEPOINT})")
    
    print()
    
    # Create manifest entry
    manifest_entry = ManifestEntry(
        video_id=VIDEO_ID,
        video_path=VIDEO_PATH,
        goal_gt="A",
        goal_A=GOAL_A,
        goal_B=GOAL_B,
        scene_id="block_scene",
        task_family="block_pick",
        traj_type="legible",
        notes="Legible trajectory for left block pick"
    )
    
    # Run inference
    print("=" * 90)
    print("RUNNING VLM INFERENCE...")
    print("=" * 90)
    print()
    
    client = GeminiClient(model="gemini-2.5-flash")
    
    try:
        result = client.evaluate_frame(
            image_bytes=frames_to_send,
            manifest_entry=manifest_entry,
            t_sec=TIMEPOINT,
            frame_idx=TIMEPOINT,
            mode=MODE
        )
        
        print("✓ Inference completed!")
        print()
        print("=" * 90)
        print("VLM RESPONSE:")
        print("=" * 90)
        print()
        print(f"  pA (left block):  {result.pA:.3f}")
        print(f"  pB (right block): {result.pB:.3f}")
        print(f"  Choice:           {result.choice}")
        print(f"  Confidence:       {result.confidence}%")
        print(f"  Cue:              {result.cue}")
        print(f"  Legible:          {result.legible}")
        print()
        print(f"  Latency:          {result.latency_ms}ms")
        print(f"  Retries:          {result.retry_count}")
        print()
        
        # Save result
        result_file = OUTPUT_DIR / f"vlm_inference_{VIDEO_ID}_t{TIMEPOINT}_{MODE}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result.model_dump(), f, indent=2)
        
        print(f"✓ Result saved to: {result_file}")
        print()
        
        print("=" * 90)
        print("ANALYSIS:")
        print("=" * 90)
        print()
        
        ground_truth = "A"
        if result.choice == ground_truth:
            print(f"✅ CORRECT! VLM predicted '{result.choice}' (Ground truth: '{ground_truth}')")
        else:
            print(f"❌ INCORRECT! VLM predicted '{result.choice}' (Ground truth: '{ground_truth}')")
        
        print()
        print(f"Visual cue identified: \"{result.cue}\"")
        print(f"Motion is {'LEGIBLE' if result.legible == 'legible_now' else 'NOT YET LEGIBLE'} at t={TIMEPOINT}s")
        print()
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("=" * 90)
    print("TEST COMPLETE")
    print("=" * 90)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
