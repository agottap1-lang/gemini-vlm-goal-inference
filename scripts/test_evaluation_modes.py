#!/usr/bin/env python3
"""
Quick test to verify the new evaluation modes work correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemini_vlm_eval.video import extract_and_cache_frames
from gemini_vlm_eval.prompt import get_instruction_prompt
from gemini_vlm_eval.schema import EvaluationResult

def test_single_frame_mode():
    """Test single_frame mode prompt and schema."""
    print("=" * 60)
    print("TEST 1: Single-Frame Mode")
    print("=" * 60)
    
    prompt = get_instruction_prompt("pick left", "pick right", 5, "test_video", mode="single_frame")
    assert "ONLY ONE image" in prompt
    assert "Use ONLY this frame" in prompt
    assert "Do NOT assume you saw earlier or later frames" in prompt
    print("✓ Single-frame prompt correct")
    
    result = EvaluationResult(
        video_id='test', video_path='test.mp4', goal_gt='A', 
        goal_A='left', goal_B='right', scene_id='s1', 
        task_family='pick', traj_type='legible', 
        t_sec=1, frame_idx=30, pA=0.7, pB=0.3, 
        cue='test', legible='legible_now',
        evaluation_mode='single_frame'
    )
    assert result.evaluation_mode == 'single_frame'
    print("✓ Single-frame schema field correct")
    print()

def test_prefix_frames_mode():
    """Test prefix_frames mode prompt and schema."""
    print("=" * 60)
    print("TEST 2: Prefix-Frames Mode")
    print("=" * 60)
    
    prompt = get_instruction_prompt("pick left", "pick right", 5, "test_video", mode="prefix_frames")
    assert "MULTIPLE images" in prompt
    assert "from t=0 to t=5" in prompt
    assert "ordered from earliest to latest" in prompt
    assert "you have observed the motion up to time t=5s" in prompt
    print("✓ Prefix-frames prompt correct")
    
    result = EvaluationResult(
        video_id='test', video_path='test.mp4', goal_gt='A',
        goal_A='left', goal_B='right', scene_id='s1',
        task_family='pick', traj_type='legible',
        t_sec=5, frame_idx=150, pA=0.8, pB=0.2,
        cue='motion toward left', legible='legible_now',
        evaluation_mode='prefix_frames'
    )
    assert result.evaluation_mode == 'prefix_frames'
    print("✓ Prefix-frames schema field correct")
    print()

def test_frame_extraction():
    """Test frame extraction and caching."""
    print("=" * 60)
    print("TEST 3: Frame Extraction and Caching")
    print("=" * 60)
    
    # Test without saving
    frames = extract_and_cache_frames(
        "videos/amb l block.mp4",
        "test_extraction",
        sample_rate_seconds=1.0,
        max_frames=3,
        save_frames=False
    )
    
    assert len(frames) == 3
    assert 0 in frames
    assert 1 in frames
    assert 2 in frames
    print(f"✓ Extracted {len(frames)} frames without saving")
    
    # Verify frame data is bytes
    for t_sec, frame_bytes in frames.items():
        assert isinstance(frame_bytes, bytes)
        assert len(frame_bytes) > 0
    print("✓ All frames are valid JPEG bytes")
    print()

def test_prefix_frame_collection():
    """Test collecting prefix frames for evaluation."""
    print("=" * 60)
    print("TEST 4: Prefix Frame Collection")
    print("=" * 60)
    
    frames_dict = extract_and_cache_frames(
        "videos/amb l block.mp4",
        "test_prefix",
        sample_rate_seconds=1.0,
        max_frames=5,
        save_frames=False
    )
    
    # Simulate prefix_frames mode at t=3
    t_sec = 3
    prefix_frames = [frames_dict[t] for t in range(0, t_sec + 1) if t in frames_dict]
    
    assert len(prefix_frames) == 4  # frames at t=0,1,2,3
    print(f"✓ Collected {len(prefix_frames)} prefix frames for t={t_sec}")
    
    # Verify all are bytes
    for frame in prefix_frames:
        assert isinstance(frame, bytes)
    print("✓ All prefix frames are valid")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING NEW EVALUATION MODES")
    print("=" * 60 + "\n")
    
    try:
        test_single_frame_mode()
        test_prefix_frames_mode()
        test_frame_extraction()
        test_prefix_frame_collection()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nThe new evaluation modes are working correctly!")
        print("\nUsage examples:")
        print("  Single-frame:   python scripts/eval_dataset.py --manifest data/manifest.jsonl --k 5 --mode single_frame --out outputs/single.jsonl")
        print("  Prefix-frames:  python scripts/eval_dataset.py --manifest data/manifest.jsonl --k 5 --mode prefix_frames --out outputs/prefix.jsonl")
        print("  With frames:    python scripts/eval_dataset.py --manifest data/manifest.jsonl --k 5 --mode prefix_frames --save-frames --out outputs/with_frames.jsonl")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
