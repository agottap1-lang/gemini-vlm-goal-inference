#!/usr/bin/env python3
"""Validate manifest.jsonl entries and video file existence."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemini_vlm_eval.schema import ManifestEntry


def validate_manifest(manifest_path: str, repo_root: Path) -> tuple[bool, list[str]]:
    """
    Validate manifest.jsonl.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    manifest_file = Path(manifest_path)
    
    if not manifest_file.exists():
        errors.append(f"Manifest file not found: {manifest_path}")
        return False, errors
    
    line_num = 0
    video_ids_seen = set()
    
    with open(manifest_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_num += 1
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
                continue
            
            # Validate required fields
            try:
                entry = ManifestEntry(**data)
            except Exception as e:
                errors.append(f"Line {line_num}: Schema validation failed - {e}")
                continue
            
            # Check for duplicate video_id
            if entry.video_id in video_ids_seen:
                errors.append(f"Line {line_num}: Duplicate video_id '{entry.video_id}'")
            video_ids_seen.add(entry.video_id)
            
            # Check video file exists
            video_path = repo_root / entry.video_path
            if not video_path.exists():
                errors.append(f"Line {line_num}: Video file not found: {entry.video_path} (resolved to {video_path})")
            
            # Validate goal_gt
            if entry.goal_gt not in ['A', 'B']:
                errors.append(f"Line {line_num}: goal_gt must be 'A' or 'B', got '{entry.goal_gt}'")
            
            # Check goal descriptions are non-empty
            if not entry.goal_A.strip():
                errors.append(f"Line {line_num}: goal_A is empty")
            if not entry.goal_B.strip():
                errors.append(f"Line {line_num}: goal_B is empty")
    
    if not video_ids_seen:
        errors.append("Manifest is empty - no valid entries found")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate manifest.jsonl")
    parser.add_argument("--manifest", default="data/manifest.jsonl", help="Path to manifest.jsonl")
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent
    is_valid, errors = validate_manifest(args.manifest, repo_root)
    
    if is_valid:
        # Count entries
        entry_count = 0
        with open(args.manifest, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry_count += 1
        
        print(f"✓ Manifest validation PASSED")
        print(f"  Found {entry_count} valid entries")
        sys.exit(0)
    else:
        print(f"✗ Manifest validation FAILED")
        print(f"  Errors found:")
        for error in errors:
            print(f"    - {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
