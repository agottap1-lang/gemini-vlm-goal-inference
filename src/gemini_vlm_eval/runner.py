import json
import logging
from pathlib import Path
from typing import Optional
from .client import GeminiClient
from .video import extract_frames
from .schema import EvaluationResult

logger = logging.getLogger(__name__)

def evaluate_image(image_path: str, client: GeminiClient, output_path: str):
    """Evaluate a single image."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    result = client.evaluate_frame(image_bytes, image_path, 0.0, 0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(result.json() + '\n')

    logger.info(f"Image evaluation complete. Result saved to {output_path}")

def evaluate_video(video_path: str, client: GeminiClient, output_path: str = None,
                  sample_rate_seconds: float = 1.0, max_frames: Optional[int] = None):
    """Evaluate frames from a video."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30
    cap.release()

    if output_path is None:
        # Generate output path based on video name
        video_name = Path(video_path).stem
        base_path = Path(f"outputs/{video_name}_results.jsonl")
        output_path = base_path
        counter = 1
        while output_path.exists():
            output_path = base_path.parent / f"{base_path.stem}_{counter}.jsonl"
            counter += 1
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for frame_num, frame_bytes, t_sec in extract_frames(video_path, sample_rate_seconds, max_frames):
            try:
                result = client.evaluate_frame(frame_bytes, video_path, t_sec, frame_num)
                f.write(result.json() + '\n')
                f.flush()  # Ensure immediate write
            except Exception as e:
                logger.error(f"Failed to evaluate frame {frame_num}: {e}")
                # Continue with other frames

    logger.info(f"Video evaluation complete. Results saved to {output_path}")
    return str(output_path)

def evaluate_folder(folder_path: str, client: GeminiClient,
                   sample_rate_seconds: float = 1.0, max_frames: Optional[int] = None):
    """Evaluate all videos in a folder."""
    from .video import extract_frames
    import cv2

    folder_path = Path(folder_path)

    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]

    for video_file in video_files:
        logger.info(f"Evaluating {video_file}")
        try:
            evaluate_video(str(video_file), client, None, sample_rate_seconds, max_frames)
        except Exception as e:
            logger.error(f"Failed to evaluate {video_file}: {e}")
            continue

    logger.info(f"Folder evaluation complete. Results saved to outputs/ directory")