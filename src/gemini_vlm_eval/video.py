import cv2
import logging
from typing import Generator, Tuple, Optional, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

def extract_frames(video_path: str, sample_rate_seconds: float = 1.0, max_frames: Optional[int] = None) -> Generator[Tuple[int, bytes, float], None, None]:
    """Extract frames from video, yielding (frame_number, jpeg_bytes). Samples at exact time intervals."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # default
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # Generate target timestamps: 0, sample_rate_seconds, 2*sample_rate_seconds, ...
        target_times = []
        t = 0.0
        while t < duration:
            target_times.append(t)
            t += sample_rate_seconds
            if max_frames and len(target_times) >= max_frames:
                break

        processed_frames = 0
        for target_time in target_times:
            frame_num = int(target_time * fps)
            if frame_num >= total_frames:
                break
            
            # Seek to the target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                # Convert to JPEG bytes
                success, buffer = cv2.imencode('.jpg', frame)
                if success:
                    yield frame_num, buffer.tobytes(), target_time
                    processed_frames += 1
                else:
                    logger.warning(f"Failed to encode frame {frame_num}")
            else:
                logger.warning(f"Failed to read frame {frame_num}")

        logger.info(f"Extracted {processed_frames} frames at exact {sample_rate_seconds}-second intervals")

    finally:
        cap.release()


def extract_and_cache_frames(
    video_path: str,
    video_id: str,
    sample_rate_seconds: float = 1.0,
    max_frames: Optional[int] = None,
    save_frames: bool = False,
    output_dir: Optional[Path] = None
) -> Dict[int, Dict[str, bytes]]:
    """
    Extract frames from video and optionally cache to disk.
    
    Args:
        video_path: Path to video file
        video_id: Video identifier for naming cached frames
        sample_rate_seconds: Sample interval in seconds (default 1.0)
        max_frames: Maximum number of frames to extract
        save_frames: If True, save frames to disk
        output_dir: Base directory for frames (default: outputs/frames)
        
    Returns:
        Dict mapping t_sec (int) to {"frame_idx": int, "jpeg_bytes": bytes}
    """
    frames_dict: Dict[int, Dict[str, bytes]] = {}
    cache_dir = None
    
    if save_frames:
        if output_dir is None:
            output_dir = Path("outputs/frames")
        cache_dir = output_dir / video_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Caching frames to {cache_dir}")
    
    for frame_num, jpeg_bytes, target_time in extract_frames(video_path, sample_rate_seconds, max_frames):
        t_sec = int(round(target_time))
        frames_dict[t_sec] = {"frame_idx": frame_num, "jpeg_bytes": jpeg_bytes}
        
        if save_frames and cache_dir:
            frame_path = cache_dir / f"t_{t_sec:03d}.png"
            # Decode jpeg bytes and save as PNG
            import numpy as np
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imwrite(str(frame_path), img)
    
    logger.info(f"Extracted and cached {len(frames_dict)} frames")
    return frames_dict
