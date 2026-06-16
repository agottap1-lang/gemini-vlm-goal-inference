import os
import json
import logging
import time
from typing import Optional, Union, List
from dotenv import load_dotenv
from google import genai
from google.genai import types
from .schema import EvaluationResult, ManifestEntry
from .prompt import get_instruction_prompt

load_dotenv()

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set")
        self.client = genai.Client(api_key=self.api_key)
        self.model = model

    def evaluate_frame(
        self, 
        image_bytes: Union[bytes, List[bytes]], 
        manifest_entry: ManifestEntry, 
        t_sec: int, 
        frame_idx: int,
        mode: str = "single_frame"
    ) -> EvaluationResult:
        """
        Evaluate frame(s) using Gemini VLM with metadata tracking.
        
        Args:
            image_bytes: Single image bytes or list of image bytes (for prefix_frames mode)
            manifest_entry: Manifest entry with video metadata
            t_sec: Current timestamp in seconds
            frame_idx: Frame index
            mode: Evaluation mode - "single_frame" or "prefix_frames"
        
        Returns:
            EvaluationResult with probabilities and metadata
        """
        
        retry_count = 0
        last_exception = None
        
        # Try up to 3 times
        for attempt in range(3):
            try:
                prompt = get_instruction_prompt(
                    manifest_entry.goal_A,
                    manifest_entry.goal_B,
                    t_sec,
                    manifest_entry.video_id,
                    mode=mode
                )
                
                # Prepare content based on mode
                if mode == "prefix_frames" and isinstance(image_bytes, list):
                    # Multiple images: send all frames in chronological order
                    content = [prompt]
                    for img_bytes in image_bytes:
                        content.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
                else:
                    # Single image
                    img_data = image_bytes if isinstance(image_bytes, bytes) else image_bytes[0]
                    content = [
                        prompt,
                        types.Part.from_bytes(data=img_data, mime_type="image/jpeg")
                    ]
                
                # Track latency
                start_time = time.time()
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=content
                )
                latency_ms = int((time.time() - start_time) * 1000)
                
                # Extract IDs from response metadata if available
                request_id = getattr(response, 'request_id', None)
                response_id = getattr(response, 'id', None)
                http_status = 200  # Success
                
                # Parse the response
                response_text = response.text.strip()
                logger.debug(f"Gemini response for {manifest_entry.video_id} at t={t_sec}s: {response_text}")

                # Extract JSON from response (Gemini might add extra text)
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start == -1 or json_end == 0:
                    raise ValueError(f"No JSON found in response: {response_text}")
                json_str = response_text[json_start:json_end]

                data = json.loads(json_str)

                # Extract model outputs
                pA = data.get('pA', 0.0)
                pB = data.get('pB', 0.0)
                cue = data.get('cue', '')
                legible = data.get('legible', 'not_legible_yet')

                # Validate probabilities
                if abs(pA + pB - 1.0) > 0.1:
                    logger.warning(f"Probabilities don't sum to 1: pA={pA}, pB={pB}")

                # Post-process: compute choice and confidence
                # Threshold: 0.52 keeps a small "truly 50/50" band but
                # avoids mislabelling any weak signal as "C".
                max_p = max(pA, pB)
                confidence = int(round(max_p * 100))
                if max_p >= 0.52:
                    choice = 'A' if pA >= pB else 'B'
                else:
                    choice = 'C'

                # Create result with full API metadata
                result = EvaluationResult(
                    # Core evaluation data
                    video_id=manifest_entry.video_id,
                    video_path=manifest_entry.video_path,
                    goal_gt=manifest_entry.goal_gt,
                    goal_A=manifest_entry.goal_A,
                    goal_B=manifest_entry.goal_B,
                    scene_id=manifest_entry.scene_id,
                    task_family=manifest_entry.task_family,
                    traj_type=manifest_entry.traj_type,
                    t_sec=t_sec,
                    frame_idx=frame_idx,
                    pA=pA,
                    pB=pB,
                    choice=choice,
                    confidence=confidence,
                    cue=cue,
                    legible=legible,
                    # API metadata (critical for reproducibility)
                    provider="google",
                    model=self.model,
                    endpoint="generativelanguage.googleapis.com/v1beta/models",
                    temperature=0.0,
                    top_p=1.0,
                    top_k=40,
                    max_output_tokens=1024,
                    candidate_count=1,
                    seed=None,
                    safety_settings=None,
                    latency_ms=latency_ms,
                    http_status=http_status,
                    retry_count=retry_count,
                    request_id=request_id,
                    response_id=response_id,
                    api_key_source="env:GEMINI_API_KEY",
                    evaluation_mode=mode
                )

                logger.info(f"{manifest_entry.video_id} t={t_sec}s: choice={result.choice}, confidence={result.confidence} (latency={latency_ms}ms, retries={retry_count})")
                return result
                
            except Exception as e:
                retry_count = attempt + 1
                last_exception = e
                if attempt < 2:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2 ** attempt)  # Simple backoff: 1s, 2s, 4s
                else:
                    logger.error(f"All 3 attempts failed for {manifest_entry.video_id} t={t_sec}s: {e}")
                    raise last_exception


def evaluate_frame(frame: bytes, prompt: str, video_id: str = "", t_sec: int = 0, manifest_entry: Optional[ManifestEntry] = None) -> EvaluationResult:
    """
    Evaluate a frame using Gemini.
    This is a wrapper for backward compatibility.
    For new code, use GeminiClient directly.
    """
    client = GeminiClient()
    
    # If manifest_entry not provided, create a minimal one
    if manifest_entry is None:
        manifest_entry = ManifestEntry(
            video_id=video_id,
            video_path="",
            goal_gt="A",
            goal_A="Goal A",
            goal_B="Goal B",
            scene_id="unknown",
            task_family="unknown",
            traj_type="unknown",
            notes=""
        )
    
    frame_idx = int(t_sec)
    return client.evaluate_frame(frame, manifest_entry, t_sec, frame_idx)