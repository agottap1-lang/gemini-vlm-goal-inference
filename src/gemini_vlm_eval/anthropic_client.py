import base64
import json
import logging
import os
import time
from typing import List, Optional, Union

from dotenv import load_dotenv

from .schema import EvaluationResult, ManifestEntry
from .prompt import get_instruction_prompt

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-5"
ENDPOINT = "api.anthropic.com/v1/messages"


class AnthropicClient:
    """Anthropic Claude vision client matching the GeminiClient interface."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL,
                 request_delay_s: float = 1.5):
        try:
            import anthropic as _anthropic  # noqa: PLC0415
            self._anthropic = _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for AnthropicClient. "
                "Install it with: pip install anthropic"
            ) from exc

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable must be set")
        self.model = model
        self.request_delay_s = request_delay_s  # pacing to avoid 429 rate limits
        self._client = self._anthropic.Anthropic(api_key=self.api_key)
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Public interface (mirrors GeminiClient.evaluate_frame exactly)
    # ------------------------------------------------------------------

    def evaluate_frame(
        self,
        image_bytes: Union[bytes, List[bytes]],
        manifest_entry: ManifestEntry,
        t_sec: int,
        frame_idx: int,
        mode: str = "single_frame",
    ) -> EvaluationResult:
        """Evaluate frame(s) using Anthropic Claude vision and return an EvaluationResult."""

        last_exception = None

        # Rate-limit pacing: enforce minimum gap between requests
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay_s:
            time.sleep(self.request_delay_s - elapsed)

        for attempt in range(3):
            try:
                prompt = get_instruction_prompt(
                    manifest_entry.goal_A,
                    manifest_entry.goal_B,
                    t_sec,
                    manifest_entry.video_id,
                    mode=mode,
                )

                # Build message content: images first, then the text prompt
                content: list = []

                if mode == "prefix_frames" and isinstance(image_bytes, list):
                    for img in image_bytes:
                        b64 = base64.b64encode(img).decode("utf-8")
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        })
                else:
                    img_data = image_bytes if isinstance(image_bytes, bytes) else image_bytes[0]
                    b64 = base64.b64encode(img_data).decode("utf-8")
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    })

                content.append({"type": "text", "text": prompt})

                start_time = time.time()
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    temperature=0.0,
                    messages=[{"role": "user", "content": content}],
                )
                self._last_request_time = time.time()
                latency_ms = int((self._last_request_time - start_time) * 1000)

                response_text = response.content[0].text.strip()
                logger.debug(
                    f"Anthropic response for {manifest_entry.video_id} at t={t_sec}s: {response_text}"
                )

                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start == -1 or json_end == 0:
                    raise ValueError(f"No JSON in response: {response_text}")
                data = json.loads(response_text[json_start:json_end])

                pA = float(data.get("pA", 0.0))
                pB = float(data.get("pB", 0.0))
                cue = str(data.get("cue", ""))
                legible = str(data.get("legible", "not_legible_yet"))

                if abs(pA + pB - 1.0) > 0.1:
                    logger.warning(f"Probabilities don't sum to 1: pA={pA}, pB={pB}")

                max_p = max(pA, pB)
                confidence = int(round(max_p * 100))
                choice = ("A" if pA >= pB else "B") if max_p >= 0.52 else "C"

                # Anthropic returns usage/id info in response object
                response_id = getattr(response, "id", None)

                return EvaluationResult(
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
                    # API metadata
                    provider="anthropic",
                    model=self.model,
                    endpoint=ENDPOINT,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,           # not used
                    max_output_tokens=512,
                    candidate_count=1,
                    seed=None,         # Anthropic doesn't support seed
                    safety_settings=None,
                    latency_ms=latency_ms,
                    http_status=200,
                    retry_count=attempt,
                    request_id=response_id,
                    response_id=response_id,
                    api_key_source="env:ANTHROPIC_API_KEY",
                    evaluation_mode=mode,
                )

            except Exception as exc:
                last_exception = exc
                logger.warning(
                    f"Anthropic attempt {attempt + 1}/3 failed for "
                    f"{manifest_entry.video_id} at t={t_sec}s: {exc}"
                )
                time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Anthropic evaluation failed after 3 attempts: {last_exception}"
        )
