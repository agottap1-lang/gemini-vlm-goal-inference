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

DEFAULT_MODEL = "gpt-4o"
ENDPOINT = "api.openai.com/v1/chat/completions"


class OpenAIClient:
    """OpenAI GPT-4o vision client matching the GeminiClient interface."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAIClient. "
                "Install it with: pip install openai"
            ) from exc

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        self.model = model
        self._client = OpenAI(api_key=self.api_key)

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
        """Evaluate frame(s) using OpenAI vision and return an EvaluationResult."""

        last_exception = None

        for attempt in range(3):
            try:
                prompt = get_instruction_prompt(
                    manifest_entry.goal_A,
                    manifest_entry.goal_B,
                    t_sec,
                    manifest_entry.video_id,
                    mode=mode,
                )

                # Build the message content list
                content: list = [{"type": "text", "text": prompt}]

                if mode == "prefix_frames" and isinstance(image_bytes, list):
                    for img in image_bytes:
                        b64 = base64.b64encode(img).decode("utf-8")
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "low",   # saves tokens; use "high" for finer detail
                            },
                        })
                else:
                    img_data = image_bytes if isinstance(image_bytes, bytes) else image_bytes[0]
                    b64 = base64.b64encode(img_data).decode("utf-8")
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "low",
                        },
                    })

                start_time = time.time()
                # GPT-5.x series requires max_completion_tokens and does not
                # accept temperature=0.0; GPT-4.x uses the old max_tokens/temperature.
                is_gpt5 = self.model.startswith("gpt-5") or self.model.startswith("o1") or self.model.startswith("o3") or self.model.startswith("o4")
                call_kwargs: dict = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": content}],
                }
                if is_gpt5:
                    call_kwargs["max_completion_tokens"] = 512
                else:
                    call_kwargs["temperature"] = 0.0
                    call_kwargs["max_tokens"] = 512
                    call_kwargs["seed"] = 42
                response = self._client.chat.completions.create(**call_kwargs)
                latency_ms = int((time.time() - start_time) * 1000)

                response_text = response.choices[0].message.content.strip()
                logger.debug(
                    f"OpenAI response for {manifest_entry.video_id} at t={t_sec}s: {response_text}"
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
                    provider="openai",
                    model=self.model,
                    endpoint=ENDPOINT,
                    temperature=0.0,          # N/A for gpt-5.x but schema requires float
                    top_p=1.0,
                    top_k=0,           # OpenAI doesn't use top_k
                    max_output_tokens=512,
                    candidate_count=1,
                    seed=42 if not is_gpt5 else 0,
                    safety_settings=None,
                    latency_ms=latency_ms,
                    http_status=200,
                    retry_count=attempt,
                    request_id=response.id,
                    response_id=response.id,
                    api_key_source="env:OPENAI_API_KEY",
                    evaluation_mode=mode,
                )

            except Exception as exc:
                last_exception = exc
                logger.warning(
                    f"OpenAI attempt {attempt + 1}/3 failed for "
                    f"{manifest_entry.video_id} at t={t_sec}s: {exc}"
                )
                time.sleep(2 ** attempt)  # exponential back-off: 1s, 2s, 4s

        raise RuntimeError(
            f"OpenAI evaluation failed after 3 attempts: {last_exception}"
        )
