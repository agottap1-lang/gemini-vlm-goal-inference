from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional

class ManifestEntry(BaseModel):
    video_id: str = Field(..., description="Unique video identifier")
    video_path: str = Field(..., description="Path to video file")
    goal_gt: Literal["A", "B"] = Field(..., description="Ground truth goal")
    goal_A: str = Field(..., description="Description of goal A")
    goal_B: str = Field(..., description="Description of goal B")
    scene_id: str = Field(..., description="Scene identifier")
    task_family: str = Field(..., description="Task family (e.g., block_pick, drawer_close)")
    traj_type: str = Field(..., description="Trajectory type (e.g., ambiguous, legible)")
    notes: str = Field(..., description="Additional notes")

class EvaluationResult(BaseModel):
    # Core evaluation data
    video_id: str = Field(..., description="Video identifier")
    video_path: str = Field(..., description="Path to video file")
    goal_gt: Literal["A", "B"] = Field(..., description="Ground truth goal")
    goal_A: str = Field(..., description="Description of goal A")
    goal_B: str = Field(..., description="Description of goal B")
    scene_id: str = Field(..., description="Scene identifier")
    task_family: str = Field(..., description="Task family")
    traj_type: str = Field(..., description="Trajectory type")
    t_sec: int = Field(..., description="Time in seconds (integer)")
    frame_idx: int = Field(..., description="Frame index")
    pA: float = Field(..., ge=0.0, le=1.0, description="Probability of goal A")
    pB: float = Field(..., ge=0.0, le=1.0, description="Probability of goal B")
    choice: Literal["A", "B", "C"] = Field(default="C", description="Predicted choice")
    confidence: int = Field(default=0, ge=0, le=100, description="Confidence score")
    cue: str = Field(..., description="Visual cue")
    legible: Literal["legible_now", "not_legible_yet"] = Field(..., description="Legibility status")
    
    # API metadata (critical for reproducibility of closed-source calls)
    provider: str = Field(default="google", description="API provider")
    model: str = Field(default="gemini-2.5-flash", description="Model identifier")
    endpoint: str = Field(default="generativelanguage.googleapis.com/v1beta/models", description="API endpoint")
    temperature: float = Field(default=0.0, description="Temperature parameter")
    top_p: float = Field(default=1.0, description="Top-P parameter")
    top_k: int = Field(default=40, description="Top-K parameter")
    max_output_tokens: int = Field(default=1024, description="Max output tokens")
    candidate_count: int = Field(default=1, description="Number of candidates")
    seed: Optional[int] = Field(default=None, description="Seed for reproducibility")
    safety_settings: Optional[dict] = Field(default=None, description="Safety filter settings")
    latency_ms: int = Field(default=0, description="API call latency in milliseconds")
    http_status: int = Field(default=200, description="HTTP response status")
    retry_count: int = Field(default=0, description="Number of retries before success")
    request_id: Optional[str] = Field(default=None, description="Request ID from API")
    response_id: Optional[str] = Field(default=None, description="Response ID from API")
    api_key_source: str = Field(default="env:GEMINI_API_KEY", description="Source of API key (no actual key stored)")
    evaluation_mode: str = Field(default="single_frame", description="Evaluation mode: single_frame or prefix_frames")

    @field_validator('choice', mode='before')
    @classmethod
    def enforce_choice_rule(cls, v, info):
        # Respect provided value; only compute if missing
        if v is not None:
            return v
        pA = info.data.get('pA', 0.0)
        pB = info.data.get('pB', 0.0)
        max_p = max(pA, pB)
        # Threshold 0.52 – keeps a tiny "truly 50/50" band but avoids
        # mislabelling weak signals as "C".
        if max_p >= 0.52:
            return 'A' if pA >= pB else 'B'
        return 'C'

    @field_validator('confidence', mode='before')
    @classmethod
    def compute_confidence(cls, v, info):
        # Respect provided value; only compute if missing
        if v is not None:
            return v
        pA = info.data.get('pA', 0.0)
        pB = info.data.get('pB', 0.0)
        return int(round(max(pA, pB) * 100))