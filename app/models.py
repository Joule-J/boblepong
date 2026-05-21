from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field


REGION_KEYS = ("face", "head", "hand", "body")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RegionWeights(BaseModel):
    face: float = 1.0
    head: float = 1.0
    hand: float = 1.0
    body: float = 1.0


class RegionScores(BaseModel):
    face: float = 0.0
    head: float = 0.0
    hand: float = 0.0
    body: float = 0.0


class VisibilityFlags(BaseModel):
    face: bool = True
    head: bool = True
    hand: bool = True
    body: bool = True


class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float = 0.0
    visibility: float | None = None


class LandmarkSnapshot(BaseModel):
    face: list[float] = Field(default_factory=list)
    head: list[float] = Field(default_factory=list)
    hand: list[float] = Field(default_factory=list)
    body: list[float] = Field(default_factory=list)
    face_points: list[LandmarkPoint] = Field(default_factory=list)
    pose_points: list[LandmarkPoint] = Field(default_factory=list)
    left_hand_points: list[LandmarkPoint] = Field(default_factory=list)
    right_hand_points: list[LandmarkPoint] = Field(default_factory=list)
    anchors: dict[str, list[float]] = Field(default_factory=dict)
    provider_name: str = "unknown"


class PoseFeatures(BaseModel):
    face: list[float] = Field(default_factory=list)
    head: list[float] = Field(default_factory=list)
    hand: list[float] = Field(default_factory=list)
    body: list[float] = Field(default_factory=list)
    missing_regions: list[str] = Field(default_factory=list)
    visibility_flags: VisibilityFlags = Field(default_factory=VisibilityFlags)
    landmark_snapshot: LandmarkSnapshot = Field(default_factory=LandmarkSnapshot)

    @computed_field
    @property
    def feature_vector(self) -> list[float]:
        return self.face + self.head + self.hand + self.body


class ReferencePoseCard(BaseModel):
    reference_id: str
    image_path: str
    motion_package_id: str | None = None
    character: str = "spongebob"
    pattern_name: str
    tags: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    landmark_snapshot: LandmarkSnapshot = Field(default_factory=LandmarkSnapshot)
    feature_vector: list[float] = Field(default_factory=list)
    region_weights: RegionWeights = Field(default_factory=RegionWeights)
    hint_templates: dict[str, list[str]] = Field(default_factory=dict)
    success_threshold: float = 82.0
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class ReferencePoseCreate(BaseModel):
    image_path: str
    motion_package_id: str
    pattern_name: str
    tags: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    character: str = "spongebob"
    region_weights: RegionWeights = Field(default_factory=RegionWeights)
    hint_templates: dict[str, list[str]] = Field(default_factory=dict)
    success_threshold: float = 82.0


class ReferencePoseUpdate(BaseModel):
    motion_package_id: str | None = None
    pattern_name: str | None = None
    tags: list[str] | None = None
    difficulty: Literal["easy", "medium", "hard"] | None = None
    character: str | None = None
    region_weights: RegionWeights | None = None
    hint_templates: dict[str, list[str]] | None = None
    success_threshold: float | None = None


class AnalyzeFrameRequest(BaseModel):
    frame_data_url: str


class CaptureMotionRequest(BaseModel):
    frame_data_urls: list[str] = Field(default_factory=list)
    pattern_name: str
    tags: list[str] = Field(default_factory=list)
    sample_label: str | None = None
    region_weights: RegionWeights = Field(default_factory=RegionWeights)
    hint_templates: dict[str, list[str]] = Field(default_factory=dict)


class TrainingCaptureRequest(BaseModel):
    frame_data_urls: list[str] = Field(default_factory=list)
    image_path: str
    pattern_name: str
    tags: list[str] = Field(default_factory=list)
    sample_label: str | None = None
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    success_threshold: float = 82.0
    region_weights: RegionWeights = Field(default_factory=RegionWeights)
    hint_templates: dict[str, list[str]] = Field(default_factory=dict)


class GameSessionCreate(BaseModel):
    target_reference_id: str | None = None
    mode: Literal["live_mimic", "guess_the_meme"] = "live_mimic"


class CandidateMatch(BaseModel):
    reference_id: str
    motion_package_id: str | None = None
    pattern_name: str
    total_score: float


class FrameAnalysisDebug(BaseModel):
    top_candidates: list[CandidateMatch] = Field(default_factory=list)
    normalized_features: dict[str, list[float]] = Field(default_factory=dict)
    missing_regions: list[str] = Field(default_factory=list)
    visibility_flags: VisibilityFlags = Field(default_factory=VisibilityFlags)
    capture_anchors: dict[str, list[float]] = Field(default_factory=dict)
    provider_name: str = "unknown"


class FrameAnalysisResponse(BaseModel):
    matched_reference_id: str | None
    matched_motion_package_id: str | None = None
    matched_pattern_name: str | None
    total_score: float
    region_scores: RegionScores
    feedback_hints: list[str] = Field(default_factory=list)
    confidence: float
    hold_progress: float
    debug: FrameAnalysisDebug


class SessionFrameRecord(BaseModel):
    total_score: float
    captured_at: str = Field(default_factory=utc_now)


class MemeCard(BaseModel):
    target_reference_id: str
    motion_package_id: str | None = None
    target_image_path: str
    best_frame_data_url: str | None = None
    pattern_name: str
    caption: str
    score: float
    share_text: str


class GameSession(BaseModel):
    session_id: str
    mode: Literal["live_mimic", "guess_the_meme"] = "live_mimic"
    target_reference_id: str
    status: Literal["active", "completed"] = "active"
    score_history: list[SessionFrameRecord] = Field(default_factory=list)
    hold_seconds_required: float = 1.2
    frame_interval_seconds: float = 0.4
    best_score: float = 0.0
    best_frame_data_url: str | None = None
    last_analysis: FrameAnalysisResponse | None = None
    result: MemeCard | None = None
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class SessionSummary(BaseModel):
    session_id: str
    status: str
    target_reference_id: str
    mode: str
    best_score: float
    hold_progress: float = 0.0


class CompletionResponse(BaseModel):
    session: SessionSummary
    result: MemeCard | None = None
    message: str


class PhotoAsset(BaseModel):
    filename: str
    image_path: str


class MotionPackage(BaseModel):
    motion_package_id: str
    pattern_name: str
    tags: list[str] = Field(default_factory=list)
    sample_label: str | None = None
    region_weights: RegionWeights = Field(default_factory=RegionWeights)
    hint_templates: dict[str, list[str]] = Field(default_factory=dict)
    landmark_snapshot: LandmarkSnapshot = Field(default_factory=LandmarkSnapshot)
    feature_vector: list[float] = Field(default_factory=list)
    source_frame_count: int = 0
    provider_name: str = "unknown"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class TrainingPhotoCandidate(BaseModel):
    image_path: str
    reference_id: str
    pattern_name: str
    score: float


class CaptureLog(BaseModel):
    capture_log_id: str
    image_path: str
    pattern_name: str
    tags: list[str] = Field(default_factory=list)
    sample_label: str | None = None
    provider_name: str = "unknown"
    motion_package_id: str
    reference_id: str
    source_frame_count: int = 0
    anchors: dict[str, list[float]] = Field(default_factory=dict)
    training_candidates: list[TrainingPhotoCandidate] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class StorageEnvelope(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list)
