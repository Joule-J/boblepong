from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.models import (
    AnalyzeFrameRequest,
    CaptureMotionRequest,
    GameSessionCreate,
    ReferencePoseCreate,
    ReferencePoseUpdate,
    TrainingCaptureRequest,
)
from app.storage import JsonStore
from app.services.game_session_service import GameSessionService
from app.services.meme_card_service import MemeCardService
from app.services.motion_package_service import MotionPackageService
from app.services.reference_pose_service import ReferencePoseService
from app.services.similarity_scoring_service import SimilarityScoringService
from app.services.training_service import TrainingService
from app.services.vision_analysis_service import VisionAnalysisService

BASE_DIR = Path(__file__).resolve().parent.parent
PHOTOS_DIR = BASE_DIR / "photos"
DATA_DIR = BASE_DIR / "app" / "data"

app = FastAPI(title="SpongeBob Meme Pose Game", version="0.1.0")
app.mount("/photos", StaticFiles(directory=PHOTOS_DIR), name="photos")
app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")

vision_service = VisionAnalysisService()
motion_package_service = MotionPackageService(
    store=JsonStore(DATA_DIR / "motion_packages.json"),
    vision_service=vision_service,
)
reference_service = ReferencePoseService(
    photos_dir=PHOTOS_DIR,
    store=JsonStore(DATA_DIR / "references.json"),
    motion_package_service=motion_package_service,
)
game_session_service = GameSessionService(
    store=JsonStore(DATA_DIR / "sessions.json"),
    reference_service=reference_service,
    vision_service=vision_service,
    scoring_service=SimilarityScoringService(),
    meme_card_service=MemeCardService(),
)
training_service = TrainingService(
    capture_log_store=JsonStore(DATA_DIR / "capture_logs.json"),
    motion_package_service=motion_package_service,
    reference_service=reference_service,
    scoring_service=SimilarityScoringService(),
)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (BASE_DIR / "app" / "templates" / "index.html").read_text()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "vision_provider": vision_service.provider_name}


@app.post("/api/vision/analyze")
def analyze_vision(payload: AnalyzeFrameRequest):
    try:
        features = vision_service.extract_from_data_url(payload.frame_data_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "provider_name": features.landmark_snapshot.provider_name or vision_service.provider_name,
        "feature_vector": features.feature_vector,
        "missing_regions": features.missing_regions,
        "visibility_flags": features.visibility_flags,
        "landmark_snapshot": vision_service.build_landmark_snapshot(features),
    }


@app.post("/api/training/photo-match")
def training_photo_match(payload: AnalyzeFrameRequest):
    try:
        features = vision_service.extract_from_data_url(payload.frame_data_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "provider_name": features.landmark_snapshot.provider_name or vision_service.provider_name,
        "candidates": training_service.training_photo_match(features),
        "anchors": features.landmark_snapshot.anchors,
    }


@app.get("/api/capture-logs")
def list_capture_logs():
    return training_service.list_capture_logs()


@app.post("/api/training/capture")
def capture_training(payload: TrainingCaptureRequest):
    try:
        log = training_service.capture_training_reference(payload)
        candidates = training_service.training_photo_match(
            vision_service.extract_from_data_url(payload.frame_data_urls[-1])
        )
        return training_service.attach_candidates_to_latest_log(log.capture_log_id, candidates)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/photo-library")
def photo_library():
    return reference_service.list_photos()


@app.get("/api/motion-packages")
def list_motion_packages():
    return motion_package_service.list_packages()


@app.post("/api/motion-packages")
def create_motion_package(payload: CaptureMotionRequest):
    try:
        return motion_package_service.capture_package(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/references")
def list_references():
    return reference_service.list_references()


@app.post("/api/references")
def create_reference(payload: ReferencePoseCreate):
    return reference_service.create_reference(payload)


@app.patch("/api/references/{reference_id}")
def update_reference(reference_id: str, payload: ReferencePoseUpdate):
    try:
        return reference_service.update_reference(reference_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Reference not found") from exc


@app.post("/api/references/{reference_id}/extract")
def extract_reference(reference_id: str):
    try:
        return reference_service.extract_reference(reference_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Reference not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/game/session")
def create_session(payload: GameSessionCreate):
    try:
        return game_session_service.create_session(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/game/session/{session_id}/analyze-frame")
def analyze_frame(session_id: str, payload: AnalyzeFrameRequest):
    try:
        return game_session_service.analyze_frame(session_id, payload.frame_data_url)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/game/session/{session_id}/complete")
def complete_session(session_id: str):
    try:
        return game_session_service.complete_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@app.get("/api/game/session/{session_id}/result")
def session_result(session_id: str):
    try:
        return game_session_service.get_result(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
