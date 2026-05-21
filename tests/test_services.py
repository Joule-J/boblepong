import base64
from io import BytesIO

from PIL import Image

from app.models import (
    CaptureMotionRequest,
    GameSessionCreate,
    LandmarkSnapshot,
    PoseFeatures,
    ReferencePoseCard,
    ReferencePoseCreate,
    RegionWeights,
    SessionFrameRecord,
)
from app.storage import JsonStore
from app.services.game_session_service import GameSessionService
from app.services.meme_card_service import MemeCardService
from app.services.motion_package_service import MotionPackageService
from app.services.reference_pose_service import ReferencePoseService
from app.services.similarity_scoring_service import SimilarityScoringService
from app.services.vision_analysis_service import VisionAnalysisService


def build_frame_data_url(color: tuple[int, int, int] = (220, 180, 90)) -> str:
    buffer = BytesIO()
    Image.new("RGB", (48, 48), color=color).save(buffer, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_feature_vector_flattens_regions():
    features = PoseFeatures(face=[1, 2], head=[3], hand=[4], body=[5, 6])
    assert features.feature_vector == [1, 2, 3, 4, 5, 6]


def test_motion_package_capture_stores_landmark_snapshot(tmp_path):
    vision = VisionAnalysisService()
    service = MotionPackageService(store=JsonStore(tmp_path / "motion_packages.json"), vision_service=vision)
    package = service.capture_package(
        CaptureMotionRequest(
            frame_data_urls=[build_frame_data_url(), build_frame_data_url()],
            pattern_name="snapshot_test",
        )
    )
    assert package.feature_vector
    assert package.landmark_snapshot.provider_name
    assert package.provider_name == package.landmark_snapshot.provider_name
    assert package.landmark_snapshot.anchors


def test_similarity_uses_region_weights():
    scoring = SimilarityScoringService()
    pose_card = ReferencePoseCard(
        reference_id="ref-1",
        image_path="/photos/demo.png",
        pattern_name="lean_left",
        region_weights=RegionWeights(face=0, head=0, hand=0, body=1),
        landmark_snapshot=LandmarkSnapshot(face=[0], head=[0], hand=[0], body=[1]),
        feature_vector=[0, 0, 0, 1],
    )
    features = PoseFeatures(face=[10], head=[10], hand=[10], body=[1])
    result = scoring.score_against_references(features, [pose_card])
    assert result is not None
    assert result.total_score == result.region_scores.body


def test_hold_progress_requires_sustained_threshold(tmp_path):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    package_store = JsonStore(tmp_path / "motion_packages.json")
    store = JsonStore(tmp_path / "references.json")
    vision = VisionAnalysisService()
    motion_package_service = MotionPackageService(store=package_store, vision_service=vision)
    package_store.save_many(
        [
            motion_package_service.capture_package(
                CaptureMotionRequest(
                    frame_data_urls=[build_frame_data_url(), build_frame_data_url(), build_frame_data_url()],
                    pattern_name="hold_test_package",
                )
            )
        ]
    )
    package = motion_package_service.list_packages()[0]
    reference_service = ReferencePoseService(
        photos_dir=photos_dir,
        store=store,
        motion_package_service=motion_package_service,
    )
    reference = reference_service.create_reference(
        ReferencePoseCreate(
            image_path="/photos/a.png",
            motion_package_id=package.motion_package_id,
            pattern_name="surprised",
            success_threshold=82,
        )
    )
    session_service = GameSessionService(
        store=JsonStore(tmp_path / "sessions.json"),
        reference_service=reference_service,
        vision_service=vision,
        scoring_service=SimilarityScoringService(),
        meme_card_service=MemeCardService(),
    )
    session = session_service.create_session(GameSessionCreate(target_reference_id=reference.reference_id))
    records = session.model_copy(update={"score_history": []})
    progress = session_service._compute_hold_progress([], 82, records)
    assert progress == 0.0

    progress = session_service._compute_hold_progress(
        [SessionFrameRecord(total_score=85), SessionFrameRecord(total_score=90), SessionFrameRecord(total_score=88)],
        82,
        session,
    )
    assert progress == 100.0
