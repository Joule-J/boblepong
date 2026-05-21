from __future__ import annotations

from uuid import uuid4

from app.models import (
    CaptureLog,
    CaptureMotionRequest,
    PoseFeatures,
    ReferencePoseCreate,
    TrainingCaptureRequest,
    TrainingPhotoCandidate,
)
from app.services.motion_package_service import MotionPackageService
from app.services.reference_pose_service import ReferencePoseService
from app.services.similarity_scoring_service import SimilarityScoringService
from app.storage import JsonStore


class TrainingService:
    def __init__(
        self,
        capture_log_store: JsonStore,
        motion_package_service: MotionPackageService,
        reference_service: ReferencePoseService,
        scoring_service: SimilarityScoringService,
    ) -> None:
        self.capture_log_store = capture_log_store
        self.motion_package_service = motion_package_service
        self.reference_service = reference_service
        self.scoring_service = scoring_service

    def capture_training_reference(self, payload: TrainingCaptureRequest) -> CaptureLog:
        package = self.motion_package_service.capture_package(
            CaptureMotionRequest(
                frame_data_urls=payload.frame_data_urls,
                pattern_name=payload.pattern_name,
                tags=payload.tags,
                sample_label=payload.sample_label,
                region_weights=payload.region_weights,
                hint_templates=payload.hint_templates,
            )
        )
        reference = self.reference_service.create_reference(
            ReferencePoseCreate(
                image_path=payload.image_path,
                motion_package_id=package.motion_package_id,
                pattern_name=payload.pattern_name,
                tags=payload.tags,
                difficulty=payload.difficulty,
                region_weights=payload.region_weights,
                hint_templates=payload.hint_templates,
                success_threshold=payload.success_threshold,
            )
        )
        log = CaptureLog(
            capture_log_id=str(uuid4()),
            image_path=payload.image_path,
            pattern_name=payload.pattern_name,
            tags=payload.tags,
            sample_label=payload.sample_label,
            provider_name=package.provider_name,
            motion_package_id=package.motion_package_id,
            reference_id=reference.reference_id,
            source_frame_count=package.source_frame_count,
            anchors=package.landmark_snapshot.anchors,
            training_candidates=[],
        )
        logs = self.list_capture_logs()
        logs.append(log)
        self.capture_log_store.save_many(logs)
        return log

    def list_capture_logs(self) -> list[CaptureLog]:
        return self.capture_log_store.load_many(CaptureLog)

    def training_photo_match(self, features: PoseFeatures) -> list[TrainingPhotoCandidate]:
        references = self.reference_service.list_references()
        if not references:
            return []
        # Training mode prioritizes face + body posture. Hands contribute lightly.
        boosted = []
        for reference in references:
            adjusted = reference.model_copy(
                update={
                    "region_weights": reference.region_weights.model_copy(
                        update={"face": 2.2, "head": 1.6, "hand": 0.4, "body": 2.0}
                    )
                }
            )
            boosted.append(adjusted)
        result = self.scoring_service.score_against_references(features, boosted)
        if result is None:
            return []
        candidates = []
        for item in result.top_candidates:
            reference = next((ref for ref in references if ref.reference_id == item.reference_id), None)
            if reference is None:
                continue
            candidates.append(
                TrainingPhotoCandidate(
                    image_path=reference.image_path,
                    reference_id=reference.reference_id,
                    pattern_name=reference.pattern_name,
                    score=item.total_score,
                )
            )
        return candidates

    def attach_candidates_to_latest_log(self, capture_log_id: str, candidates: list[TrainingPhotoCandidate]) -> CaptureLog:
        logs = self.list_capture_logs()
        for index, log in enumerate(logs):
            if log.capture_log_id != capture_log_id:
                continue
            updated = log.model_copy(update={"training_candidates": candidates})
            logs[index] = updated
            self.capture_log_store.save_many(logs)
            return updated
        raise KeyError(capture_log_id)
