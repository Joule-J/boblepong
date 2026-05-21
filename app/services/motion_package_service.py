from __future__ import annotations

from uuid import uuid4

from app.models import CaptureMotionRequest, LandmarkSnapshot, MotionPackage, PoseFeatures, utc_now
from app.storage import JsonStore
from app.services.vision_analysis_service import VisionAnalysisService


class MotionPackageService:
    def __init__(self, store: JsonStore, vision_service: VisionAnalysisService):
        self.store = store
        self.vision_service = vision_service

    def list_packages(self) -> list[MotionPackage]:
        return self.store.load_many(MotionPackage)

    def get_package(self, motion_package_id: str) -> MotionPackage:
        for item in self.list_packages():
            if item.motion_package_id == motion_package_id:
                return item
        raise KeyError(motion_package_id)

    def capture_package(self, payload: CaptureMotionRequest) -> MotionPackage:
        if not payload.frame_data_urls:
            raise ValueError("At least one frame is required to build a motion package.")
        samples = [self.vision_service.extract_from_data_url(frame) for frame in payload.frame_data_urls]
        aggregated = self._aggregate_features(samples)
        package = MotionPackage(
            motion_package_id=str(uuid4()),
            pattern_name=payload.pattern_name,
            tags=payload.tags,
            sample_label=payload.sample_label,
            region_weights=payload.region_weights,
            hint_templates=payload.hint_templates,
            landmark_snapshot=self.vision_service.build_landmark_snapshot(aggregated),
            feature_vector=aggregated.feature_vector,
            source_frame_count=len(payload.frame_data_urls),
            provider_name=aggregated.landmark_snapshot.provider_name or self.vision_service.provider_name,
        )
        packages = self.list_packages()
        packages.append(package)
        self.store.save_many(packages)
        return package

    def _aggregate_features(self, samples: list[PoseFeatures]) -> PoseFeatures:
        providers = {sample.landmark_snapshot.provider_name for sample in samples if sample.landmark_snapshot.provider_name}
        snapshot = LandmarkSnapshot(
            face_points=self._avg_points([sample.landmark_snapshot.face_points for sample in samples]),
            pose_points=self._avg_points([sample.landmark_snapshot.pose_points for sample in samples]),
            left_hand_points=self._avg_points([sample.landmark_snapshot.left_hand_points for sample in samples]),
            right_hand_points=self._avg_points([sample.landmark_snapshot.right_hand_points for sample in samples]),
            anchors=self._avg_anchors([sample.landmark_snapshot.anchors for sample in samples]),
            provider_name=providers.pop() if len(providers) == 1 else "mixed",
        )
        return PoseFeatures(
            face=self._avg_region([sample.face for sample in samples]),
            head=self._avg_region([sample.head for sample in samples]),
            hand=self._avg_region([sample.hand for sample in samples]),
            body=self._avg_region([sample.body for sample in samples]),
            missing_regions=self._missing_regions(samples),
            visibility_flags=self._visibility(samples),
            landmark_snapshot=snapshot,
        )

    def _avg_region(self, vectors: list[list[float]]) -> list[float]:
        valid = [vector for vector in vectors if vector]
        if not valid:
            return []
        max_len = max(len(vector) for vector in valid)
        result = []
        for idx in range(max_len):
            values = [vector[idx] for vector in valid if idx < len(vector)]
            result.append(round(sum(values) / len(values), 4))
        return result

    def _missing_regions(self, samples: list[PoseFeatures]) -> list[str]:
        missing = []
        for region in ("face", "head", "hand", "body"):
            if any(region in sample.missing_regions for sample in samples):
                missing.append(region)
        return missing

    def _visibility(self, samples: list[PoseFeatures]):
        from app.models import VisibilityFlags

        return VisibilityFlags(
            face=all(sample.visibility_flags.face for sample in samples),
            head=all(sample.visibility_flags.head for sample in samples),
            hand=all(sample.visibility_flags.hand for sample in samples),
            body=all(sample.visibility_flags.body for sample in samples),
        )

    def _avg_points(self, point_sets):
        from app.models import LandmarkPoint

        valid = [points for points in point_sets if points]
        if not valid:
            return []
        max_len = max(len(points) for points in valid)
        averaged = []
        for idx in range(max_len):
            samples = [points[idx] for points in valid if idx < len(points)]
            averaged.append(
                LandmarkPoint(
                    x=round(sum(item.x for item in samples) / len(samples), 4),
                    y=round(sum(item.y for item in samples) / len(samples), 4),
                    z=round(sum(item.z for item in samples) / len(samples), 4),
                    visibility=round(
                        sum((item.visibility or 0.0) for item in samples) / len(samples), 4
                    )
                    if any(item.visibility is not None for item in samples)
                    else None,
                )
            )
        return averaged

    def _avg_anchors(self, anchor_maps: list[dict[str, list[float]]]) -> dict[str, list[float]]:
        keys = {key for anchors in anchor_maps for key in anchors.keys()}
        averaged: dict[str, list[float]] = {}
        for key in keys:
            values = [anchors[key] for anchors in anchor_maps if key in anchors]
            if not values:
                continue
            width = max(len(item) for item in values)
            averaged[key] = [
                round(sum(item[idx] for item in values if idx < len(item)) / len(values), 4)
                for idx in range(width)
            ]
        return averaged
