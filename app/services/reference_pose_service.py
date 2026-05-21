from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.models import PhotoAsset, ReferencePoseCard, ReferencePoseCreate, ReferencePoseUpdate, utc_now
from app.storage import JsonStore
from app.services.motion_package_service import MotionPackageService


class ReferencePoseService:
    def __init__(self, photos_dir: Path, store: JsonStore, motion_package_service: MotionPackageService):
        self.photos_dir = photos_dir
        self.store = store
        self.motion_package_service = motion_package_service

    def list_photos(self) -> list[PhotoAsset]:
        assets = []
        for path in sorted(self.photos_dir.glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            assets.append(PhotoAsset(filename=path.name, image_path=f"/photos/{path.name}"))
        return assets

    def list_references(self) -> list[ReferencePoseCard]:
        return self.store.load_many(ReferencePoseCard)

    def get_reference(self, reference_id: str) -> ReferencePoseCard:
        for reference in self.list_references():
            if reference.reference_id == reference_id:
                return reference
        raise KeyError(reference_id)

    def create_reference(self, payload: ReferencePoseCreate) -> ReferencePoseCard:
        references = self.list_references()
        package = self.motion_package_service.get_package(payload.motion_package_id)
        reference = ReferencePoseCard(
            reference_id=str(uuid4()),
            landmark_snapshot=package.landmark_snapshot,
            feature_vector=package.feature_vector,
            **payload.model_dump(),
        )
        references.append(reference)
        self.store.save_many(references)
        return reference

    def update_reference(self, reference_id: str, payload: ReferencePoseUpdate) -> ReferencePoseCard:
        references = self.list_references()
        for idx, reference in enumerate(references):
            if reference.reference_id != reference_id:
                continue
            package_id = payload.motion_package_id or reference.motion_package_id
            update_data = {k: v for k, v in payload.model_dump().items() if v is not None}
            if package_id:
                package = self.motion_package_service.get_package(package_id)
                update_data["landmark_snapshot"] = package.landmark_snapshot
                update_data["feature_vector"] = package.feature_vector
            updated = reference.model_copy(update=update_data)
            updated.updated_at = utc_now()
            references[idx] = updated
            self.store.save_many(references)
            return updated
        raise KeyError(reference_id)

    def extract_reference(self, reference_id: str) -> ReferencePoseCard:
        references = self.list_references()
        for idx, reference in enumerate(references):
            if reference.reference_id != reference_id:
                continue
            if not reference.motion_package_id:
                raise ValueError("Reference is not bound to a motion package.")
            package = self.motion_package_service.get_package(reference.motion_package_id)
            updated = reference.model_copy(
                update={
                    "landmark_snapshot": package.landmark_snapshot,
                    "feature_vector": package.feature_vector,
                    "updated_at": utc_now(),
                }
            )
            references[idx] = updated
            self.store.save_many(references)
            return updated
        raise KeyError(reference_id)

    def resolve_image_path(self, image_path: str) -> Path:
        if image_path.startswith("/photos/"):
            return self.photos_dir / image_path.replace("/photos/", "", 1)
        path = Path(image_path)
        if not path.is_absolute():
            path = self.photos_dir.parent / image_path
        return path
