import base64
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import DATA_DIR, app


client = TestClient(app)


def first_photo() -> tuple[Path, str]:
    photos = client.get("/api/photo-library")
    assert photos.status_code == 200
    photo_path = photos.json()[0]["image_path"]
    image_path = Path(__file__).resolve().parents[1] / photo_path.lstrip("/")
    return image_path, photo_path


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["vision_provider"] in {"heuristic-fallback", "mediapipe"}


def test_vision_analyze_endpoint():
    image_path, _ = first_photo()
    buffer = BytesIO()
    Image.open(image_path).convert("RGB").save(buffer, format="JPEG")
    frame_data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")
    response = client.post("/api/vision/analyze", json={"frame_data_url": frame_data_url})
    assert response.status_code == 200
    payload = response.json()
    assert payload["provider_name"] in {"heuristic-fallback", "mediapipe"}
    assert isinstance(payload["feature_vector"], list)
    assert "landmark_snapshot" in payload


def test_training_capture_creates_log_and_reference():
    for name in ("references.json", "sessions.json", "motion_packages.json", "capture_logs.json"):
        path = DATA_DIR / name
        if path.exists():
            path.unlink()
    image_path, photo_path = first_photo()
    buffer = BytesIO()
    Image.open(image_path).convert("RGB").save(buffer, format="JPEG")
    frame_data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

    capture = client.post(
        "/api/training/capture",
        json={
            "frame_data_urls": [frame_data_url, frame_data_url, frame_data_url],
            "image_path": photo_path,
            "pattern_name": "training_capture_pose",
            "tags": ["training", "pose"],
            "sample_label": "capture_1",
            "difficulty": "medium",
            "success_threshold": 82,
            "region_weights": {"face": 1.8, "head": 1.2, "hand": 0.5, "body": 1.8},
            "hint_templates": {},
        },
    )
    assert capture.status_code == 200
    capture_payload = capture.json()
    assert capture_payload["motion_package_id"]
    assert capture_payload["reference_id"]

    logs = client.get("/api/capture-logs")
    assert logs.status_code == 200
    assert len(logs.json()) == 1

    references = client.get("/api/references")
    assert references.status_code == 200
    assert len(references.json()) == 1


def test_reference_extract_and_session_flow():
    for name in ("references.json", "sessions.json", "motion_packages.json"):
        path = DATA_DIR / name
        if path.exists():
            path.unlink()
    image_path, photo_path = first_photo()
    buffer = BytesIO()
    Image.open(image_path).convert("RGB").save(buffer, format="JPEG")
    frame_data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

    package = client.post(
        "/api/motion-packages",
        json={
            "frame_data_urls": [frame_data_url, frame_data_url, frame_data_url],
            "pattern_name": "api_motion_package",
            "tags": ["api", "motion"],
            "sample_label": "test_take",
            "region_weights": {"face": 1, "head": 1, "hand": 1, "body": 1},
            "hint_templates": {},
        },
    )
    assert package.status_code == 200
    motion_package_id = package.json()["motion_package_id"]

    created = client.post(
        "/api/references",
        json={
            "image_path": photo_path,
            "motion_package_id": motion_package_id,
            "pattern_name": "api_flow_pose",
            "tags": ["api", "flow"],
            "difficulty": "easy",
            "success_threshold": 70,
            "region_weights": {"face": 1, "head": 1, "hand": 1, "body": 1},
            "hint_templates": {},
        },
    )
    assert created.status_code == 200
    reference_id = created.json()["reference_id"]

    extracted = client.post(f"/api/references/{reference_id}/extract")
    assert extracted.status_code == 200
    assert extracted.json()["feature_vector"]

    session = client.post("/api/game/session", json={"mode": "live_mimic", "target_reference_id": reference_id})
    assert session.status_code == 200
    session_id = session.json()["session_id"]

    analysis = client.post(
        f"/api/game/session/{session_id}/analyze-frame",
        json={"frame_data_url": frame_data_url},
    )
    assert analysis.status_code == 200
    analysis_payload = analysis.json()
    assert analysis_payload["matched_reference_id"] == reference_id
    assert analysis_payload["matched_motion_package_id"] == motion_package_id
    assert analysis_payload["debug"]["top_candidates"]
