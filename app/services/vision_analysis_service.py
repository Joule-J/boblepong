from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.models import LandmarkPoint, LandmarkSnapshot, PoseFeatures, VisibilityFlags


@dataclass
class ImagePayload:
    image: Image.Image
    raw_bytes: bytes


class VisionAnalysisService:
    def __init__(self) -> None:
        self._mediapipe_available = False
        self._mp = None
        self._face_mesh = None
        self._pose = None
        self._hands = None
        try:
            import mediapipe as mp

            self._mp = mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
            )
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5,
            )
            self._mediapipe_available = True
        except Exception:
            self._mediapipe_available = False

    def extract_from_path(self, image_path: str) -> PoseFeatures:
        path = Path(image_path)
        image = Image.open(path).convert("RGB")
        return self._extract_from_image(image)

    def extract_from_data_url(self, frame_data_url: str) -> PoseFeatures:
        payload = self.decode_data_url(frame_data_url)
        return self._extract_from_image(payload.image)

    def decode_data_url(self, data_url: str) -> ImagePayload:
        if "," not in data_url:
            raise ValueError("Invalid data URL")
        _, encoded = data_url.split(",", 1)
        raw = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return ImagePayload(image=image, raw_bytes=raw)

    def _extract_from_image(self, image: Image.Image) -> PoseFeatures:
        if self._mediapipe_available:
            extracted = self._mediapipe_extract(image)
            if extracted is not None and extracted.feature_vector:
                return extracted
        return self._heuristic_extract(image)

    def _mediapipe_extract(self, image: Image.Image) -> PoseFeatures | None:
        if not self._face_mesh or not self._pose or not self._hands:
            return None
        rgb = np.asarray(image.convert("RGB"))
        face_result = self._face_mesh.process(rgb)
        pose_result = self._pose.process(rgb)
        hand_result = self._hands.process(rgb)

        face_points = []
        pose_points = []
        left_hand_points = []
        right_hand_points = []
        if face_result.multi_face_landmarks:
            face_points = [self._point_from_landmark(item) for item in face_result.multi_face_landmarks[0].landmark]
        if pose_result.pose_landmarks:
            pose_points = [self._point_from_landmark(item) for item in pose_result.pose_landmarks.landmark]
        if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
            for handedness, hand_landmarks in zip(hand_result.multi_handedness, hand_result.multi_hand_landmarks):
                label = handedness.classification[0].label.lower()
                parsed = [self._point_from_landmark(item) for item in hand_landmarks.landmark]
                if label == "left":
                    left_hand_points = parsed
                else:
                    right_hand_points = parsed

        visibility = VisibilityFlags(
            face=bool(face_points),
            head=bool(face_points or pose_points),
            hand=bool(left_hand_points or right_hand_points),
            body=bool(pose_points),
        )
        missing = [region for region, present in visibility.model_dump().items() if not present]
        snapshot = LandmarkSnapshot(
            face_points=face_points,
            pose_points=pose_points,
            left_hand_points=left_hand_points,
            right_hand_points=right_hand_points,
            anchors=self._build_anchors(face_points, pose_points, left_hand_points, right_hand_points),
            provider_name="mediapipe",
        )
        face_features = self._face_features(face_points)
        head_features = self._head_features(face_points, pose_points)
        hand_features = self._hand_features(left_hand_points, right_hand_points, pose_points)
        body_features = self._body_features(pose_points)
        return PoseFeatures(
            face=self._normalize(face_features),
            head=self._normalize(head_features),
            hand=self._normalize(hand_features),
            body=self._normalize(body_features),
            missing_regions=missing,
            visibility_flags=visibility,
            landmark_snapshot=snapshot,
        )

    def _heuristic_extract(self, image: Image.Image) -> PoseFeatures:
        array = np.asarray(image.resize((180, 180)), dtype=np.float32) / 255.0
        gray = array.mean(axis=2)
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        energy = (gx + gy) / 2.0

        top = gray[:80, :]
        mid = gray[60:130, :]
        bottom = gray[100:, :]
        left = gray[:, :90]
        right = gray[:, 90:]

        face = [
            float(np.mean(top)),
            float(np.std(top)),
            float(np.mean(energy[:70, 40:140])),
            float(np.mean(np.abs(top[:, :90] - top[:, 90:]))),
            float(np.mean(gray[45:85, 60:120])),
        ]
        head = [
            float(np.mean(left) - np.mean(right)),
            float(np.mean(top[:40, :]) - np.mean(top[40:, :])),
            float(np.mean(energy[:80, :90]) - np.mean(energy[:80, 90:])),
        ]
        hand = [
            float(np.mean(energy[70:130, :45])),
            float(np.mean(energy[70:130, 135:])),
            float(np.mean(gray[70:130, :45])),
            float(np.mean(gray[70:130, 135:])),
            float(np.mean(energy[60:120, :])),
        ]
        body = [
            float(np.mean(bottom)),
            float(np.std(bottom)),
            float(np.mean(energy[110:, 50:130])),
            float(np.mean(mid[:, :90]) - np.mean(mid[:, 90:])),
            float(np.mean(gray[90:170, 70:110])),
        ]

        visibility = VisibilityFlags(
            face=face[1] > 0.03,
            head=True,
            hand=max(hand[0], hand[1]) > 0.02,
            body=body[1] > 0.02,
        )
        missing = [region for region, present in visibility.model_dump().items() if not present]
        snapshot = LandmarkSnapshot(
            face=face,
            head=head,
            hand=hand,
            body=body,
            anchors={
                "face_center": [0.5, 0.22],
                "left_hand_center": [0.18, 0.55],
                "right_hand_center": [0.82, 0.55],
                "body_center": [0.5, 0.72],
            },
            provider_name="heuristic-fallback",
        )
        normalized = PoseFeatures(
            face=self._normalize(face),
            head=self._normalize(head),
            hand=self._normalize(hand),
            body=self._normalize(body),
            missing_regions=missing,
            visibility_flags=visibility,
            landmark_snapshot=snapshot,
        )
        return normalized

    def build_landmark_snapshot(self, features: PoseFeatures) -> LandmarkSnapshot:
        snapshot = features.landmark_snapshot.model_copy(deep=True)
        snapshot.face = features.face
        snapshot.head = features.head
        snapshot.hand = features.hand
        snapshot.body = features.body
        snapshot.provider_name = snapshot.provider_name or self.provider_name
        return snapshot

    def _point_from_landmark(self, landmark) -> LandmarkPoint:
        return LandmarkPoint(
            x=round(float(landmark.x), 4),
            y=round(float(landmark.y), 4),
            z=round(float(getattr(landmark, "z", 0.0)), 4),
            visibility=round(float(getattr(landmark, "visibility", 0.0)), 4) if hasattr(landmark, "visibility") else None,
        )

    def _safe_distance(self, points: list[LandmarkPoint], idx_a: int, idx_b: int) -> float:
        if len(points) <= max(idx_a, idx_b):
            return 0.0
        a = points[idx_a]
        b = points[idx_b]
        return float(np.hypot(a.x - b.x, a.y - b.y))

    def _safe_angle(self, points: list[LandmarkPoint], idx_a: int, idx_b: int, idx_c: int) -> float:
        if len(points) <= max(idx_a, idx_b, idx_c):
            return 0.0
        a = np.array([points[idx_a].x, points[idx_a].y], dtype=np.float32)
        b = np.array([points[idx_b].x, points[idx_b].y], dtype=np.float32)
        c = np.array([points[idx_c].x, points[idx_c].y], dtype=np.float32)
        ba = a - b
        bc = c - b
        ba_norm = np.linalg.norm(ba) or 1.0
        bc_norm = np.linalg.norm(bc) or 1.0
        cosine = float(np.clip(np.dot(ba, bc) / (ba_norm * bc_norm), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine))) / 180.0

    def _face_features(self, face_points: list[LandmarkPoint]) -> list[float]:
        if not face_points:
            return []
        mouth_open = self._safe_distance(face_points, 13, 14)
        left_eye = self._safe_distance(face_points, 159, 145)
        right_eye = self._safe_distance(face_points, 386, 374)
        brow_left = self._safe_distance(face_points, 70, 159)
        brow_right = self._safe_distance(face_points, 300, 386)
        face_width = self._safe_distance(face_points, 234, 454)
        face_height = self._safe_distance(face_points, 10, 152)
        nose = face_points[1] if len(face_points) > 1 else face_points[0]
        return [
            nose.x,
            nose.y,
            mouth_open,
            left_eye,
            right_eye,
            brow_left,
            brow_right,
            face_width / (face_height or 1.0),
        ]

    def _head_features(self, face_points: list[LandmarkPoint], pose_points: list[LandmarkPoint]) -> list[float]:
        if face_points:
            left_eye = face_points[33]
            right_eye = face_points[263]
            nose = face_points[1]
            mouth = face_points[13]
            eye_mid_x = (left_eye.x + right_eye.x) / 2.0
            eye_mid_y = (left_eye.y + right_eye.y) / 2.0
            roll = np.arctan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x) / np.pi
            yaw = nose.x - eye_mid_x
            pitch = mouth.y - eye_mid_y
            return [float(yaw), float(pitch), float(roll)]
        if pose_points:
            return [pose_points[0].x, pose_points[0].y, pose_points[0].z]
        return []

    def _hand_features(
        self,
        left_hand_points: list[LandmarkPoint],
        right_hand_points: list[LandmarkPoint],
        pose_points: list[LandmarkPoint],
    ) -> list[float]:
        left = self._single_hand_features(left_hand_points)
        right = self._single_hand_features(right_hand_points)
        wrist_heights = []
        for idx in (15, 16):
            if len(pose_points) > idx:
                wrist_heights.append(pose_points[idx].y)
        return left + right + wrist_heights

    def _single_hand_features(self, hand_points: list[LandmarkPoint]) -> list[float]:
        if not hand_points:
            return [0.0, 0.0, 0.0, 0.0]
        wrist = hand_points[0]
        index_tip = hand_points[8]
        thumb_tip = hand_points[4]
        pinky_tip = hand_points[20]
        openness = self._safe_distance(hand_points, 8, 20)
        return [wrist.x, wrist.y, self._safe_distance(hand_points, 4, 8), openness]

    def _body_features(self, pose_points: list[LandmarkPoint]) -> list[float]:
        if not pose_points:
            return []
        shoulder_width = self._safe_distance(pose_points, 11, 12)
        hip_width = self._safe_distance(pose_points, 23, 24)
        shoulder_slope = pose_points[12].y - pose_points[11].y if len(pose_points) > 12 else 0.0
        torso_lean = ((pose_points[11].x + pose_points[12].x) / 2.0) - ((pose_points[23].x + pose_points[24].x) / 2.0)
        left_elbow = self._safe_angle(pose_points, 11, 13, 15)
        right_elbow = self._safe_angle(pose_points, 12, 14, 16)
        return [
            shoulder_width,
            hip_width,
            shoulder_slope,
            torso_lean,
            left_elbow,
            right_elbow,
            pose_points[11].y,
            pose_points[12].y,
        ]

    def _build_anchors(
        self,
        face_points: list[LandmarkPoint],
        pose_points: list[LandmarkPoint],
        left_hand_points: list[LandmarkPoint],
        right_hand_points: list[LandmarkPoint],
    ) -> dict[str, list[float]]:
        anchors: dict[str, list[float]] = {}
        if face_points:
            nose = face_points[1]
            anchors["face_center"] = [nose.x, nose.y]
        if left_hand_points:
            anchors["left_hand_center"] = [left_hand_points[0].x, left_hand_points[0].y]
        if right_hand_points:
            anchors["right_hand_center"] = [right_hand_points[0].x, right_hand_points[0].y]
        if len(pose_points) > 24:
            shoulder_center = [
                round((pose_points[11].x + pose_points[12].x) / 2.0, 4),
                round((pose_points[11].y + pose_points[12].y) / 2.0, 4),
            ]
            hip_center = [
                round((pose_points[23].x + pose_points[24].x) / 2.0, 4),
                round((pose_points[23].y + pose_points[24].y) / 2.0, 4),
            ]
            anchors["shoulder_center"] = shoulder_center
            anchors["hip_center"] = hip_center
            anchors["body_center"] = [
                round((shoulder_center[0] + hip_center[0]) / 2.0, 4),
                round((shoulder_center[1] + hip_center[1]) / 2.0, 4),
            ]
        return anchors

    def _normalize(self, values: list[float]) -> list[float]:
        if not values:
            return []
        arr = np.asarray(values, dtype=np.float32)
        mean = float(arr.mean())
        std = float(arr.std()) or 1.0
        normalized = (arr - mean) / std
        clipped = np.clip(normalized, -2.0, 2.0)
        return [round(float(v), 4) for v in clipped]

    @property
    def provider_name(self) -> str:
        return "mediapipe" if self._mediapipe_available else "heuristic-fallback"
