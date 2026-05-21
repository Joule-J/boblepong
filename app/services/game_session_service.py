from __future__ import annotations

from pathlib import Path
from random import choice
from uuid import uuid4

from app.models import (
    CompletionResponse,
    FrameAnalysisDebug,
    FrameAnalysisResponse,
    GameSession,
    GameSessionCreate,
    SessionFrameRecord,
    SessionSummary,
    utc_now,
)
from app.storage import JsonStore
from app.services.meme_card_service import MemeCardService
from app.services.reference_pose_service import ReferencePoseService
from app.services.similarity_scoring_service import SimilarityScoringService
from app.services.vision_analysis_service import VisionAnalysisService


class GameSessionService:
    def __init__(
        self,
        store: JsonStore,
        reference_service: ReferencePoseService,
        vision_service: VisionAnalysisService,
        scoring_service: SimilarityScoringService,
        meme_card_service: MemeCardService,
    ) -> None:
        self.store = store
        self.reference_service = reference_service
        self.vision_service = vision_service
        self.scoring_service = scoring_service
        self.meme_card_service = meme_card_service

    def create_session(self, payload: GameSessionCreate) -> GameSession:
        references = self.reference_service.list_references()
        if not references:
            raise ValueError("Create and extract at least one reference first.")
        target_id = payload.target_reference_id or choice(references).reference_id
        session = GameSession(session_id=str(uuid4()), mode=payload.mode, target_reference_id=target_id)
        sessions = self.store.load_many(GameSession)
        sessions.append(session)
        self.store.save_many(sessions)
        return session

    def analyze_frame(self, session_id: str, frame_data_url: str) -> FrameAnalysisResponse:
        sessions = self.store.load_many(GameSession)
        session = self._require_session(session_id, sessions)
        references = [ref for ref in self.reference_service.list_references() if ref.feature_vector]
        if not references:
            raise ValueError("No extracted references available.")
        features = self.vision_service.extract_from_data_url(frame_data_url)
        scoring = self.scoring_service.score_against_references(features, references)
        if scoring is None:
            raise ValueError("No references available.")

        score_history = session.score_history + [SessionFrameRecord(total_score=scoring.total_score)]
        progress = self._compute_hold_progress(score_history, self._target_reference(session).success_threshold, session)
        analysis = FrameAnalysisResponse(
            matched_reference_id=scoring.reference.reference_id,
            matched_motion_package_id=scoring.reference.motion_package_id,
            matched_pattern_name=scoring.reference.pattern_name,
            total_score=scoring.total_score,
            region_scores=scoring.region_scores,
            feedback_hints=scoring.feedback_hints,
            confidence=scoring.confidence,
            hold_progress=progress,
            debug=FrameAnalysisDebug(
                top_candidates=scoring.top_candidates,
                normalized_features={
                    "face": features.face,
                    "head": features.head,
                    "hand": features.hand,
                    "body": features.body,
                },
                missing_regions=features.missing_regions,
                visibility_flags=features.visibility_flags,
                capture_anchors=features.landmark_snapshot.anchors,
                provider_name=features.landmark_snapshot.provider_name or self.vision_service.provider_name,
            ),
        )

        updated = session.model_copy(
            update={
                "score_history": score_history[-12:],
                "best_score": max(session.best_score, scoring.total_score),
                "best_frame_data_url": frame_data_url if scoring.total_score >= session.best_score else session.best_frame_data_url,
                "last_analysis": analysis,
                "updated_at": utc_now(),
            }
        )
        self._replace_session(updated, sessions)
        self.store.save_many(sessions)
        return analysis

    def complete_session(self, session_id: str) -> CompletionResponse:
        sessions = self.store.load_many(GameSession)
        session = self._require_session(session_id, sessions)
        target = self._target_reference(session)
        last_analysis = session.last_analysis
        progress = last_analysis.hold_progress if last_analysis else 0.0
        success = session.best_score >= target.success_threshold and progress >= 100.0
        result = self.meme_card_service.build_card(target, session.best_frame_data_url, session.best_score) if success else None
        updated = session.model_copy(update={"status": "completed", "result": result, "updated_at": utc_now()})
        self._replace_session(updated, sessions)
        self.store.save_many(sessions)
        message = "Session completed with a meme card." if success else "Session completed, but threshold was not sustained."
        return CompletionResponse(
            session=self._build_summary(updated),
            result=result,
            message=message,
        )

    def get_session(self, session_id: str) -> GameSession:
        return self._require_session(session_id, self.store.load_many(GameSession))

    def get_result(self, session_id: str) -> CompletionResponse:
        session = self.get_session(session_id)
        message = "Result ready." if session.result else "Result not ready yet."
        return CompletionResponse(
            session=self._build_summary(session),
            result=session.result,
            message=message,
        )

    def _build_summary(self, session: GameSession) -> SessionSummary:
        return SessionSummary(
            session_id=session.session_id,
            status=session.status,
            target_reference_id=session.target_reference_id,
            mode=session.mode,
            best_score=session.best_score,
            hold_progress=session.last_analysis.hold_progress if session.last_analysis else 0.0,
        )

    def _compute_hold_progress(self, history: list[SessionFrameRecord], threshold: float, session: GameSession) -> float:
        sustained = 0
        for item in reversed(history):
            if item.total_score < threshold:
                break
            sustained += 1
        hold_seconds = sustained * session.frame_interval_seconds
        progress = min(100.0, (hold_seconds / session.hold_seconds_required) * 100.0)
        return round(progress, 2)

    def _target_reference(self, session: GameSession):
        return self.reference_service.get_reference(session.target_reference_id)

    def _require_session(self, session_id: str, sessions: list[GameSession]) -> GameSession:
        for session in sessions:
            if session.session_id == session_id:
                return session
        raise KeyError(session_id)

    def _replace_session(self, updated: GameSession, sessions: list[GameSession]) -> None:
        for idx, session in enumerate(sessions):
            if session.session_id == updated.session_id:
                sessions[idx] = updated
                return
        sessions.append(updated)
