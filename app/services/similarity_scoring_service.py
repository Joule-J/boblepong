from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest

from app.models import CandidateMatch, PoseFeatures, ReferencePoseCard, RegionScores


@dataclass
class SimilarityResult:
    reference: ReferencePoseCard
    total_score: float
    region_scores: RegionScores
    top_candidates: list[CandidateMatch]
    confidence: float
    feedback_hints: list[str]


class SimilarityScoringService:
    def score_against_references(
        self, features: PoseFeatures, references: list[ReferencePoseCard]
    ) -> SimilarityResult | None:
        if not references:
            return None

        ranked: list[tuple[ReferencePoseCard, float, RegionScores]] = []
        for reference in references:
            region_scores = RegionScores(
                face=self._score_region(features.face, reference.landmark_snapshot.face),
                head=self._score_region(features.head, reference.landmark_snapshot.head),
                hand=self._score_region(features.hand, reference.landmark_snapshot.hand),
                body=self._score_region(features.body, reference.landmark_snapshot.body),
            )
            total_score = self._weighted_score(region_scores, reference)
            ranked.append((reference, total_score, region_scores))

        ranked.sort(key=lambda item: item[1], reverse=True)
        best_reference, best_score, best_regions = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = round(max(0.0, min(100.0, best_score - runner_up + 50.0)), 2)
        top_candidates = [
            CandidateMatch(
                reference_id=reference.reference_id,
                motion_package_id=reference.motion_package_id,
                pattern_name=reference.pattern_name,
                total_score=round(score, 2),
            )
            for reference, score, _ in ranked[:3]
        ]
        hints = self._build_hints(features, best_reference, best_regions)
        return SimilarityResult(
            reference=best_reference,
            total_score=round(best_score, 2),
            region_scores=best_regions,
            top_candidates=top_candidates,
            confidence=confidence,
            feedback_hints=hints,
        )

    def _score_region(self, current: list[float], target: list[float]) -> float:
        if not current or not target:
            return 0.0
        deltas = [abs(a - b) for a, b in zip_longest(current, target, fillvalue=0.0)]
        avg_delta = sum(deltas) / len(deltas)
        score = max(0.0, 100.0 - (avg_delta * 32.0))
        return round(score, 2)

    def _weighted_score(self, scores: RegionScores, reference: ReferencePoseCard) -> float:
        weights = reference.region_weights
        total_weight = weights.face + weights.head + weights.hand + weights.body
        if total_weight <= 0:
            total_weight = 1.0
        weighted_sum = (
            scores.face * weights.face
            + scores.head * weights.head
            + scores.hand * weights.hand
            + scores.body * weights.body
        )
        return weighted_sum / total_weight

    def _build_hints(
        self, features: PoseFeatures, reference: ReferencePoseCard, region_scores: RegionScores
    ) -> list[str]:
        hints = []
        defaults = {
            "face": "Mimiğini hedefe daha çok yaklaştır.",
            "head": "Kafa açını biraz değiştir.",
            "hand": "Ellerini daha görünür ve belirgin tut.",
            "body": "Gövde ve omuz duruşunu hedefe yaklaştır.",
        }
        template_map = reference.hint_templates or {}
        for region in ("face", "head", "hand", "body"):
            if region in features.missing_regions:
                hints.append(f"{region.title()} bölgesi görünmüyor, kameraya daha net gir.")
                continue
            score = getattr(region_scores, region)
            if score >= 80:
                continue
            region_templates = template_map.get(region) or []
            hints.append(region_templates[0] if region_templates else defaults[region])
        return hints[:3]
