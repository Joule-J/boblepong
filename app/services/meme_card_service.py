from __future__ import annotations

from app.models import MemeCard, ReferencePoseCard


class MemeCardService:
    def build_card(
        self,
        reference: ReferencePoseCard,
        best_frame_data_url: str | None,
        score: float,
    ) -> MemeCard:
        caption = self._caption_for(reference, score)
        return MemeCard(
            target_reference_id=reference.reference_id,
            motion_package_id=reference.motion_package_id,
            target_image_path=reference.image_path,
            best_frame_data_url=best_frame_data_url,
            pattern_name=reference.pattern_name,
            caption=caption,
            score=round(score, 2),
            share_text=f"I matched {reference.pattern_name} at {round(score, 1)}% in SpongeBob Meme Pose Game.",
        )

    def _caption_for(self, reference: ReferencePoseCard, score: float) -> str:
        templates = [
            f"{reference.pattern_name} unlocked. SpongeBob would approve this chaos.",
            f"{reference.pattern_name} energy captured at {round(score)}%.",
            f"You hit {reference.pattern_name} mode and the room felt it.",
        ]
        return templates[int(score) % len(templates)]
