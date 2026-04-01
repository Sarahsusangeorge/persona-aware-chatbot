from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class PersonaProfile(BaseModel):
    """Structured representation of a user's personality extracted from chat history."""

    personality_traits: list[str] = Field(
        default_factory=list,
        description="Dominant personality traits (e.g. friendly, analytical, sarcastic)",
    )
    tone_preference: str = Field(
        default="neutral",
        description="Dominant conversational tone",
    )
    communication_style: str = Field(
        default="balanced",
        description="Style descriptor such as verbose, concise, casual, or formal",
    )
    emotional_tendency: str = Field(
        default="neutral",
        description="Emotional leaning (e.g. optimistic, empathetic, reserved)",
    )
    summary: str = Field(
        default="",
        description="Natural-language persona description for prompt injection",
    )
    raw_history: list[str] = Field(
        default_factory=list,
        description="Source chat messages used to derive this profile",
    )

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Persist the profile as a JSON file. Creates parent dirs if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> PersonaProfile:
        """Load a profile from a JSON file."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.model_validate_json(text)

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> PersonaProfile:
        return cls.model_validate(data)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def prompt_block(self) -> str:
        """Format the profile as a text block suitable for prompt injection."""
        if self.summary:
            return self.summary
        parts = []
        if self.personality_traits:
            parts.append(f"Traits: {', '.join(self.personality_traits)}")
        parts.append(f"Tone: {self.tone_preference}")
        parts.append(f"Style: {self.communication_style}")
        parts.append(f"Emotional tendency: {self.emotional_tendency}")
        return ". ".join(parts)

    @classmethod
    def default(cls) -> PersonaProfile:
        """Return a neutral fallback profile."""
        return cls(
            personality_traits=["neutral"],
            tone_preference="neutral",
            communication_style="balanced",
            emotional_tendency="neutral",
            summary="A balanced, neutral conversationalist.",
        )
