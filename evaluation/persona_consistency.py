"""Persona-consistency and engagement scoring.

* **Persona Consistency**: cosine similarity between the persona profile
  embedding and the generated response embedding (via sentence-transformers).
* **Engagement Score**: composite of response length, question-asking rate,
  and vocabulary diversity.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer

from config import config
from persona_generator.persona_schema import PersonaProfile

logger = logging.getLogger(__name__)


class PersonaConsistencyScorer:
    """Measures how well a response aligns with a persona profile."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or config.models.sentence_transformer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _embed(self, texts: list[str]) -> torch.Tensor:
        return self.model.encode(
            texts, convert_to_tensor=True, show_progress_bar=False
        )

    def score(self, persona: PersonaProfile, response: str) -> float:
        """Cosine similarity between persona description and response (0-1)."""
        persona_text = persona.prompt_block()
        embeddings = self._embed([persona_text, response])
        cos_sim = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
        )
        return float(cos_sim.item())

    def score_batch(
        self,
        persona: PersonaProfile,
        responses: list[str],
    ) -> list[float]:
        """Score multiple responses against the same persona."""
        persona_text = persona.prompt_block()
        all_texts = [persona_text] + responses
        embeddings = self._embed(all_texts)

        persona_emb = embeddings[0].unsqueeze(0)
        response_embs = embeddings[1:]

        sims = torch.nn.functional.cosine_similarity(
            persona_emb, response_embs
        )
        return sims.tolist()


# ── Engagement scoring ───────────────────────────────────────────────

def _question_rate(text: str) -> float:
    """Fraction of sentences that are questions."""
    sentences = [s.strip() for s in text.replace("!", ".").split(".") if s.strip()]
    if not sentences:
        return 0.0
    questions = sum(1 for s in text.split("?") if s.strip()) - 1
    questions = max(0, text.count("?"))
    return min(questions / max(len(sentences), 1), 1.0)


def _vocabulary_diversity(text: str) -> float:
    """Type-token ratio (unique words / total words), capped at 1.0."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _length_score(text: str, ideal_min: int = 20, ideal_max: int = 200) -> float:
    """Score based on response length — rewards the sweet-spot range."""
    n = len(text.split())
    if n < ideal_min:
        return n / ideal_min
    if n > ideal_max:
        return max(0.5, 1.0 - (n - ideal_max) / ideal_max)
    return 1.0


def engagement_score(text: str) -> dict[str, float]:
    """Compute a composite engagement score and its components.

    Components (each 0-1):
    - ``length``: penalises too-short or too-long responses.
    - ``question_rate``: higher if the response asks questions.
    - ``vocabulary_diversity``: type-token ratio.
    - ``composite``: weighted average.
    """
    length = _length_score(text)
    questions = _question_rate(text)
    diversity = _vocabulary_diversity(text)

    composite = 0.4 * length + 0.3 * questions + 0.3 * diversity

    return {
        "length": round(length, 4),
        "question_rate": round(questions, 4),
        "vocabulary_diversity": round(diversity, 4),
        "composite": round(composite, 4),
    }
