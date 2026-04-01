"""Response validation and relevance enforcement.

Uses fast heuristic checks (no heavy embedding models) to validate output
quality in real-time, with an optional slow semantic scoring path for
evaluation and metrics only.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from config import config
from context_analyzer.analyzer import ContextFeatures

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of response validation checks."""

    is_relevant: bool
    relevance_score: float
    keyword_overlap: float
    answers_question: bool
    is_generic: bool
    is_garbage: bool
    explanation: str

    @property
    def passed(self) -> bool:
        return self.is_relevant and not self.is_generic and not self.is_garbage


GENERIC_RESPONSES = [
    "i'm not sure",
    "that's interesting",
    "tell me more",
    "i see",
    "that's nice",
    "i don't know",
    "hmm",
    "okay",
    "sure",
    "right",
    "indeed",
    "absolutely",
]

HOSTILE_PHRASES = [
    "you are so lazy", "you're so lazy", "you are stupid", "you're stupid",
    "you are dumb", "you're dumb", "you are an idiot", "you're an idiot",
    "you are useless", "you're useless", "your fault", "you suck",
    "you are wrong", "you're wrong about everything",
    "you deserve", "you should be ashamed", "what is wrong with you",
    "how dare you", "shut up", "go away", "leave me alone",
    "nobody cares", "no one cares", "get over it", "stop complaining",
    "stop whining", "grow up", "man up",
]

PROFANITY_PATTERNS = [
    r"\bfuck\w*\b", r"\bshit\w*\b", r"\bdamn\w*\b", r"\bbitch\w*\b",
    r"\bass\b", r"\basshole\w*\b", r"\bbastard\w*\b", r"\bcrap\b",
    r"\bdick\b", r"\bpiss\w*\b",
]


class RelevanceChecker:
    """Validates response quality using fast heuristics only.

    Embedding-based semantic scoring is available via ``score_relevance_slow``
    for evaluation pipelines but is NOT used in the real-time generation loop.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or "cpu"
        self._embedder = None

    def validate(
        self,
        query: str,
        response: str,
        context: Optional[ContextFeatures] = None,
    ) -> ValidationResult:
        """Fast validation using heuristics only (no embedding models)."""
        is_garbage = self._is_garbage(response, query, context)

        if is_garbage:
            return ValidationResult(
                is_relevant=False,
                relevance_score=0.0,
                keyword_overlap=0.0,
                answers_question=False,
                is_generic=False,
                is_garbage=True,
                explanation="FAILED: Response is incoherent or low quality",
            )

        keyword_overlap = self._keyword_overlap(query, response, context)
        answers_q = self._answers_question(query, response, context)
        is_generic = self._is_generic(response)

        relevance_score = self._heuristic_relevance(
            query, response, keyword_overlap, answers_q, context,
        )

        is_relevant = relevance_score >= 0.3 and not is_generic

        explanation = self._build_explanation(
            relevance_score, keyword_overlap, answers_q, is_generic,
        )

        return ValidationResult(
            is_relevant=is_relevant,
            relevance_score=relevance_score,
            keyword_overlap=keyword_overlap,
            answers_question=answers_q,
            is_generic=is_generic,
            is_garbage=False,
            explanation=explanation,
        )

    def _is_garbage(
        self,
        response: str,
        query: str = "",
        context: Optional[ContextFeatures] = None,
    ) -> bool:
        """Comprehensive quality gate catching all forms of bad output."""
        if not response or len(response.strip()) < 10:
            return True

        alpha_chars = [c for c in response if c.isalpha()]
        if alpha_chars:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if upper_ratio > 0.4 and len(alpha_chars) > 15:
                return True

        response_lower = response.lower()
        profanity_count = sum(
            1 for p in PROFANITY_PATTERNS if re.search(p, response_lower)
        )
        if profanity_count >= 1:
            return True

        for phrase in HOSTILE_PHRASES:
            if phrase in response_lower:
                return True

        exclaim = response.count("!")
        question = response.count("?")
        if exclaim > 8 or question > 8 or (exclaim + question) > 12:
            return True

        multi_punct = len(re.findall(r"[!?]{3,}", response))
        if multi_punct >= 3:
            return True

        words = response.split()
        if len(words) > 8:
            word_list = [w.lower().strip(".,!?;:'\"()") for w in words]
            unique = set(word_list)
            if len(unique) / len(word_list) < 0.25:
                return True

        if len(words) > 15:
            sentences = re.split(r"[.!?]+", response)
            real = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
            if len(real) < 1:
                return True

        if len(response) > 30:
            non_alpha = sum(
                1 for c in response
                if not c.isalpha() and not c.isspace() and c not in ".,!?;:'-\"()"
            )
            if non_alpha / len(response) > 0.15:
                return True

        topic_shifts = 0
        sentences = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]
        if len(sentences) >= 4:
            for i in range(1, len(sentences)):
                prev_words = set(sentences[i - 1].lower().split())
                curr_words = set(sentences[i].lower().split())
                content_prev = {w for w in prev_words if len(w) > 3}
                content_curr = {w for w in curr_words if len(w) > 3}
                if content_prev and content_curr and not (content_prev & content_curr):
                    topic_shifts += 1
            if topic_shifts >= len(sentences) - 1 and len(sentences) >= 4:
                return True

        if len(words) > 300:
            return True

        return False

    def _heuristic_relevance(
        self,
        query: str,
        response: str,
        keyword_overlap: float,
        answers_q: bool,
        context: Optional[ContextFeatures] = None,
    ) -> float:
        """Fast relevance scoring without embeddings."""
        score = 0.3

        score += keyword_overlap * 0.3

        if answers_q:
            score += 0.1

        query_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()))
        response_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", response.lower()))
        if query_words:
            direct_overlap = len(query_words & response_words) / len(query_words)
            score += direct_overlap * 0.2

        resp_len = len(response.split())
        if 10 <= resp_len <= 50:
            score += 0.1
        elif resp_len < 5:
            score -= 0.2

        return min(max(score, 0.0), 1.0)

    def _keyword_overlap(
        self,
        query: str,
        response: str,
        context: Optional[ContextFeatures] = None,
    ) -> float:
        if context and context.keywords:
            keywords = set(context.keywords)
        else:
            stop = {"the", "and", "for", "are", "but", "not", "you", "all",
                    "can", "had", "her", "was", "one", "our", "out", "has",
                    "have", "just", "been", "like", "will", "with", "this",
                    "that", "from", "they", "what", "about", "would"}
            keywords = {
                w for w in re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
                if w not in stop
            }

        if not keywords:
            return 0.5

        response_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", response.lower()))
        overlap = keywords & response_words
        return len(overlap) / len(keywords)

    def _answers_question(
        self,
        query: str,
        response: str,
        context: Optional[ContextFeatures] = None,
    ) -> bool:
        is_question = (context.is_question if context else False) or "?" in query
        if not is_question:
            return True

        response_lower = response.lower().strip()
        if len(response_lower.split()) < 5:
            return False
        if any(response_lower.startswith(g) for g in GENERIC_RESPONSES):
            return False
        return True

    def _is_generic(self, response: str) -> bool:
        response_lower = response.lower().strip()
        if len(response_lower.split()) < 4:
            return True
        for generic in GENERIC_RESPONSES:
            if response_lower == generic or response_lower == generic + ".":
                return True
        return False

    def _build_explanation(
        self,
        relevance: float,
        keyword_overlap: float,
        answers_q: bool,
        is_generic: bool,
    ) -> str:
        parts = [f"Relevance: {relevance:.2f}", f"Keyword overlap: {keyword_overlap:.2f}"]
        if not answers_q:
            parts.append("Response may not answer the question")
        if is_generic:
            parts.append("Response appears generic")
        if relevance >= 0.3 and not is_generic:
            parts.append("PASSED")
        else:
            parts.append("FAILED")
        return " | ".join(parts)

    def score_relevance(self, query: str, response: str) -> float:
        """Fast relevance score for ranking."""
        if self._is_garbage(response, query):
            return 0.0
        kw = self._keyword_overlap(query, response)
        return self._heuristic_relevance(query, response, kw, True)

    def score_relevance_slow(self, query: str, response: str) -> float:
        """Embedding-based relevance (for evaluation only, not real-time)."""
        import torch
        from sentence_transformers import SentenceTransformer

        if self._embedder is None:
            self._embedder = SentenceTransformer(
                config.models.sentence_transformer, device=self.device
            )
        embeddings = self._embedder.encode(
            [query, response], convert_to_tensor=True, show_progress_bar=False
        )
        sim = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
        )
        return float(sim.item())
