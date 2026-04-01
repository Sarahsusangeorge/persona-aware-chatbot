"""Context understanding module.

Extracts intent, emotion, keywords, and entity features from user queries
using fast regex/heuristic methods only (no heavy model loading).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from config import config

logger = logging.getLogger(__name__)


@dataclass
class ContextFeatures:
    """Structured representation of extracted context from a user query."""

    intent: str = "unknown"
    intent_confidence: float = 0.0
    emotion: str = "neutral"
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    is_question: bool = False
    topic: str = ""
    query_embedding: object = None

    def prompt_block(self) -> str:
        parts = [f"Intent: {self.intent}"]
        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")
        if self.entities:
            parts.append(f"Entities: {', '.join(self.entities)}")
        parts.append(f"Emotion: {self.emotion}")
        if self.topic:
            parts.append(f"Topic: {self.topic}")
        return ". ".join(parts)


INTENT_PATTERNS: dict[str, list[str]] = {
    "question": [
        r"^(what|who|where|when|why|how|which|can|could|would|should|is|are|do|does|did|will|has|have)\b",
        r"\?\s*$",
    ],
    "request": [
        r"^(please|can you|could you|would you|help me|i need|i want|give me|show me|tell me|explain)",
        r"^(write|create|generate|make|build|design|draft|compose|prepare)\b",
    ],
    "opinion": [
        r"^(what do you think|do you think|how do you feel|your opinion|thoughts on)",
        r"(should i|would you recommend|what.+best|which.+better)",
    ],
    "greeting": [
        r"^(hi|hello|hey|good morning|good afternoon|good evening|howdy|greetings)\b",
        r"^(how are you|how's it going|what's up|sup)\b",
    ],
    "complaint": [
        r"(not working|doesn't work|broken|terrible|worst|hate|frustrated|annoyed)",
        r"(problem with|issue with|bug|error|wrong|failed|can't|cannot)",
    ],
    "gratitude": [
        r"^(thanks|thank you|thx|ty|appreciate|grateful)\b",
    ],
    "farewell": [
        r"^(bye|goodbye|see you|take care|good night|talk later|gotta go)\b",
    ],
    "instruction": [
        r"^(do|make|set|change|update|fix|add|remove|delete|edit|modify|configure)\b",
    ],
    "emotional_sharing": [
        r"(i feel|i'm feeling|i am feeling|makes me feel|i've been feeling)",
        r"(i'm (sad|happy|angry|scared|worried|anxious|stressed|depressed|lonely|overwhelmed))",
    ],
    "information": [
        r"(tell me about|explain|describe|what is|define|meaning of|difference between)",
    ],
}

EMOTION_LEXICON: dict[str, list[str]] = {
    "happy": ["happy", "glad", "excited", "thrilled", "joyful", "wonderful", "awesome",
              "great", "amazing", "fantastic", "love", "enjoy", "fun", "celebrate"],
    "sad": ["sad", "unhappy", "depressed", "down", "miserable", "heartbroken",
            "disappointed", "grief", "loss", "miss", "lonely", "cry", "tear"],
    "angry": ["angry", "furious", "mad", "annoyed", "irritated", "frustrated",
              "outraged", "hate", "terrible", "worst", "ridiculous", "unacceptable"],
    "fearful": ["scared", "afraid", "worried", "anxious", "nervous", "terrified",
                "panic", "stress", "overwhelmed", "dread", "uneasy"],
    "surprised": ["surprised", "shocked", "amazed", "astonished", "wow",
                  "unexpected", "unbelievable", "incredible", "can't believe"],
    "disgusted": ["disgusted", "gross", "revolting", "sick", "eww", "awful",
                  "horrible", "nasty", "repulsive"],
}

ENTITY_PATTERNS: list[tuple[str, str]] = [
    (r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b", "proper_noun"),
    (r"\b\d{1,2}[:/]\d{2}\b", "time"),
    (r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", "date"),
    (r"\b\d+(?:\.\d+)?%\b", "percentage"),
    (r"\$\d+(?:,\d{3})*(?:\.\d{2})?\b", "money"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
]


class ContextAnalyzer:
    """Extracts context from user queries using fast regex/heuristics.

    No heavy models are loaded during ``analyze()`` — this runs in
    milliseconds using only pattern matching and keyword lookups.
    """

    def analyze(self, query: str, conversation_history: Optional[list[str]] = None) -> ContextFeatures:
        """Fast context extraction (regex only, no model loading)."""
        features = ContextFeatures()

        features.intent, features.intent_confidence = self._classify_intent(query)
        features.emotion = self._detect_emotion(query)
        features.keywords = self._extract_keywords(query, conversation_history)
        features.entities = self._extract_entities(query)
        features.is_question = features.intent in ("question", "opinion") or "?" in query
        features.topic = self._extract_topic(query, features.keywords)

        return features

    def _classify_intent(self, query: str) -> tuple[str, float]:
        query_lower = query.lower().strip()
        scores: dict[str, int] = {}

        for intent, patterns in INTENT_PATTERNS.items():
            score = sum(
                1 for p in patterns if re.search(p, query_lower, re.IGNORECASE)
            )
            if score > 0:
                scores[intent] = score

        if not scores:
            return "statement", 0.5

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        total = sum(scores.values())
        confidence = scores[best] / total if total > 0 else 0.5
        return best, min(confidence, 1.0)

    def _detect_emotion(self, query: str) -> str:
        query_lower = query.lower()
        scores: dict[str, int] = {}

        for emotion, words in EMOTION_LEXICON.items():
            scores[emotion] = sum(1 for w in words if w in query_lower)

        if not any(scores.values()):
            return "neutral"

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best if scores[best] > 0 else "neutral"

    def _extract_keywords(
        self, query: str, history: Optional[list[str]] = None
    ) -> list[str]:
        stop_words = {
            "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she",
            "it", "its", "they", "them", "what", "which", "who", "whom", "this",
            "that", "these", "those", "am", "is", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "shall", "may", "might", "can", "a",
            "an", "the", "and", "but", "if", "or", "not", "no", "nor", "so",
            "too", "very", "just", "about", "to", "from", "in", "on", "at",
            "for", "with", "of", "by", "as", "into", "like", "how", "when",
            "where", "why", "all", "each", "some", "any", "few", "more", "most",
            "than", "also", "only", "then", "there", "here", "now", "up", "out",
            "over", "after", "before", "between", "under", "again", "hi",
            "hello", "hey", "please", "thanks", "thank", "yes", "no", "ok",
            "okay", "sure", "well", "really", "get", "got", "going", "go",
            "know", "think", "want", "need", "tell", "say", "said",
        }

        words = re.findall(r"\b[a-zA-Z]{2,}\b", query.lower())
        keywords = [w for w in words if w not in stop_words]

        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)

        return unique[: config.context.max_keyword_count]

    def _extract_entities(self, query: str) -> list[str]:
        entities = []
        for pattern, _label in ENTITY_PATTERNS:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        return entities[:10]

    def _extract_topic(self, query: str, keywords: list[str]) -> str:
        if not keywords:
            return ""
        important = [kw for kw in keywords if len(kw) > 3]
        return " ".join(important[:3]) if important else keywords[0]

    def compute_query_response_similarity(
        self, query: str, response: str
    ) -> float:
        """Cosine similarity between query and response embeddings.

        Lazy-loads sentence-transformers only when called (evaluation only).
        """
        import torch
        from sentence_transformers import SentenceTransformer

        if not hasattr(self, "_embedder") or self._embedder is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._embedder = SentenceTransformer(
                config.models.sentence_transformer, device=device
            )
        embeddings = self._embedder.encode(
            [query, response], convert_to_tensor=True, show_progress_bar=False
        )
        sim = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
        )
        return float(sim.item())
