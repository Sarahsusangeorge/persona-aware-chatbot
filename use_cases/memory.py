"""Conversation memory system for long-running interactions.

Maintains short-term (recent turns) and long-term (summarized facts) memory
to keep the model contextually grounded across extended conversations.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from config import config

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memorable fact extracted from the conversation."""

    fact: str
    source_turn: int
    importance: float = 0.5


class ConversationMemory:
    """Manages short-term and long-term conversation memory."""

    def __init__(
        self,
        max_short_term: int | None = None,
        max_long_term: int | None = None,
        summary_interval: int | None = None,
    ):
        self.max_short_term = max_short_term or config.memory.max_short_term
        self.max_long_term = max_long_term or config.memory.max_long_term
        self.summary_interval = summary_interval or config.memory.summary_interval

        self.short_term: deque[dict[str, str]] = deque(maxlen=self.max_short_term)
        self.long_term: list[MemoryEntry] = []
        self.turn_count: int = 0
        self.user_facts: dict[str, str] = {}

    def add_turn(self, role: str, content: str) -> None:
        """Record a new conversational turn."""
        self.short_term.append({"role": role, "content": content})
        self.turn_count += 1

        if role == "user":
            self._extract_user_facts(content)

        if self.turn_count % self.summary_interval == 0 and self.turn_count > 0:
            self._consolidate()

    def _extract_user_facts(self, text: str) -> None:
        """Heuristic extraction of memorable facts from user messages."""
        text_lower = text.lower()

        preference_signals = [
            ("name", r"(?:my name is|i'm called|call me) (\w+)"),
            ("likes", r"(?:i (?:like|love|enjoy|prefer)) (.+?)(?:\.|$|,)"),
            ("dislikes", r"(?:i (?:hate|dislike|can't stand|don't like)) (.+?)(?:\.|$|,)"),
            ("occupation", r"(?:i (?:work as|am a|work in)) (.+?)(?:\.|$|,)"),
            ("hobby", r"(?:i (?:do|play|practice)) (.+?)(?:\.|$|,)"),
            ("location", r"(?:i (?:live in|am from|come from)) (.+?)(?:\.|$|,)"),
        ]

        import re
        for key, pattern in preference_signals:
            match = re.search(pattern, text_lower)
            if match:
                value = match.group(1).strip()
                if value and len(value) > 1:
                    self.user_facts[key] = value
                    self.long_term.append(
                        MemoryEntry(
                            fact=f"User {key}: {value}",
                            source_turn=self.turn_count,
                            importance=0.8,
                        )
                    )

    def _consolidate(self) -> None:
        """Summarize older short-term turns into long-term memory entries."""
        if len(self.short_term) < 4:
            return

        older_turns = list(self.short_term)[:len(self.short_term) // 2]
        topics = set()
        for turn in older_turns:
            words = turn["content"].lower().split()
            important = [w for w in words if len(w) > 4]
            topics.update(important[:3])

        if topics:
            summary = f"Earlier discussion covered: {', '.join(list(topics)[:5])}"
            self.long_term.append(
                MemoryEntry(
                    fact=summary,
                    source_turn=self.turn_count,
                    importance=0.4,
                )
            )

        if len(self.long_term) > self.max_long_term:
            self.long_term.sort(key=lambda e: e.importance, reverse=True)
            self.long_term = self.long_term[: self.max_long_term]

    def get_context_window(self, max_turns: int = 10) -> list[dict[str, str]]:
        """Return the most recent turns for prompt construction."""
        return list(self.short_term)[-max_turns:]

    def get_long_term_summary(self) -> str:
        """Format long-term memory as a text block for prompt injection."""
        if not self.long_term and not self.user_facts:
            return ""

        parts = []
        if self.user_facts:
            facts = [f"- {k}: {v}" for k, v in self.user_facts.items()]
            parts.append("Known about user:\n" + "\n".join(facts))

        important = sorted(
            self.long_term, key=lambda e: e.importance, reverse=True
        )[:5]
        if important:
            summaries = [f"- {e.fact}" for e in important]
            parts.append("Conversation history notes:\n" + "\n".join(summaries))

        return "\n".join(parts)

    def clear(self) -> None:
        self.short_term.clear()
        self.long_term.clear()
        self.user_facts.clear()
        self.turn_count = 0

    def __len__(self) -> int:
        return self.turn_count
