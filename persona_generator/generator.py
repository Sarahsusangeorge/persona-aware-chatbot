from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import config
from persona_generator.persona_schema import PersonaProfile

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "Analyze the following chat history and extract a personality profile.\n"
    "Chat history:\n{history}\n\n"
    "Provide the following in a structured format:\n"
    "Personality traits: <comma-separated list>\n"
    "Tone: <dominant tone>\n"
    "Communication style: <style description>\n"
    "Emotional tendency: <emotional leaning>\n"
    "Summary: <one-sentence persona description>"
)


class PersonaGenerator:
    """Extracts a structured persona profile from chat history using flan-t5-small."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.model_name = model_name or config.models.persona_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir) if cache_dir else config.paths.personas_dir

        self._tokenizer = None
        self._model = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(
                self.device
            )
            self._model.eval()
        return self._model

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        chat_history: list[str],
        use_cache: bool = True,
    ) -> PersonaProfile:
        """Generate a PersonaProfile from a list of chat messages.

        Args:
            chat_history: Ordered list of messages from the conversation.
            use_cache: If True, return a cached profile when the same history
                       has been analysed before.

        Returns:
            A populated PersonaProfile instance.
        """
        if use_cache:
            cached = self._load_cached(chat_history)
            if cached is not None:
                logger.info("Returning cached persona profile.")
                return cached

        prompt = self._build_prompt(chat_history)
        raw_output = self._run_model(prompt)
        profile = self._parse_output(raw_output, chat_history)

        if use_cache:
            self._save_cached(profile, chat_history)

        return profile

    def update(
        self,
        existing_profile: PersonaProfile,
        new_messages: list[str],
    ) -> PersonaProfile:
        """Re-analyse with additional messages appended to the existing history."""
        combined = existing_profile.raw_history + new_messages
        return self.generate(combined, use_cache=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, chat_history: list[str]) -> str:
        history_text = "\n".join(chat_history[-20:])  # cap context length
        return PROMPT_TEMPLATE.format(history=history_text)

    @torch.no_grad()
    def _run_model(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _parse_output(self, text: str, chat_history: list[str]) -> PersonaProfile:
        """Parse the T5 output into a PersonaProfile, with fallback defaults."""
        traits = self._extract_field(text, r"[Pp]ersonality\s*traits?:\s*(.+)")
        tone = self._extract_field(text, r"[Tt]one:\s*(.+)")
        style = self._extract_field(text, r"[Cc]ommunication\s*style:\s*(.+)")
        emotion = self._extract_field(text, r"[Ee]motional\s*tendency:\s*(.+)")
        summary = self._extract_field(text, r"[Ss]ummary:\s*(.+)")

        trait_list = (
            [t.strip() for t in traits.split(",") if t.strip()]
            if traits
            else ["neutral"]
        )

        return PersonaProfile(
            personality_traits=trait_list,
            tone_preference=tone or "neutral",
            communication_style=style or "balanced",
            emotional_tendency=emotion or "neutral",
            summary=summary or text.strip() or "A conversational user.",
            raw_history=chat_history,
        )

    @staticmethod
    def _extract_field(text: str, pattern: str) -> str | None:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip().rstrip(".")
            next_field = re.search(
                r"\n\s*(?:Personality|Tone|Communication|Emotional|Summary)", value
            )
            if next_field:
                value = value[: next_field.start()].strip()
            return value
        return None

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _cache_key(self, chat_history: list[str]) -> str:
        blob = json.dumps(chat_history, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def _cache_path(self, chat_history: list[str]) -> Path:
        return self.cache_dir / f"persona_{self._cache_key(chat_history)}.json"

    def _load_cached(self, chat_history: list[str]) -> PersonaProfile | None:
        path = self._cache_path(chat_history)
        if path.exists():
            try:
                return PersonaProfile.load(path)
            except Exception:
                logger.warning("Corrupt cache file %s — regenerating.", path)
        return None

    def _save_cached(self, profile: PersonaProfile, chat_history: list[str]) -> None:
        try:
            profile.save(self._cache_path(chat_history))
        except Exception:
            logger.warning("Failed to write persona cache.", exc_info=True)
