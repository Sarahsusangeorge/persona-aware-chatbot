"""LLM API client for response generation.

Uses the OpenAI-compatible API format to call Groq (free tier, serves
Llama 3.3 70B).  Can also be pointed at any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import logging
from typing import Optional

from openai import OpenAI

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_TONE_TEMPERATURES: dict[str, float] = {
    "formal": 0.7,
    "sarcastic": 0.9,
    "empathetic": 0.8,
}


class APIClient:
    """Thin wrapper around an OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
    ):
        key = api_key or config.openai.api_key
        url = base_url or config.openai.base_url
        self._client = OpenAI(api_key=key, base_url=url)
        self.model = model or config.openai.model
        self.max_tokens = max_tokens or config.openai.max_tokens

    def chat(
        self,
        messages: list[dict[str, str]],
        tone: str = "formal",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send messages to the API and return the assistant response text.

        Returns an empty string on any API error so the caller can fall
        back to the template system.
        """
        temp = temperature if temperature is not None else _TONE_TEMPERATURES.get(tone, 0.8)
        tokens = max_tokens or self.max_tokens

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )
            text = response.choices[0].message.content or ""
            return text.strip()
        except Exception as exc:
            logger.error("API call failed: %s", exc)
            return ""
