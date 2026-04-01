from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from context_analyzer.analyzer import ContextFeatures
from persona_generator.persona_schema import PersonaProfile
from use_cases.modes import UseCaseMode, get_mode


@dataclass
class Turn:
    """A single conversational turn."""
    role: str  # "User" or "Assistant"
    text: str


class PromptBuilder:
    """Constructs prompts that combine persona context, tone, conversation
    history, context features, and use-case mode into a natural conversation
    format that GPT-2 can complete coherently.

    Uses a natural dialogue format rather than tagged instructions, since
    base GPT-2 is a text-completion model, not instruction-tuned.
    """

    TONE_DESCRIPTIONS: dict[str, str] = {
        "formal": "professional, polite, and well-structured",
        "sarcastic": "witty, clever, and playfully sarcastic",
        "empathetic": "warm, caring, and emotionally supportive",
    }

    def __init__(self, max_history_turns: int = 10):
        self.max_history_turns = max_history_turns

    def build(
        self,
        persona: PersonaProfile,
        current_query: str,
        tone: str,
        conversation_history: Optional[list[Turn]] = None,
        context: Optional[ContextFeatures] = None,
        use_case: Optional[UseCaseMode] = None,
        memory_summary: str = "",
    ) -> str:
        """Build a natural-language prompt that GPT-2 can complete.

        Uses a narrative setup followed by dialogue rather than structured
        tags, since base GPT-2 handles natural text continuation much
        better than instruction-following.
        """
        sections: list[str] = []

        sections.append(self._build_scenario(persona, tone, use_case, memory_summary))
        sections.append("")
        sections.append(self._build_dialogue(current_query, conversation_history))

        return "\n".join(sections)

    def build_from_raw(
        self,
        persona: PersonaProfile,
        current_query: str,
        tone: str,
        raw_history: Optional[list[str]] = None,
        context: Optional[ContextFeatures] = None,
        use_case: Optional[UseCaseMode] = None,
        memory_summary: str = "",
    ) -> str:
        """Convenience wrapper that accepts raw alternating strings."""
        turns: list[Turn] | None = None
        if raw_history:
            turns = []
            roles = ["User", "Assistant"]
            for i, msg in enumerate(raw_history):
                turns.append(Turn(role=roles[i % 2], text=msg))
        return self.build(
            persona, current_query, tone, turns, context, use_case, memory_summary,
        )

    def _build_scenario(
        self,
        persona: PersonaProfile,
        tone: str,
        use_case: Optional[UseCaseMode] = None,
        memory_summary: str = "",
    ) -> str:
        """Create a natural scenario description GPT-2 can use as context."""
        tone_desc = self.TONE_DESCRIPTIONS.get(tone, f"{tone}")
        persona_desc = persona.prompt_block()

        scenario = (
            f"The following is a conversation between a User and an Assistant. "
            f"The Assistant is {tone_desc}. "
            f"The Assistant's personality: {persona_desc}. "
            f"The Assistant always gives helpful, relevant replies that directly "
            f"address what the User said."
        )

        if use_case and use_case.name != "general":
            scenario += f" The Assistant acts as a {use_case.display_name}."

        if memory_summary:
            scenario += f" Background: {memory_summary}"

        return scenario

    def build_messages(
        self,
        persona: PersonaProfile,
        current_query: str,
        tone: str,
        conversation_history: Optional[list[Turn]] = None,
        context: Optional[ContextFeatures] = None,
        use_case: Optional[UseCaseMode] = None,
        memory_summary: str = "",
    ) -> list[dict[str, str]]:
        """Build an OpenAI chat-format message list (system/user/assistant)."""
        system_text = self._build_system_message(persona, tone, use_case, memory_summary)
        messages: list[dict[str, str]] = [{"role": "system", "content": system_text}]

        if conversation_history:
            recent = conversation_history[-self.max_history_turns :]
            for turn in recent:
                role = "user" if turn.role == "User" else "assistant"
                messages.append({"role": role, "content": turn.text})

        messages.append({"role": "user", "content": current_query})
        return messages

    def _build_system_message(
        self,
        persona: PersonaProfile,
        tone: str,
        use_case: Optional[UseCaseMode] = None,
        memory_summary: str = "",
    ) -> str:
        """Compose a rich system prompt for instruction-tuned models."""
        tone_desc = self.TONE_DESCRIPTIONS.get(tone, tone)
        persona_desc = persona.prompt_block()

        parts = [
            f"You are a conversational assistant. Your tone is {tone_desc}.",
            f"Your personality: {persona_desc}.",
            "Always give helpful, relevant replies that directly address what the user said.",
            "Match the emotional register of the user's message — if they are venting, be supportive; if they are asking a question, answer it.",
        ]

        if use_case and use_case.name != "general":
            parts.append(f"Domain role: {use_case.display_name}. {use_case.system_instruction}")
            if use_case.response_constraints:
                parts.append("Constraints: " + "; ".join(use_case.response_constraints) + ".")

        if memory_summary:
            parts.append(f"Background from prior conversation: {memory_summary}")

        return " ".join(parts)

    def _build_dialogue(
        self,
        current_query: str,
        history: Optional[list[Turn]] = None,
    ) -> str:
        """Format conversation history and current query as dialogue."""
        lines = []

        if history:
            recent = history[-self.max_history_turns :]
            for turn in recent:
                lines.append(f"{turn.role}: {turn.text}")

        lines.append(f"User: {current_query}")
        lines.append("Assistant:")

        return "\n".join(lines)
