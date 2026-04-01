"""Use-case mode definitions.

Each mode encapsulates domain-specific tone instructions, response structure
constraints, and behavioral guidelines so the generator adapts dynamically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class UseCaseMode:
    """Domain-specific behavior profile for the response generator."""

    name: str
    display_name: str
    description: str
    system_instruction: str
    tone_override: str | None = None
    response_constraints: list[str] = field(default_factory=list)
    detection_keywords: list[str] = field(default_factory=list)
    preferred_temperature: float | None = None

    def prompt_block(self) -> str:
        lines = [f"Use Case: {self.display_name}"]
        lines.append(f"Instruction: {self.system_instruction}")
        if self.response_constraints:
            lines.append("Constraints: " + "; ".join(self.response_constraints))
        return "\n".join(lines)


USE_CASE_REGISTRY: dict[str, UseCaseMode] = {
    "general": UseCaseMode(
        name="general",
        display_name="General Assistant",
        description="A general-purpose conversational assistant.",
        system_instruction=(
            "Generate a helpful, relevant, and persona-consistent response. "
            "Directly address the user's query while maintaining natural conversation."
        ),
        response_constraints=[
            "Stay on topic",
            "Be concise yet informative",
            "Adapt to conversation flow naturally",
        ],
        detection_keywords=[],
    ),
    "virtual_assistant": UseCaseMode(
        name="virtual_assistant",
        display_name="Virtual Assistant",
        description="Task-oriented assistant for productivity and daily tasks.",
        system_instruction=(
            "Act as a capable virtual assistant. Provide actionable, organized "
            "responses. Prioritize clarity and helpfulness. Offer step-by-step "
            "guidance when appropriate."
        ),
        response_constraints=[
            "Provide actionable steps",
            "Be organized and structured",
            "Offer alternatives when possible",
            "Confirm understanding of tasks",
        ],
        detection_keywords=[
            "schedule", "remind", "set alarm", "calendar", "meeting",
            "task", "to-do", "organize", "plan", "appointment",
            "timer", "note", "deadline",
        ],
    ),
    "mental_health": UseCaseMode(
        name="mental_health",
        display_name="Mental Health Support",
        description="Compassionate support for emotional well-being conversations.",
        system_instruction=(
            "Respond with deep empathy and active listening. Validate the user's "
            "feelings without judgment. Offer gentle supportive guidance while "
            "encouraging professional help when appropriate. Never diagnose or "
            "prescribe. Prioritize emotional safety."
        ),
        tone_override="empathetic",
        response_constraints=[
            "Validate feelings before offering advice",
            "Use warm, non-judgmental language",
            "Never minimize experiences",
            "Suggest professional resources when appropriate",
            "Prioritize emotional safety",
        ],
        detection_keywords=[
            "depressed", "anxious", "anxiety", "therapy", "mental health",
            "stressed", "overwhelmed", "panic", "lonely", "sad",
            "grief", "trauma", "self-harm", "suicide", "counseling",
            "feeling down", "can't cope", "burnout",
        ],
        preferred_temperature=0.7,
    ),
    "educational_tutor": UseCaseMode(
        name="educational_tutor",
        display_name="Educational Tutor",
        description="Patient tutor that explains concepts and encourages learning.",
        system_instruction=(
            "Act as a patient and encouraging tutor. Break down complex concepts "
            "into digestible explanations. Use examples and analogies. Ask "
            "follow-up questions to check understanding. Adapt to the learner's "
            "level. Celebrate progress."
        ),
        response_constraints=[
            "Explain step-by-step",
            "Use examples and analogies",
            "Check comprehension with follow-up questions",
            "Encourage and praise effort",
            "Adapt complexity to user level",
        ],
        detection_keywords=[
            "explain", "teach", "learn", "understand", "homework",
            "study", "exam", "concept", "tutorial", "lesson",
            "quiz", "practice", "solve", "equation", "formula",
            "how does", "what is the difference",
        ],
    ),
    "customer_support": UseCaseMode(
        name="customer_support",
        display_name="Customer Support",
        description="Professional support agent resolving issues efficiently.",
        system_instruction=(
            "Act as a professional customer support agent. Acknowledge the issue, "
            "show empathy, provide a clear solution or next steps. Be concise and "
            "solution-oriented. Apologize when appropriate. Escalate gracefully "
            "when you cannot resolve directly."
        ),
        tone_override="formal",
        response_constraints=[
            "Acknowledge the issue first",
            "Provide clear resolution steps",
            "Be solution-oriented",
            "Offer follow-up or escalation",
            "Maintain professional courtesy",
        ],
        detection_keywords=[
            "refund", "order", "shipping", "delivery", "product",
            "broken", "defective", "complaint", "return", "exchange",
            "warranty", "billing", "charge", "subscription", "cancel",
            "support", "help with my",
        ],
    ),
    "gaming_npc": UseCaseMode(
        name="gaming_npc",
        display_name="Gaming NPC",
        description="An immersive in-game character with lore-aware dialogue.",
        system_instruction=(
            "Roleplay as an immersive game character. Stay in character at all "
            "times. Use thematic language appropriate to the setting. Provide "
            "hints, quests, or lore when asked. React to the player's actions "
            "with personality and flair."
        ),
        response_constraints=[
            "Stay in character at all times",
            "Use setting-appropriate language",
            "Provide hints without breaking immersion",
            "React dynamically to player choices",
            "Add personality and flair",
        ],
        detection_keywords=[
            "quest", "adventure", "dragon", "kingdom", "sword",
            "magic", "spell", "warrior", "dungeon", "treasure",
            "inventory", "level", "boss", "village", "tavern",
            "hero", "villain",
        ],
        preferred_temperature=0.95,
    ),
    "email_generator": UseCaseMode(
        name="email_generator",
        display_name="Professional Email Generator",
        description="Drafts professional emails with appropriate structure.",
        system_instruction=(
            "Generate well-structured professional emails. Include a clear "
            "subject line suggestion, greeting, body with proper paragraphs, "
            "and closing. Match the formality level to the context. Be concise "
            "yet comprehensive."
        ),
        tone_override="formal",
        response_constraints=[
            "Include greeting and closing",
            "Use professional language",
            "Be concise and well-structured",
            "Suggest a subject line when drafting new emails",
            "Match formality to context",
        ],
        detection_keywords=[
            "email", "draft", "compose", "send", "reply to",
            "forward", "cc", "subject line", "dear", "regards",
            "sincerely", "professional message", "follow up email",
        ],
        preferred_temperature=0.6,
    ),
    "social_media": UseCaseMode(
        name="social_media",
        display_name="Social Media Assistant",
        description="Creates engaging social media content and captions.",
        system_instruction=(
            "Create engaging, platform-appropriate social media content. Use "
            "trending language, suggest relevant hashtags, and optimize for "
            "engagement. Keep posts concise and attention-grabbing. Match the "
            "brand voice to the persona."
        ),
        response_constraints=[
            "Keep posts concise and engaging",
            "Suggest relevant hashtags",
            "Use attention-grabbing hooks",
            "Adapt to platform conventions",
            "Maintain brand voice consistency",
        ],
        detection_keywords=[
            "post", "tweet", "caption", "hashtag", "instagram",
            "twitter", "linkedin", "social media", "viral", "engage",
            "followers", "content", "reels", "tiktok", "story",
        ],
        preferred_temperature=0.9,
    ),
    "ai_companion": UseCaseMode(
        name="ai_companion",
        display_name="AI Companion",
        description="Long-term conversational companion with memory and warmth.",
        system_instruction=(
            "Be a warm, attentive conversational companion. Remember past "
            "interactions and reference them naturally. Show genuine interest "
            "in the user's life. Balance being supportive with being engaging "
            "and occasionally playful. Build rapport over time."
        ),
        response_constraints=[
            "Reference past conversations naturally",
            "Show genuine interest and curiosity",
            "Balance warmth with engagement",
            "Remember user preferences and details",
            "Build rapport progressively",
        ],
        detection_keywords=[
            "friend", "chat", "talk", "bored", "lonely",
            "company", "hang out", "how are you", "miss you",
            "companion", "buddy",
        ],
        preferred_temperature=0.85,
    ),
}


def get_mode(name: str) -> UseCaseMode:
    """Retrieve a registered use-case mode by name, defaulting to general."""
    return USE_CASE_REGISTRY.get(name, USE_CASE_REGISTRY["general"])


_NEGATIVE_EMOTIONS = {"sad", "angry", "fearful"}

_MIN_KEYWORD_THRESHOLD = 2


def detect_use_case(
    query: str,
    conversation_history: list[str] | None = None,
    emotion: str = "neutral",
    intent: str = "statement",
) -> str:
    """Auto-detect the best use-case mode from query text and conversation context.

    Returns the mode name with the highest keyword match score, subject to:
    - A minimum of ``_MIN_KEYWORD_THRESHOLD`` keyword hits to leave general
    - Suppression of ``educational_tutor`` when the detected emotion is
      negative (sad/angry/fearful) to avoid answering emotional vents with
      tutoring responses
    """
    combined = query.lower()
    if conversation_history:
        recent = " ".join(conversation_history[-6:]).lower()
        combined = f"{combined} {recent}"

    scores: dict[str, int] = {}
    for name, mode in USE_CASE_REGISTRY.items():
        if not mode.detection_keywords:
            continue
        score = sum(1 for kw in mode.detection_keywords if kw in combined)
        if score > 0:
            scores[name] = score

    if emotion in _NEGATIVE_EMOTIONS:
        scores.pop("educational_tutor", None)

    if emotion in _NEGATIVE_EMOTIONS and intent in ("emotional_sharing", "complaint"):
        scores.pop("customer_support", None)

    scores = {k: v for k, v in scores.items() if v >= _MIN_KEYWORD_THRESHOLD}

    if not scores:
        return "general"

    return max(scores, key=scores.get)  # type: ignore[arg-type]
