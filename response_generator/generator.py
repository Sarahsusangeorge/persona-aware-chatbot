from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Optional

from config import config
from context_analyzer.analyzer import ContextAnalyzer, ContextFeatures
from context_analyzer.relevance import RelevanceChecker, ValidationResult
from persona_generator.persona_schema import PersonaProfile
from response_generator.api_client import APIClient
from response_generator.prompt_builder import PromptBuilder, Turn
from use_cases.modes import UseCaseMode, get_mode, detect_use_case

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Full output from the response generator including metadata."""

    response: str
    tone: str
    use_case: str
    context: Optional[ContextFeatures] = None
    validation: Optional[ValidationResult] = None
    attempts: int = 1
    explanation: str = ""
    used_fallback: bool = False


# ---------------------------------------------------------------------------
# Smart fallback response templates keyed by intent, emotion, tone, use-case
# ---------------------------------------------------------------------------

_GREETING = {
    "formal": [
        "Good day! How may I assist you today?",
        "Hello! I'm here to help. What can I do for you?",
        "Greetings! Please let me know how I can be of service.",
    ],
    "sarcastic": [
        "Well, hello there! I was just sitting here waiting for someone interesting to show up.",
        "Hey! You caught me right in the middle of doing absolutely nothing. Perfect timing.",
        "Oh hi! Finally, some company. What's on your mind?",
    ],
    "empathetic": [
        "Hi there! It's really nice to hear from you. How are you doing today?",
        "Hello! I'm glad you reached out. What's on your mind?",
        "Hey! Welcome — I'm here for you. What would you like to talk about?",
    ],
}

_EMOTIONAL = {
    "sad": {
        "formal": [
            "I understand this is a difficult situation. Please know that your feelings are entirely valid, and I'm here to help however I can.",
            "I appreciate you sharing that with me. These experiences can be challenging, but they don't define your capabilities or worth.",
        ],
        "sarcastic": [
            "Well, that's rough. But hey, if it helps, the universe has a weird way of balancing things out eventually. You've got this.",
            "Oof, that sounds like a bad time. On the bright side, it can pretty much only go up from here, right?",
        ],
        "empathetic": [
            "I'm really sorry you're going through that. It sounds incredibly tough, and it's completely okay to feel the way you do. You're not alone in this.",
            "That must be really hard. I want you to know that your feelings are completely valid, and I'm here to listen whenever you need.",
            "I hear you, and I'm sorry you're dealing with this. It takes real strength to talk about these things. How can I best support you right now?",
        ],
    },
    "angry": {
        "formal": [
            "I understand your frustration, and it's completely valid. Let's see what we can do to address the situation constructively.",
            "Your concerns are noted and taken seriously. I'd like to help find a resolution that works for you.",
        ],
        "sarcastic": [
            "Yeah, that would definitely tick me off too. Some things just have a special talent for being incredibly annoying.",
            "Oh, that's infuriating. I'd be throwing my hands up too. Want to vent about it?",
        ],
        "empathetic": [
            "I can really feel your frustration, and honestly, it makes total sense given what you're dealing with. It's okay to be angry about this.",
            "That sounds incredibly frustrating. Your feelings are completely understandable. I'm here to help however I can.",
        ],
    },
    "fearful": {
        "formal": [
            "Your concerns are understandable. Let's approach this methodically and address each worry one step at a time.",
        ],
        "sarcastic": [
            "Look, I get it — scary stuff. But you've made it through every bad day so far, which is a pretty solid track record.",
        ],
        "empathetic": [
            "It's completely natural to feel scared or anxious about this. Those feelings show you care deeply. Take a deep breath — we can figure this out together.",
            "I hear your worry, and I want you to know it's okay to feel this way. You're not alone, and there's no rush to have all the answers.",
        ],
    },
    "happy": {
        "formal": [
            "That's wonderful to hear! It's always a pleasure to share in good news. What's made your day?",
        ],
        "sarcastic": [
            "Well, look at you having a great time! I love the energy. Don't let me bring you down with my charm.",
        ],
        "empathetic": [
            "That's amazing! I'm so happy for you! You absolutely deserve to feel this good. Tell me more!",
            "Oh, that's wonderful! Your happiness is infectious. What happened?",
        ],
    },
}

_QUESTION = {
    "formal": [
        "That's an excellent question. {topic}Let me share my perspective on this with you.",
        "A very thoughtful inquiry. {topic}The key considerations here involve weighing the options carefully. What specific aspect would you like me to focus on?",
        "Good question. {topic}I'd be happy to help you think through this. Could you tell me a bit more about what specifically you'd like to know?",
    ],
    "sarcastic": [
        "Oh, now that's a question! {topic}Let me put on my thinking cap for this one.",
        "Interesting question! {topic}I could give you the short answer, but where's the fun in that? Let me elaborate.",
        "You're asking the real questions here! {topic}Let me break this down for you.",
    ],
    "empathetic": [
        "That's a really great question, and I'm glad you asked. {topic}Let me help you think through this carefully.",
        "I appreciate you asking that — it shows you're really thoughtful about this. {topic}Here's how I'd approach it.",
        "What a wonderful question! {topic}I'd love to help you explore this together.",
    ],
}

_COMPLAINT = {
    "formal": [
        "I sincerely apologize for the inconvenience you've experienced. Let me help resolve this issue promptly.",
        "I understand your frustration, and I take your concerns seriously. Let's work together on finding a solution.",
    ],
    "sarcastic": [
        "Yeah, that's definitely not how things should work. Someone somewhere clearly dropped the ball on this one.",
        "Well, that's spectacularly unhelpful. Let's see if we can fix what went wrong.",
    ],
    "empathetic": [
        "I'm so sorry you've had to deal with that. That must be really frustrating, and you shouldn't have to go through it.",
        "I completely understand your frustration. That's not the experience you deserve. Let's see how we can make it right.",
    ],
}

_REQUEST = {
    "formal": [
        "Certainly, I'd be happy to assist with that. {topic}Let me provide you with the information you need.",
        "Of course. {topic}I'll do my best to help you with that request right away.",
    ],
    "sarcastic": [
        "Your wish is my command! Well, sort of. {topic}Let me see what I can whip up for you.",
        "Oh, you need help with that? {topic}Lucky for you, helping is literally what I do.",
    ],
    "empathetic": [
        "Absolutely, I'd love to help! {topic}Let's work through this together step by step.",
        "Of course! {topic}I want to make sure you get exactly what you need. Let's get started.",
    ],
}

_FAREWELL = {
    "formal": [
        "It has been a pleasure speaking with you. Don't hesitate to reach out if you need anything else. Goodbye!",
        "Thank you for the conversation. I wish you all the best. Take care!",
    ],
    "sarcastic": [
        "Leaving already? And here I thought we were just getting started. Take care out there!",
        "Fine, fine, go live your life. I'll just be here. No rush coming back. Bye!",
    ],
    "empathetic": [
        "It was really lovely talking with you! Take care of yourself, and remember I'm always here if you need someone.",
        "I'm so glad we got to chat. Take care, and don't hesitate to come back anytime you want to talk!",
    ],
}

_GRATITUDE = {
    "formal": [
        "You're most welcome. I'm glad I could be of assistance. Is there anything else I can help with?",
        "It was my pleasure. Please don't hesitate to reach out if you need further help.",
    ],
    "sarcastic": [
        "You're welcome! See, I can actually be helpful when I try. Don't tell anyone though.",
        "Aw, that's nice of you to say. I'll try not to let it go to my head.",
    ],
    "empathetic": [
        "You're so welcome! It genuinely makes me happy to know I could help. That means a lot!",
        "I'm really glad I could be there for you! Don't ever hesitate to reach out.",
    ],
}

_GENERAL = {
    "formal": [
        "I appreciate you sharing that. {topic}Would you like to discuss this further? I'm happy to explore any aspect in more detail.",
        "That's a noteworthy point. {topic}I'd be glad to delve deeper into this topic if you'd like.",
    ],
    "sarcastic": [
        "Well, that's certainly something! {topic}I've got plenty of thoughts on this. Want to hear them?",
        "Interesting! {topic}I could go on about this all day, but I'll pace myself. What's next?",
    ],
    "empathetic": [
        "Thank you for sharing that with me — I really appreciate it. {topic}How are you feeling about all of this?",
        "That really resonates with me. {topic}I'd love to hear more about your thoughts on this.",
    ],
}

_USE_CASE = {
    "mental_health": [
        "I really appreciate you opening up about this. Your feelings are completely valid, and it takes real courage to talk about them. I'm here for you — how are you feeling right now?",
        "I hear you, and I want you to know that what you're feeling is absolutely okay. You're not alone in this. Would you like to talk more about what's going on?",
        "Thank you for trusting me with this. It sounds like you're carrying a lot. Remember, it's okay to not be okay sometimes. What would feel most helpful right now?",
    ],
    "educational_tutor": [
        "Great question! Let me break this down in a way that makes it clearer. Every complex concept is really just simpler ideas connected together. Where would you like me to start?",
        "I love that you're thinking about this! Let me explain step by step. What's the part that feels most confusing to you?",
        "That's a really thoughtful question, and I can see you're on the right track. Let me help fill in the gaps so it all clicks.",
    ],
    "customer_support": [
        "I understand your concern, and I'm here to help get this resolved as quickly as possible. Could you share a few more details so I can assist you better?",
        "I apologize for the inconvenience. Your satisfaction matters, and I want to make sure we address this properly. Let me walk you through the next steps.",
        "Thank you for bringing this to my attention. Let me look into this right away and find the best solution for you.",
    ],
    "gaming_npc": [
        "Ah, adventurer! You've come to the right place. I sense great potential in you. What quest brings you to these lands today?",
        "Well met, traveler! The path ahead is perilous, but I see courage in your eyes. How may I aid your journey?",
        "Greetings, brave one! Tales of your deeds echo through these halls. What knowledge do you seek?",
    ],
    "email_generator": [
        "I'd be happy to help you draft that email! Here's a structure you can use:\n\nSubject: [Your topic]\n\nDear [Recipient],\n\nI hope this message finds you well. [Your main point here.]\n\nPlease let me know if you have any questions.\n\nBest regards,\n[Your name]",
    ],
    "social_media": [
        "Here's an engaging post idea! Start with a hook that grabs attention, share your key message in 1-2 punchy sentences, and end with a call to action. Want me to draft something specific?",
    ],
    "virtual_assistant": [
        "Sure, I can help you with that! Let me organize this for you. What would you like me to tackle first?",
        "Got it! I'll help you get this sorted out efficiently. Here's what I'd suggest as the first step.",
    ],
    "ai_companion": [
        "Hey, I'm glad you're here! It's always nice to chat. What's been on your mind lately?",
        "Oh, tell me more about that! I'm genuinely curious. What made you think of it?",
    ],
}


class ResponseGenerator:
    """Generates persona-aware, context-relevant, tone-controlled responses.

    When an OpenAI API key is configured the generator calls GPT-4o-mini via
    ``APIClient``.  If the API is unavailable (no key, network error) it
    falls back to a curated template system keyed on intent, emotion, and
    tone.
    """

    def __init__(
        self,
        api_client: Optional[APIClient] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        context_analyzer: Optional[ContextAnalyzer] = None,
        relevance_checker: Optional[RelevanceChecker] = None,
    ):
        self.api_client = api_client
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.context_analyzer = context_analyzer or ContextAnalyzer()
        self.relevance_checker = relevance_checker or RelevanceChecker()

    def generate(
        self,
        persona: PersonaProfile,
        query: str,
        tone: str = "formal",
        conversation_history: Optional[list[Turn]] = None,
        use_case: Optional[UseCaseMode] = None,
        memory_summary: str = "",
        max_new_tokens: Optional[int] = None,
        validate: bool = True,
    ) -> str:
        result = self.generate_full(
            persona=persona, query=query, tone=tone,
            conversation_history=conversation_history,
            use_case=use_case, memory_summary=memory_summary,
            max_new_tokens=max_new_tokens, validate=validate,
        )
        return result.response

    def generate_full(
        self,
        persona: PersonaProfile,
        query: str,
        tone: str = "formal",
        conversation_history: Optional[list[Turn]] = None,
        use_case: Optional[UseCaseMode] = None,
        memory_summary: str = "",
        max_new_tokens: Optional[int] = None,
        validate: bool = True,
    ) -> GenerationResult:
        """Generate with full metadata.

        Tries the OpenAI API first; falls back to curated templates on error.
        """
        context = self.context_analyzer.analyze(
            query,
            [t.text for t in conversation_history] if conversation_history else None,
        )

        effective_tone = tone
        if use_case and use_case.tone_override:
            effective_tone = use_case.tone_override

        used_fallback = False
        response = ""
        validation = None

        if self.api_client is not None:
            messages = self.prompt_builder.build_messages(
                persona=persona,
                current_query=query,
                tone=effective_tone,
                conversation_history=conversation_history,
                context=context,
                use_case=use_case,
                memory_summary=memory_summary,
            )
            response = self.api_client.chat(messages, tone=effective_tone)

        if response and validate:
            validation = self.relevance_checker.validate(query, response, context)
            if not validation.passed:
                logger.info(
                    "API output failed quality check: %s", validation.explanation,
                )
                response = ""

        if not response:
            response = self._smart_fallback(
                query, effective_tone, context, use_case, conversation_history,
            )
            used_fallback = True
            validation = self.relevance_checker.validate(query, response, context)

        explanation = self._build_explanation(
            effective_tone, use_case, context, validation, used_fallback,
        )

        return GenerationResult(
            response=response,
            tone=effective_tone,
            use_case=use_case.name if use_case else "general",
            context=context,
            validation=validation,
            attempts=1,
            explanation=explanation,
            used_fallback=used_fallback,
        )

    def generate_candidates(
        self,
        persona: PersonaProfile,
        query: str,
        tone: str = "formal",
        conversation_history: Optional[list[Turn]] = None,
        use_case: Optional[UseCaseMode] = None,
        memory_summary: str = "",
        num_candidates: int = 3,
        max_new_tokens: Optional[int] = None,
    ) -> list[str]:
        """Generate multiple candidate responses ranked by relevance."""
        context = self.context_analyzer.analyze(query)
        effective_tone = tone
        if use_case and use_case.tone_override:
            effective_tone = use_case.tone_override

        candidates: list[str] = []

        if self.api_client is not None:
            messages = self.prompt_builder.build_messages(
                persona=persona, current_query=query, tone=effective_tone,
                conversation_history=conversation_history, context=context,
                use_case=use_case, memory_summary=memory_summary,
            )
            for _ in range(num_candidates):
                resp = self.api_client.chat(messages, tone=effective_tone)
                if resp:
                    val = self.relevance_checker.validate(query, resp, context)
                    if val.passed:
                        candidates.append(resp)

        if not candidates:
            fallback = self._smart_fallback(
                query, effective_tone, context, use_case, conversation_history,
            )
            return [fallback]

        scored = [(self.relevance_checker.score_relevance(query, c), c) for c in candidates]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [c for _, c in scored]

    def chat(
        self,
        persona: PersonaProfile,
        query: str,
        tone: str = "formal",
        raw_history: Optional[list[str]] = None,
        use_case_name: Optional[str] = None,
        memory_summary: str = "",
        max_new_tokens: Optional[int] = None,
    ) -> str:
        turns: list[Turn] | None = None
        if raw_history:
            roles = ["User", "Assistant"]
            turns = [Turn(role=roles[i % 2], text=msg) for i, msg in enumerate(raw_history)]
        use_case = get_mode(use_case_name) if use_case_name else None
        return self.generate(
            persona=persona, query=query, tone=tone,
            conversation_history=turns, use_case=use_case,
            memory_summary=memory_summary, max_new_tokens=max_new_tokens,
        )

    # ------------------------------------------------------------------
    # Smart fallback
    # ------------------------------------------------------------------

    def _smart_fallback(
        self,
        query: str,
        tone: str,
        context: Optional[ContextFeatures] = None,
        use_case: Optional[UseCaseMode] = None,
        conversation_history: Optional[list[Turn]] = None,
    ) -> str:
        """Produce a relevant response using templates matched to intent,
        emotion, tone, and use-case mode.

        Priority order (highest first):
        1. Emotion (sad/angry/fearful/happy) -- never overridden
        2. Strong intents (greeting/farewell/gratitude/complaint)
        3. Use-case templates -- only for neutral, non-emotional statements
        4. Question / request / general
        """

        topic = ""
        if context and context.topic:
            topic = f"Regarding {context.topic} — "
        elif context and context.keywords:
            topic = f"On the topic of {', '.join(context.keywords[:2])} — "

        intent = context.intent if context else "statement"
        emotion = context.emotion if context else "neutral"

        # --- 1. Emotion always wins ---
        if intent == "emotional_sharing" or emotion in ("sad", "angry", "fearful"):
            emo_key = emotion if emotion in _EMOTIONAL else "sad"
            pool = _EMOTIONAL[emo_key].get(tone, _EMOTIONAL[emo_key]["empathetic"])
            return random.choice(pool)

        if emotion == "happy":
            pool = _EMOTIONAL["happy"].get(tone, _EMOTIONAL["happy"]["empathetic"])
            return random.choice(pool)

        # --- 2. Strong intents ---
        if intent == "greeting":
            return random.choice(_GREETING.get(tone, _GREETING["formal"]))

        if intent == "farewell":
            return random.choice(_FAREWELL.get(tone, _FAREWELL["formal"]))

        if intent == "gratitude":
            return random.choice(_GRATITUDE.get(tone, _GRATITUDE["formal"]))

        if intent == "complaint":
            return random.choice(_COMPLAINT.get(tone, _COMPLAINT["formal"]))

        # --- 3. Use-case templates (neutral context only) ---
        if use_case and use_case.name in _USE_CASE and emotion == "neutral":
            return random.choice(_USE_CASE[use_case.name])

        # --- 4. Question / request / general ---
        if intent == "question" or (context and context.is_question):
            tmpl = random.choice(_QUESTION.get(tone, _QUESTION["formal"]))
            return tmpl.replace("{topic}", topic)

        if intent in ("request", "instruction"):
            tmpl = random.choice(_REQUEST.get(tone, _REQUEST["formal"]))
            return tmpl.replace("{topic}", topic)

        tmpl = random.choice(_GENERAL.get(tone, _GENERAL["formal"]))
        return tmpl.replace("{topic}", topic)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_explanation(
        tone: str,
        use_case: Optional[UseCaseMode],
        context: Optional[ContextFeatures],
        validation: Optional[ValidationResult],
        used_fallback: bool,
    ) -> str:
        parts = [f"Tone: {tone}"]
        if use_case:
            parts.append(f"Mode: {use_case.display_name}")
        if context:
            parts.append(f"Intent: {context.intent}")
            parts.append(f"Emotion: {context.emotion}")
            if context.keywords:
                parts.append(f"Topics: {', '.join(context.keywords[:5])}")
        if validation:
            parts.append(f"Relevance: {validation.relevance_score:.2f}")
        if used_fallback:
            parts.append("Smart fallback (model output was low quality)")
        return " | ".join(parts)
