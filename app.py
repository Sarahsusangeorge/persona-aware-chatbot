"""Streamlit frontend for the Universal Persona-Aware Response Generator.

Provides:
- Sidebar: chat-history upload, persona generation, tone selection,
  use-case mode, memory controls, save/load
- Main area: chat interface with context-aware responses,
  response explanations, and validation indicators
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import config
from context_analyzer.analyzer import ContextAnalyzer
from context_analyzer.relevance import RelevanceChecker
from persona_generator.persona_schema import PersonaProfile
from persona_generator.generator import PersonaGenerator
from response_generator.generator import ResponseGenerator, GenerationResult
from response_generator.api_client import APIClient
from response_generator.prompt_builder import Turn
from use_cases.modes import USE_CASE_REGISTRY, get_mode, detect_use_case
from use_cases.memory import ConversationMemory

logger = logging.getLogger(__name__)

TONES = ["formal", "sarcastic", "empathetic"]
SAMPLE_HISTORIES_PATH = config.paths.sample_histories

USE_CASE_DISPLAY = {
    name: mode.display_name for name, mode in USE_CASE_REGISTRY.items()
}


def _init_state() -> None:
    defaults = {
        "messages": [],
        "persona": None,
        "chat_history_raw": [],
        "tone": "formal",
        "use_case_mode": "general",
        "auto_detect_mode": True,
        "persona_generator": None,
        "response_generator": None,
        "context_analyzer": None,
        "relevance_checker": None,
        "memory": ConversationMemory(),
        "models_loaded": False,
        "show_explanations": False,
        "last_generation_result": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _load_models() -> None:
    """Lazy-load heavy models only once per session."""
    if st.session_state.models_loaded:
        return
    with st.spinner("Loading models (first run only)..."):
        st.session_state.persona_generator = PersonaGenerator()
        st.session_state.context_analyzer = ContextAnalyzer()
        st.session_state.relevance_checker = RelevanceChecker()

        api_client = APIClient() if config.openai.use_api else None
        st.session_state.response_generator = ResponseGenerator(
            api_client=api_client,
            context_analyzer=st.session_state.context_analyzer,
            relevance_checker=st.session_state.relevance_checker,
        )
        st.session_state.models_loaded = True
        logger.info("Models loaded successfully (API: %s).", config.openai.use_api)


@st.cache_data
def _load_sample_histories() -> dict[str, list[str]]:
    if SAMPLE_HISTORIES_PATH.exists():
        return json.loads(SAMPLE_HISTORIES_PATH.read_text(encoding="utf-8"))
    return {}


def _render_sidebar() -> None:
    with st.sidebar:
        st.title("Configuration")

        # ── Chat history input ──
        st.subheader("Chat History")

        input_method = st.radio(
            "Provide chat history via:",
            ["Sample histories", "Paste text", "Upload JSON"],
            horizontal=True,
        )

        if input_method == "Sample histories":
            samples = _load_sample_histories()
            if samples:
                choice = st.selectbox(
                    "Select a sample",
                    list(samples.keys()),
                    format_func=lambda k: k.replace("_", " ").title(),
                )
                if st.button("Load sample", use_container_width=True):
                    st.session_state.chat_history_raw = samples[choice]
                    st.success(f"Loaded {len(samples[choice])} messages.")
            else:
                st.info("No sample histories found.")

        elif input_method == "Paste text":
            pasted = st.text_area(
                "Paste messages (one per line):",
                height=150,
                placeholder="Hey, how are you?\nI'm doing great!\n...",
            )
            if st.button("Use pasted history", use_container_width=True) and pasted.strip():
                lines = [l.strip() for l in pasted.strip().splitlines() if l.strip()]
                st.session_state.chat_history_raw = lines
                st.success(f"Loaded {len(lines)} messages.")

        elif input_method == "Upload JSON":
            uploaded = st.file_uploader("Upload JSON file", type=["json"])
            if uploaded is not None:
                try:
                    data = json.loads(uploaded.read().decode("utf-8"))
                    if isinstance(data, list):
                        st.session_state.chat_history_raw = data
                    elif isinstance(data, dict):
                        first_key = next(iter(data))
                        st.session_state.chat_history_raw = data[first_key]
                    st.success(f"Loaded {len(st.session_state.chat_history_raw)} messages.")
                except Exception as e:
                    st.error(f"Failed to parse JSON: {e}")

        if st.session_state.chat_history_raw:
            with st.expander(
                f"Current history ({len(st.session_state.chat_history_raw)} msgs)",
                expanded=False,
            ):
                for i, msg in enumerate(st.session_state.chat_history_raw):
                    role = "User" if i % 2 == 0 else "Assistant"
                    st.text(f"{role}: {msg}")

        st.divider()

        # ── Persona generation ──
        st.subheader("Persona Profile")

        if st.button(
            "Generate Persona",
            use_container_width=True,
            disabled=not st.session_state.chat_history_raw,
            help="Analyze chat history to extract a personality profile",
        ):
            _load_models()
            with st.spinner("Analyzing personality..."):
                gen: PersonaGenerator = st.session_state.persona_generator
                profile = gen.generate(st.session_state.chat_history_raw)
                st.session_state.persona = profile

        if st.session_state.persona is not None:
            _render_persona_card(st.session_state.persona)

            col_save, col_clear = st.columns(2)
            with col_save:
                if st.button("Save persona", use_container_width=True):
                    path = config.paths.personas_dir / "current_persona.json"
                    st.session_state.persona.save(path)
                    st.success("Saved!")
            with col_clear:
                if st.button("Clear persona", use_container_width=True):
                    st.session_state.persona = None
                    st.rerun()

            saved_path = config.paths.personas_dir / "current_persona.json"
            if saved_path.exists():
                if st.button("Load saved persona", use_container_width=True):
                    st.session_state.persona = PersonaProfile.load(saved_path)
                    st.success("Loaded saved persona.")
        else:
            st.info("No persona generated yet. Load a chat history and click 'Generate Persona'.")
            saved_path = config.paths.personas_dir / "current_persona.json"
            if saved_path.exists():
                if st.button("Load saved persona", use_container_width=True):
                    st.session_state.persona = PersonaProfile.load(saved_path)
                    st.success("Loaded saved persona.")

        st.divider()

        # ── Use-Case Mode ──
        st.subheader("Use-Case Mode")

        st.session_state.auto_detect_mode = st.toggle(
            "Auto-detect mode",
            value=st.session_state.auto_detect_mode,
            help="Automatically detect the best use-case mode from your message",
        )

        if not st.session_state.auto_detect_mode:
            mode_names = list(USE_CASE_DISPLAY.keys())
            current_idx = mode_names.index(st.session_state.use_case_mode) if st.session_state.use_case_mode in mode_names else 0
            st.session_state.use_case_mode = st.selectbox(
                "Select mode:",
                mode_names,
                index=current_idx,
                format_func=lambda k: USE_CASE_DISPLAY[k],
            )

        current_mode = get_mode(st.session_state.use_case_mode)
        with st.container(border=True):
            st.caption(f"**Active:** {current_mode.display_name}")
            st.caption(current_mode.description)

        st.divider()

        # ── Tone selection ──
        st.subheader("Tone")
        st.session_state.tone = st.selectbox(
            "Response tone:",
            TONES,
            format_func=str.capitalize,
            index=TONES.index(st.session_state.tone),
        )

        tone_descriptions = {
            "formal": "Professional and structured responses.",
            "sarcastic": "Witty and playfully sarcastic replies.",
            "empathetic": "Warm, supportive, and understanding tone.",
        }
        st.caption(tone_descriptions.get(st.session_state.tone, ""))

        st.divider()

        # ── Response controls ──
        st.subheader("Response Settings")

        st.session_state.show_explanations = st.toggle(
            "Show response explanations",
            value=st.session_state.show_explanations,
            help="Display why this tone and style were chosen",
        )

        st.divider()

        # ── Memory & chat controls ──
        st.subheader("Conversation")

        memory: ConversationMemory = st.session_state.memory
        st.caption(f"Turns: {len(memory)} | Facts: {len(memory.user_facts)}")

        if memory.user_facts:
            with st.expander("Remembered facts"):
                for k, v in memory.user_facts.items():
                    st.text(f"{k}: {v}")

        col_clear_chat, col_clear_mem = st.columns(2)
        with col_clear_chat:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col_clear_mem:
            if st.button("Clear memory", use_container_width=True):
                st.session_state.memory = ConversationMemory()
                st.session_state.messages = []
                st.rerun()

        st.divider()
        if config.openai.use_api:
            st.caption(f"Model: {config.openai.model}")
        else:
            st.caption("No API key — using template fallback only. "
                       "Get a free Groq key at console.groq.com/keys")


def _render_persona_card(persona: PersonaProfile) -> None:
    """Display persona profile as a styled info card."""
    with st.container(border=True):
        if persona.personality_traits:
            st.markdown(
                "**Traits:** "
                + ", ".join(f"`{t}`" for t in persona.personality_traits)
            )
        st.markdown(f"**Tone:** {persona.tone_preference}")
        st.markdown(f"**Style:** {persona.communication_style}")
        st.markdown(f"**Emotion:** {persona.emotional_tendency}")
        if persona.summary:
            st.markdown(f"*{persona.summary}*")


def _render_explanation(result: GenerationResult) -> None:
    """Render generation metadata in a collapsible panel."""
    with st.expander("Response details", expanded=False):
        cols = st.columns(3)
        with cols[0]:
            st.metric("Relevance", f"{result.validation.relevance_score:.2f}" if result.validation else "N/A")
        with cols[1]:
            st.metric("Intent", result.context.intent if result.context else "N/A")
        with cols[2]:
            st.metric("Attempts", str(result.attempts))

        if result.context:
            st.caption(f"Emotion: {result.context.emotion}")
            if result.context.keywords:
                st.caption(f"Keywords: {', '.join(result.context.keywords[:6])}")

        st.caption(f"Mode: {result.use_case} | Tone: {result.tone}")

        if result.validation:
            status = "Passed" if result.validation.passed else "Needs improvement"
            st.caption(f"Validation: {status}")


def _render_chat() -> None:
    st.title("EchoPersona")
    st.caption(
        "Context-aware, persona-driven responses with multi-use-case support "
        "and dynamic tone switching."
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if (
                msg["role"] == "assistant"
                and st.session_state.show_explanations
                and "result" in msg
            ):
                _render_explanation(msg["result"])

    prompt = st.chat_input(
        "Type your message...",
        disabled=st.session_state.persona is None,
    )

    if st.session_state.persona is None:
        st.info(
            "To start chatting, load a chat history in the sidebar and "
            "generate a persona profile."
        )
        return

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        _load_models()

        memory: ConversationMemory = st.session_state.memory
        memory.add_turn("user", prompt)

        analyzer: ContextAnalyzer = st.session_state.context_analyzer
        quick_ctx = analyzer.analyze(prompt)

        if st.session_state.auto_detect_mode:
            history_texts = [m["content"] for m in st.session_state.messages]
            detected = detect_use_case(
                prompt, history_texts,
                emotion=quick_ctx.emotion,
                intent=quick_ctx.intent,
            )
            st.session_state.use_case_mode = detected

        use_case = get_mode(st.session_state.use_case_mode)
        tone = st.session_state.tone

        if use_case.tone_override:
            tone = use_case.tone_override

        conversation_turns: list[Turn] = []
        for msg in st.session_state.messages[:-1]:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_turns.append(Turn(role=role, text=msg["content"]))

        gen: ResponseGenerator = st.session_state.response_generator
        memory_summary = memory.get_long_term_summary()

        with st.chat_message("assistant"):
            with st.spinner(""):
                result = gen.generate_full(
                    persona=st.session_state.persona,
                    query=prompt,
                    tone=tone,
                    conversation_history=conversation_turns if conversation_turns else None,
                    use_case=use_case,
                    memory_summary=memory_summary,
                    validate=True,
                )

            response = result.response
            if not response:
                response = "I'd like to help with that. Could you tell me a bit more?"
                result.response = response

            st.markdown(response)

            if result.used_fallback and config.openai.use_api:
                st.caption("_API unavailable — using template response_")

            if st.session_state.show_explanations:
                _render_explanation(result)

        memory.add_turn("assistant", response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "result": result,
        })


def main() -> None:
    st.set_page_config(
        page_title="EchoPersona",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_state()
    _render_sidebar()
    _render_chat()


if __name__ == "__main__":
    main()
