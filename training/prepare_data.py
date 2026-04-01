"""Data loading and preprocessing for PersonaChat.

Produces three artefact types:
1. **Persona-extraction pairs** for T5 fine-tuning
   (input = conversation text, target = structured persona profile).
2. **Tone-labelled prompt-completion pairs** for GPT-2 LoRA training
   (one subset per tone: formal, sarcastic, empathetic).
3. **Synthetic samples** to augment underrepresented tones.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset

from config import config

logger = logging.getLogger(__name__)

# ── Keyword heuristics for tone classification ────────────────────────

TONE_KEYWORDS: dict[str, list[str]] = {
    "formal": [
        "certainly", "regarding", "furthermore", "therefore", "accordingly",
        "i would suggest", "per our", "as per", "kindly", "shall we",
        "comprehensive", "preliminary", "subsequently", "in summary",
    ],
    "sarcastic": [
        "oh great", "oh sure", "yeah right", "shocking", "clearly",
        "totally", "wow", "genius", "obviously", "as if", "sure thing",
        "what a surprise", "lucky me", "how wonderful",
    ],
    "empathetic": [
        "i understand", "i'm sorry", "that must be", "feel free",
        "it's okay", "i appreciate", "take care", "sending you",
        "don't worry", "i hear you", "that sounds tough", "hugs",
        "you're not alone", "i can imagine",
    ],
}


def classify_tone(text: str) -> Optional[str]:
    """Return the best-matching tone label or None if inconclusive."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for tone, keywords in TONE_KEYWORDS.items():
        scores[tone] = sum(1 for kw in keywords if kw in text_lower)

    best_tone = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best_tone] == 0:
        return None
    second = sorted(scores.values(), reverse=True)[1]
    if scores[best_tone] <= second:
        return None
    return best_tone


# ── PersonaChat loading ──────────────────────────────────────────────

def load_personachat(split: str = "train") -> Dataset:
    """Load the PersonaChat dataset from HuggingFace."""
    ds = load_dataset("bavard/personachat_truecased", split=split)
    logger.info("Loaded PersonaChat %s split: %d examples", split, len(ds))
    return ds


# ── T5 persona-extraction pairs ─────────────────────────────────────

def _persona_strings_to_profile(persona_lines: list[str]) -> str:
    """Convert PersonaChat persona sentences into a structured profile string
    suitable as a T5 training target."""
    traits = []
    tone = "neutral"
    style = "balanced"
    emotion = "neutral"

    for line in persona_lines:
        line_lower = line.lower().strip()
        traits.append(line.strip().rstrip("."))

        if any(w in line_lower for w in ["formal", "professional", "polite"]):
            tone = "formal"
            style = "formal"
        elif any(w in line_lower for w in ["sarcas", "witty", "ironic", "joke"]):
            tone = "sarcastic"
            style = "casual"
        elif any(w in line_lower for w in ["empath", "caring", "kind", "support"]):
            tone = "empathetic"
            style = "warm"

        if any(w in line_lower for w in ["happy", "optimis", "cheerful", "love"]):
            emotion = "optimistic"
        elif any(w in line_lower for w in ["concern", "worr", "anxious"]):
            emotion = "empathetic"

    summary = ". ".join(persona_lines[:3])

    target = (
        f"Personality traits: {', '.join(traits[:5])}\n"
        f"Tone: {tone}\n"
        f"Communication style: {style}\n"
        f"Emotional tendency: {emotion}\n"
        f"Summary: {summary}"
    )
    return target


def build_persona_extraction_dataset(
    raw_dataset: Dataset,
    max_samples: Optional[int] = None,
) -> Dataset:
    """Build (conversation_text, structured_profile) pairs for T5 training."""
    inputs: list[str] = []
    targets: list[str] = []

    for i, example in enumerate(raw_dataset):
        if max_samples and i >= max_samples:
            break

        persona_lines: list[str] = example.get("personality", [])
        utterances: list[str] = example.get("utterances", [])
        if not utterances:
            history = example.get("history", [])
            candidates = example.get("candidates", [])
            conv_text = "\n".join(history + candidates[-1:]) if history else ""
        else:
            last_utt = utterances[-1] if utterances else {}
            history = last_utt.get("history", [])
            conv_text = "\n".join(history)

        if not conv_text or not persona_lines:
            continue

        prompt = (
            "Analyze the following chat history and extract a personality profile.\n"
            f"Chat history:\n{conv_text}\n\n"
            "Provide: personality traits, tone, communication style, "
            "emotional tendency, summary."
        )
        target = _persona_strings_to_profile(persona_lines)

        inputs.append(prompt)
        targets.append(target)

    ds = Dataset.from_dict({"input_text": inputs, "target_text": targets})
    logger.info("Built persona-extraction dataset: %d pairs", len(ds))
    return ds


# ── Tone-labelled prompt-completion pairs for LoRA ───────────────────

def _format_lora_sample(
    persona_lines: list[str],
    history: list[str],
    response: str,
    tone: str,
) -> dict[str, str]:
    """Create a single prompt-completion pair for LoRA fine-tuning."""
    persona_block = ". ".join(persona_lines[:3]) if persona_lines else "A conversational user."

    turns = []
    roles = ["User", "Assistant"]
    for i, msg in enumerate(history[-10:]):
        turns.append(f"{roles[i % 2]}: {msg}")

    prompt = (
        f"[Persona]: {persona_block}\n"
        f"[Style]: {tone.capitalize()}\n"
        "[Conversation]:\n"
        + "\n".join(turns)
        + "\nAssistant:"
    )
    return {"text": f"{prompt} {response}", "prompt": prompt, "completion": response}


def build_tone_datasets(
    raw_dataset: Dataset,
    max_per_tone: Optional[int] = None,
) -> dict[str, Dataset]:
    """Split PersonaChat into tone-labelled subsets for LoRA training.

    Each conversation is classified by keyword heuristics on the response text.
    Returns a dict mapping tone name to a Dataset of prompt-completion pairs.
    """
    tone_buckets: dict[str, list[dict]] = {t: [] for t in config.lora.tones}

    for example in raw_dataset:
        persona_lines: list[str] = example.get("personality", [])

        utterances = example.get("utterances", [])
        if not utterances:
            history = example.get("history", [])
            candidates = example.get("candidates", [])
            if not history or not candidates:
                continue
            response = candidates[-1] if isinstance(candidates[-1], str) else ""
        else:
            last_utt = utterances[-1] if utterances else {}
            history = last_utt.get("history", [])
            candidates = last_utt.get("candidates", [])
            if not candidates:
                continue
            response = candidates[-1] if isinstance(candidates[-1], str) else ""

        if not response:
            continue

        tone = classify_tone(response)
        if tone is None:
            persona_text = " ".join(persona_lines).lower()
            tone = classify_tone(persona_text)

        if tone is None:
            continue
        if max_per_tone and len(tone_buckets[tone]) >= max_per_tone:
            continue

        sample = _format_lora_sample(persona_lines, history, response, tone)
        tone_buckets[tone].append(sample)

    result: dict[str, Dataset] = {}
    for tone, samples in tone_buckets.items():
        if samples:
            result[tone] = Dataset.from_list(samples)
            logger.info("Tone '%s': %d samples", tone, len(samples))
        else:
            logger.warning("Tone '%s': 0 samples — will generate synthetic data", tone)

    return result


# ── Synthetic data generation ────────────────────────────────────────

SYNTHETIC_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "formal": [
        {"history": ["What are the project timelines?"], "response": "The project is scheduled for completion by the end of Q3. I shall provide a detailed breakdown at the next meeting."},
        {"history": ["Can you help with the report?"], "response": "Certainly. I would be pleased to assist with the quarterly report. Please forward the relevant documentation at your earliest convenience."},
        {"history": ["What do you think about the proposal?"], "response": "I find the proposal to be well-structured and comprehensive. However, I would recommend further analysis of the cost projections."},
        {"history": ["How should we proceed?"], "response": "I would suggest we convene a brief meeting to align on priorities. Subsequently, we can distribute tasks accordingly."},
        {"history": ["Is there anything else to discuss?"], "response": "I believe we have addressed the primary agenda items. Should any further matters arise, please do not hesitate to reach out."},
    ],
    "sarcastic": [
        {"history": ["Did you finish the assignment?"], "response": "Oh, you mean the one due three weeks ago? Yeah, I finished it somewhere between my existential crisis and my fourth cup of coffee."},
        {"history": ["The meeting is at 8am tomorrow."], "response": "8am? How delightful. Nothing says 'productive day' like being summoned before the sun has had its coffee."},
        {"history": ["Can you believe the weather today?"], "response": "Wow, rain again. What a shocking twist. I'm beginning to think the sky has commitment issues."},
        {"history": ["We need to work overtime this week."], "response": "Oh sure, because what I really needed was less sleep and more spreadsheets. Living the dream."},
        {"history": ["The new policy seems reasonable."], "response": "Sure, the same way a pop quiz on Monday morning seems 'reasonable.' The intent is there, the joy is not."},
    ],
    "empathetic": [
        {"history": ["I'm really stressed about the deadline."], "response": "I completely understand how you feel. Deadlines can be overwhelming. Remember, it's okay to ask for help if you need it."},
        {"history": ["I didn't get the promotion."], "response": "I'm sorry to hear that. It must be really disappointing. Your hard work is still valuable, and this doesn't define your worth."},
        {"history": ["I've been feeling lonely lately."], "response": "That sounds really tough, and I appreciate you sharing that with me. You're not alone in feeling this way, and I'm here for you."},
        {"history": ["My pet passed away yesterday."], "response": "Oh no, I'm so sorry for your loss. Losing a pet is like losing a family member. Take all the time you need to grieve."},
        {"history": ["I failed my exam."], "response": "I hear you, and I know that must feel discouraging. One exam doesn't define your abilities. What matters is that you keep going."},
    ],
}


def generate_synthetic_data(
    min_per_tone: int = 50,
    existing_counts: Optional[dict[str, int]] = None,
) -> dict[str, Dataset]:
    """Generate synthetic prompt-completion pairs for underrepresented tones.

    Templates are repeated and lightly varied to reach ``min_per_tone`` samples.
    """
    existing_counts = existing_counts or {}
    result: dict[str, Dataset] = {}

    persona_lines = ["I am a helpful conversational partner"]

    for tone, templates in SYNTHETIC_TEMPLATES.items():
        current_count = existing_counts.get(tone, 0)
        needed = max(0, min_per_tone - current_count)
        if needed == 0:
            continue

        samples: list[dict] = []
        idx = 0
        while len(samples) < needed:
            template = templates[idx % len(templates)]
            sample = _format_lora_sample(
                persona_lines,
                template["history"],
                template["response"],
                tone,
            )
            samples.append(sample)
            idx += 1

        result[tone] = Dataset.from_list(samples)
        logger.info("Generated %d synthetic samples for tone '%s'", len(samples), tone)

    return result


# ── Save / load helpers ──────────────────────────────────────────────

def save_dataset(dataset: Dataset, path: Path, name: str = "data") -> Path:
    out = path / name
    out.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out))
    logger.info("Saved dataset '%s' to %s (%d rows)", name, out, len(dataset))
    return out


def save_all(
    persona_ds: Dataset,
    tone_datasets: dict[str, Dataset],
    base_dir: Optional[Path] = None,
) -> None:
    base = base_dir or config.paths.personachat_dir
    base.mkdir(parents=True, exist_ok=True)

    save_dataset(persona_ds, base, "persona_extraction")
    for tone, ds in tone_datasets.items():
        save_dataset(ds, base, f"lora_{tone}")


# ── Main pipeline ────────────────────────────────────────────────────

def prepare(
    max_persona_samples: Optional[int] = None,
    max_per_tone: Optional[int] = None,
    min_synthetic_per_tone: int = 50,
) -> tuple[Dataset, dict[str, Dataset]]:
    """Run the full data preparation pipeline.

    Returns:
        (persona_extraction_dataset, {tone: lora_dataset})
    """
    logger.info("Loading PersonaChat …")
    raw_train = load_personachat("train")

    logger.info("Building persona-extraction pairs …")
    persona_ds = build_persona_extraction_dataset(raw_train, max_persona_samples)

    logger.info("Building tone-labelled LoRA datasets …")
    tone_datasets = build_tone_datasets(raw_train, max_per_tone)

    existing_counts = {t: len(ds) for t, ds in tone_datasets.items()}
    synthetic = generate_synthetic_data(min_synthetic_per_tone, existing_counts)

    from datasets import concatenate_datasets
    for tone, syn_ds in synthetic.items():
        if tone in tone_datasets:
            tone_datasets[tone] = concatenate_datasets([tone_datasets[tone], syn_ds])
        else:
            tone_datasets[tone] = syn_ds

    logger.info("Saving processed datasets …")
    save_all(persona_ds, tone_datasets)

    logger.info("Data preparation complete.")
    return persona_ds, tone_datasets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    prepare()
