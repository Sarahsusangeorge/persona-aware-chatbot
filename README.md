# Universal Persona-Aware Response Generator

An advanced NLP pipeline that extracts user personality profiles from chat history and generates **context-aware, persona-consistent, multi-use-case** responses with dynamic tone switching, served through a Streamlit chat interface.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Stage 1: Persona Generator                    │
│                                                                  │
│  Chat History ──► flan-t5-small (Encoder-Decoder) ──► Persona   │
│   (raw msgs)        fine-tuned for extraction         Profile    │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Stage 2: Context Analyzer                      │
│                                                                  │
│  User Query ──► Intent Detection ──► Keywords ──► Context       │
│                 Emotion Detection      Entities    Features      │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Stage 3: Response Generator                    │
│                                                                  │
│  Persona Profile ──┐                                             │
│  Context Features ─┤──► Prompt Builder ──► GPT-2 + LoRA ──►    │
│  Use-Case Mode ────┤        │              ▲               ┌──┐ │
│  Memory Summary ───┘        │         LoRA Switcher        │  │ │
│                             │    ┌────────┬──────────┐     │  │ │
│                             │    formal  sarcastic  empathetic │ │
│                             │                              │  │ │
│                     Relevance Checker ◄────────────────────┘  │ │
│                     (validate → regenerate if off-topic)      │ │
└──────────────────────────────┬────────────────────────────────┘ │
                               ▼                                   │
┌──────────────────────────────────────────────────────────────────┘
│                     Streamlit Frontend                            │
│                                                                  │
│  Sidebar: history, persona, use-case mode, tone, memory         │
│  Main:    chat interface with explanations & validation          │
└──────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Context-Aware Responses** — Every response directly addresses the user's query using intent detection, keyword extraction, and semantic relevance scoring
- **Persona Consistency** — Extracted personality traits, tone preferences, and communication style persist across the conversation
- **Multi-Use-Case Modes** — 9 built-in domain modes that dynamically adapt response behavior:
  - General Assistant, Virtual Assistant, Mental Health Support, Educational Tutor, Customer Support, Gaming NPC, Email Generator, Social Media Assistant, AI Companion
- **Auto-Detection** — Automatically infers the best use-case mode from the conversation
- **Response Validation** — Validates relevance and regenerates if the response is off-topic or generic
- **Conversation Memory** — Short-term (recent turns) and long-term (extracted facts) memory system
- **Dynamic Tone Switching** — Zero-overhead LoRA adapter swapping between formal, sarcastic, and empathetic styles
- **Response Explanations** — Optional transparency panel showing detected intent, emotion, relevance score, and generation metadata

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. A CUDA GPU is recommended but not required — the models run on CPU.

### 2. Prepare data

```bash
python -m training.prepare_data
```

Downloads PersonaChat from HuggingFace, builds persona-extraction pairs for T5 training, creates tone-labelled subsets for LoRA training, and generates synthetic samples for underrepresented tones.

### 3. Train models

```bash
# Fine-tune T5 for persona extraction
python -m training.train_persona

# Train LoRA adapters for all three tones
python -m training.train_lora
```

Training parameters are configured in `config.py` and can be overridden via CLI flags (run with `--help`).

### 4. Launch the app

```bash
streamlit run app.py
```

## Project Structure

```
nlp project/
├── app.py                           # Streamlit chat frontend
├── config.py                        # Central config (models, paths, hyperparams, use-cases)
├── requirements.txt
├── README.md
│
├── context_analyzer/                # NEW: Context understanding module
│   ├── __init__.py
│   ├── analyzer.py                  # Intent, emotion, keyword, entity extraction
│   └── relevance.py                 # Response validation & relevance enforcement
│
├── use_cases/                       # NEW: Multi-use-case support
│   ├── __init__.py
│   ├── modes.py                     # 9 domain mode definitions with auto-detection
│   └── memory.py                    # Short-term + long-term conversation memory
│
├── persona_generator/
│   ├── __init__.py
│   ├── persona_schema.py            # Pydantic PersonaProfile model with serialization
│   └── generator.py                 # T5-based persona extraction with caching
│
├── response_generator/
│   ├── __init__.py
│   ├── adapter_manager.py           # LoRA adapter loading and dynamic switching
│   ├── prompt_builder.py            # Prompt construction with context + use-case sections
│   └── generator.py                 # GPT-2 generation with validation loop
│
├── training/
│   ├── __init__.py
│   ├── prepare_data.py              # Dataset loading, tone classification, synthetic generation
│   ├── train_persona.py             # Fine-tune flan-t5-small for persona extraction
│   └── train_lora.py                # Train LoRA adapters (formal, sarcastic, empathetic)
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py                  # BLEU, ROUGE-L, BERTScore, context relevance, full pipeline
│   └── persona_consistency.py       # Cosine-similarity persona scorer + engagement metrics
│
├── data/
│   ├── sample_histories.json        # 10 pre-built demo chat histories across domains
│   ├── personas/                    # Saved persona profile JSON files
│   ├── personachat/                 # PersonaChat dataset (auto-downloaded)
│   └── synthetic/                   # Hand-crafted synthetic conversations per tone
│
├── models/                          # Saved fine-tuned T5 model (created by train_persona)
└── lora_adapters/                   # LoRA adapter weights (created by train_lora)
    ├── formal/
    ├── sarcastic/
    └── empathetic/
```

## How Context-Awareness Works

The system solves the critical problem of generic/irrelevant responses through a multi-layer approach:

### 1. Context Understanding (Stage 2)

The `ContextAnalyzer` extracts from each user query:
- **Intent** — question, request, opinion, greeting, complaint, emotional sharing, etc.
- **Emotion** — happy, sad, angry, fearful, surprised, neutral
- **Keywords** — content-bearing terms after stop-word removal
- **Entities** — proper nouns, dates, times, money, emails
- **Topic** — primary subject distilled from keywords

### 2. Enhanced Prompt Engineering

Prompts now include structured context sections:

```
[Persona]: Enthusiastic outdoor lover who communicates warmly.
[User Intent]: Intent: question. Keywords: outdoor, activities. Emotion: neutral.
[Conversation Context]: Known about user: hobby: kayaking
[Use Case]: Educational Tutor — Act as a patient and encouraging tutor...
[Style]: Empathetic — Respond with warmth and empathy.
[Instruction]: Generate a relevant and persona-consistent response that directly answers the user's question.
[Conversation]:
User: What other outdoor activities do you enjoy?
Assistant:
```

### 3. Relevance Enforcement

The `RelevanceChecker` validates each response:
- **Semantic similarity** (cosine distance) between query and response embeddings
- **Keyword overlap** — checks that response addresses query topics
- **Generic detection** — flags canned/empty responses
- **Question-answering check** — ensures questions get real answers

If validation fails, the generator retries with slightly higher temperature (up to 3 attempts), keeping the best-scoring response.

## Use-Case Modes

| Mode | Description | Tone Override | Key Behaviors |
|------|-------------|---------------|---------------|
| General | Default assistant | — | Balanced, helpful |
| Virtual Assistant | Task-oriented | — | Actionable steps, organized |
| Mental Health | Emotional support | Empathetic | Validates feelings, non-judgmental |
| Educational Tutor | Teaching | — | Step-by-step, examples, encouragement |
| Customer Support | Issue resolution | Formal | Solution-oriented, professional |
| Gaming NPC | In-game character | — | Stay in character, lore-aware |
| Email Generator | Email drafting | Formal | Structured, greeting/closing |
| Social Media | Content creation | — | Engaging, hashtags, hooks |
| AI Companion | Long-term chat | — | Memory-aware, rapport-building |

Modes can be manually selected or auto-detected from conversation context using keyword matching.

## Models

| Component          | Model                | Parameters | Purpose                                    |
|--------------------|----------------------|------------|--------------------------------------------|
| Persona extractor  | `flan-t5-small`      | 80M        | Analyze chat history → structured profile  |
| Response generator | `gpt2`               | 124M       | Generate persona-aware responses           |
| LoRA adapters      | rank=8, alpha=32     | ~2MB each  | Tone-specific style control                |
| Embeddings         | `all-MiniLM-L6-v2`  | 22M        | Context relevance + persona consistency    |

## Evaluation

Run the evaluation suite on a JSON file of reference-hypothesis pairs:

```bash
python -m evaluation.evaluate test_results.json --output results.json
```

### Metrics

| Metric                    | Library                 | Description                                              |
|---------------------------|-------------------------|----------------------------------------------------------|
| BLEU (corpus + sentence)  | `nltk`                  | N-gram overlap between reference and generated text      |
| ROUGE-L (F1)              | `rouge-score`           | Longest common subsequence overlap                       |
| BERTScore (P/R/F1)        | `bert-score`            | Contextual embedding similarity                          |
| Persona Consistency       | `sentence-transformers` | Cosine similarity between persona and response embeddings |
| Context Relevance         | `sentence-transformers` | Cosine similarity between query and response embeddings  |
| Engagement (composite)    | custom                  | Weighted: length + question rate + vocabulary diversity   |

### Evaluation file format

```json
[
  {
    "query": "What other outdoor activities do you enjoy?",
    "reference": "I also enjoy rock climbing and kayaking.",
    "hypothesis": "In addition to hiking, I regularly go rock climbing and have been getting into kayaking.",
    "persona": {
      "personality_traits": ["friendly", "outdoorsy"],
      "tone_preference": "casual",
      "communication_style": "warm",
      "emotional_tendency": "optimistic",
      "summary": "An enthusiastic outdoor lover."
    }
  }
]
```

Flags: `--skip-bert-score` (faster), `--skip-context-relevance`.

## Programmatic Usage

```python
from persona_generator import PersonaGenerator
from response_generator import ResponseGenerator
from use_cases import get_mode

# Extract persona from chat history
pg = PersonaGenerator()
history = [
    "Hey! How's it going?",
    "I'm doing great! Just got back from hiking.",
    "That sounds awesome! Where did you go?",
    "The nature reserve — wildflowers are blooming!",
]
persona = pg.generate(history)

# Generate a context-aware response with use-case mode
rg = ResponseGenerator()
result = rg.generate_full(
    persona=persona,
    query="What other outdoor activities do you enjoy?",
    tone="empathetic",
    use_case=get_mode("ai_companion"),
    validate=True,
)
print(result.response)
print(f"Relevance: {result.validation.relevance_score:.2f}")
print(f"Intent: {result.context.intent}")
```

## Configuration

All settings in `config.py` as nested dataclasses:

```python
from config import config

config.models.base_llm                    # "gpt2"
config.models.persona_model               # "google/flan-t5-small"
config.lora.rank                          # 8
config.generation.max_length              # 256
config.generation.max_validation_retries  # 3
config.generation.min_relevance_threshold # 0.3
config.use_cases.auto_detect              # True
config.memory.max_short_term              # 20
config.memory.max_long_term               # 100
```

## Key Design Decisions

- **GPT-2 over TinyLlama** — No authentication required, smaller memory footprint, battle-tested PEFT/LoRA support
- **flan-t5-small for persona extraction** — Instruction-tuned text-to-text format naturally fits profile generation
- **Three separate LoRA adapters** — True dynamic switching via `set_adapter()`, each ~2MB, trained independently
- **Validation loop** — Regenerates off-topic responses up to 3 times with progressive temperature increase
- **Sentence-transformers for relevance** — Lightweight semantic similarity using `all-MiniLM-L6-v2` enables real-time validation without a separate LLM call
- **Heuristic use-case detection** — Keyword matching is fast and interpretable for mode selection; a classifier could be added for higher accuracy
- **Streamlit over React** — Faster to build, native Python integration, `st.chat_message` provides a polished chat UI

## Requirements

- Python 3.10+
- PyTorch 2.0+
- 8GB+ RAM (16GB recommended for training)
- CUDA GPU optional but recommended for training

See `requirements.txt` for full dependency list.

## License

This project is for educational and research purposes.
