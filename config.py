from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class LLMAPIConfig:
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model: str = "llama-3.3-70b-versatile"
    base_url: str = "https://api.groq.com/openai/v1"
    max_tokens: int = 256

    @property
    def use_api(self) -> bool:
        return bool(self.api_key)


@dataclass
class ModelConfig:
    base_llm: str = "gpt2"
    persona_model: str = "google/flan-t5-small"
    sentence_transformer: str = "all-MiniLM-L6-v2"


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["c_attn", "c_proj"])
    tones: list[str] = field(default_factory=lambda: ["formal", "sarcastic", "empathetic"])


@dataclass
class GenerationConfig:
    max_length: int = 256
    max_validation_retries: int = 3
    min_relevance_threshold: float = 0.3
    formal: dict = field(default_factory=lambda: {"temperature": 0.7, "top_p": 0.9})
    sarcastic: dict = field(default_factory=lambda: {"temperature": 0.95, "top_p": 0.95})
    empathetic: dict = field(default_factory=lambda: {"temperature": 0.8, "top_p": 0.9})

    def params_for_tone(self, tone: str) -> dict:
        defaults = {"temperature": 0.8, "top_p": 0.9}
        return getattr(self, tone, defaults)


@dataclass
class UseCaseConfig:
    available_modes: list[str] = field(default_factory=lambda: [
        "general",
        "virtual_assistant",
        "mental_health",
        "educational_tutor",
        "customer_support",
        "gaming_npc",
        "email_generator",
        "social_media",
        "ai_companion",
    ])
    default_mode: str = "general"
    auto_detect: bool = True


@dataclass
class ContextConfig:
    max_keyword_count: int = 10
    intent_confidence_threshold: float = 0.3
    emotion_labels: list[str] = field(default_factory=lambda: [
        "neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted",
    ])


@dataclass
class MemoryConfig:
    max_short_term: int = 20
    max_long_term: int = 100
    summary_interval: int = 10


@dataclass
class PathConfig:
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    personachat_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "personachat")
    synthetic_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "synthetic")
    personas_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "personas")
    adapter_dir: Path = field(default_factory=lambda: BASE_DIR / "lora_adapters")
    models_dir: Path = field(default_factory=lambda: BASE_DIR / "models")
    sample_histories: Path = field(
        default_factory=lambda: BASE_DIR / "data" / "sample_histories.json"
    )

    def adapter_path(self, tone: str) -> Path:
        return self.adapter_dir / tone


@dataclass
class TrainingConfig:
    persona_lr: float = 5e-5
    persona_epochs: int = 3
    persona_batch_size: int = 8
    lora_lr: float = 2e-4
    lora_epochs: int = 5
    lora_batch_size: int = 4
    gradient_accumulation_steps: int = 4


@dataclass
class AppConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    openai: LLMAPIConfig = field(default_factory=LLMAPIConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    use_cases: UseCaseConfig = field(default_factory=UseCaseConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


config = AppConfig()
