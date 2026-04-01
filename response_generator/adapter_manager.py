from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import config

logger = logging.getLogger(__name__)


class AdapterManager:
    """Manages a GPT-2 base model with dynamically switchable LoRA adapters.

    Adapters are lazy-loaded on first use and switched via
    ``model.set_adapter()`` for zero-overhead tone changes.
    """

    def __init__(
        self,
        base_model_name: Optional[str] = None,
        adapter_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.base_model_name = base_model_name or config.models.base_llm
        self.adapter_dir = Path(adapter_dir) if adapter_dir else config.paths.adapter_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer: Optional[AutoTokenizer] = None
        self._base_model: Optional[AutoModelForCausalLM] = None
        self._peft_model: Optional[PeftModel] = None
        self._loaded_adapters: set[str] = set()
        self._active_tone: Optional[str] = None

    # ------------------------------------------------------------------
    # Lazy-loaded base artefacts
    # ------------------------------------------------------------------

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    @property
    def base_model(self) -> AutoModelForCausalLM:
        if self._base_model is None:
            logger.info("Loading base model: %s", self.base_model_name)
            self._base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name
            ).to(self.device)
        return self._base_model

    @property
    def model(self) -> PeftModel | AutoModelForCausalLM:
        """Return the PEFT-wrapped model if any adapter is loaded, else the base."""
        return self._peft_model if self._peft_model is not None else self.base_model

    # ------------------------------------------------------------------
    # Adapter lifecycle
    # ------------------------------------------------------------------

    def adapter_path(self, tone: str) -> Path:
        return self.adapter_dir / tone

    def adapter_available(self, tone: str) -> bool:
        path = self.adapter_path(tone)
        return path.exists() and any(path.iterdir())

    def load_adapter(self, tone: str) -> None:
        """Load a LoRA adapter for *tone*. Skips if already loaded."""
        if tone in self._loaded_adapters:
            return

        path = self.adapter_path(tone)
        if not self.adapter_available(tone):
            logger.warning(
                "No adapter weights found at %s — tone '%s' will use base model.",
                path,
                tone,
            )
            return

        if self._peft_model is None:
            logger.info("Attaching first adapter '%s' from %s", tone, path)
            self._peft_model = PeftModel.from_pretrained(
                self.base_model, str(path), adapter_name=tone
            )
            self._peft_model.to(self.device)
        else:
            logger.info("Loading additional adapter '%s' from %s", tone, path)
            self._peft_model.load_adapter(str(path), adapter_name=tone)

        self._loaded_adapters.add(tone)
        self._active_tone = tone
        logger.info("Adapter '%s' loaded and active.", tone)

    def switch_adapter(self, tone: str) -> None:
        """Activate the adapter for *tone*, loading it first if necessary."""
        if tone == self._active_tone:
            return

        if tone not in self._loaded_adapters:
            self.load_adapter(tone)

        if tone in self._loaded_adapters and self._peft_model is not None:
            self._peft_model.set_adapter(tone)
            self._active_tone = tone
            logger.info("Switched to adapter '%s'.", tone)
        else:
            logger.info(
                "No adapter for '%s'; generation will use the base model.", tone
            )
            self._active_tone = None

    def preload_all(self) -> None:
        """Load every available adapter up front (useful at app startup)."""
        for tone in config.lora.tones:
            if self.adapter_available(tone):
                self.load_adapter(tone)

    # ------------------------------------------------------------------
    # Convenience: build a fresh LoRA config (used by training scripts)
    # ------------------------------------------------------------------

    @staticmethod
    def make_lora_config(
        rank: Optional[int] = None,
        alpha: Optional[int] = None,
        dropout: Optional[float] = None,
        target_modules: Optional[list[str]] = None,
    ) -> LoraConfig:
        return LoraConfig(
            r=rank or config.lora.rank,
            lora_alpha=alpha or config.lora.alpha,
            lora_dropout=dropout if dropout is not None else config.lora.dropout,
            target_modules=target_modules or config.lora.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def get_trainable_model(self, tone: str) -> PeftModel:
        """Wrap the base model with a fresh LoRA config for training."""
        lora_cfg = self.make_lora_config()
        peft_model = get_peft_model(self.base_model, lora_cfg)
        peft_model.print_trainable_parameters()
        return peft_model

    # ------------------------------------------------------------------
    # State info
    # ------------------------------------------------------------------

    @property
    def active_tone(self) -> str | None:
        return self._active_tone

    @property
    def loaded_tones(self) -> list[str]:
        return sorted(self._loaded_adapters)

    def __repr__(self) -> str:
        return (
            f"AdapterManager(base={self.base_model_name!r}, "
            f"loaded={self.loaded_tones}, active={self._active_tone!r})"
        )
