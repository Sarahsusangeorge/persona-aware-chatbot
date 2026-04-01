"""Train LoRA adapters for GPT-2 -- one adapter per tone (formal, sarcastic, empathetic).

Each adapter is small (~2 MB) and can be hot-swapped at inference time via
``AdapterManager.switch_adapter()``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from config import config

logger = logging.getLogger(__name__)

MAX_SEQ_LENGTH = 512


# ── Dataset helpers ──────────────────────────────────────────────────

def load_tone_dataset(tone: str, data_dir: Optional[Path] = None) -> Dataset:
    """Load the pre-processed LoRA dataset for a single tone."""
    base = data_dir or config.paths.personachat_dir
    path = base / f"lora_{tone}"
    if not path.exists():
        raise FileNotFoundError(
            f"Tone dataset not found at {path}. Run prepare_data.py first."
        )
    ds = load_from_disk(str(path))
    logger.info("Loaded LoRA dataset for '%s': %d samples", tone, len(ds))
    return ds


def tokenize_for_clm(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
) -> Dataset:
    """Tokenize the ``text`` column for causal language modelling."""

    def _tokenize(batch):
        encodings = tokenizer(
            batch["text"],
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding="max_length",
        )
        encodings["labels"] = encodings["input_ids"].copy()
        return encodings

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing for CLM",
    )
    return tokenized


# ── Single-tone training ─────────────────────────────────────────────

def train_single_tone(
    tone: str,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    base_model_name: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    eval_fraction: float = 0.1,
) -> Path:
    """Train a LoRA adapter for one tone and save it."""
    logger.info("═" * 60)
    logger.info("Training LoRA adapter for tone: %s", tone)
    logger.info("═" * 60)

    model = AutoModelForCausalLM.from_pretrained(base_model_name)

    lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    tokenized = tokenize_for_clm(dataset, tokenizer)

    if len(tokenized) > 10:
        split = tokenized.train_test_split(test_size=eval_fraction, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = tokenized, None

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    tone_output = output_dir / tone
    tone_output.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(tone_output / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=eval_ds is not None,
        logging_steps=25,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train()

    adapter_save_path = config.paths.adapter_path(tone)
    adapter_save_path.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_save_path))
    logger.info("Adapter for '%s' saved to %s", tone, adapter_save_path)

    del peft_model, model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return adapter_save_path


# ── Full pipeline ────────────────────────────────────────────────────

def train_all(
    tones: Optional[list[str]] = None,
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    gradient_accumulation_steps: Optional[int] = None,
    eval_fraction: float = 0.1,
) -> dict[str, Path]:
    """Train LoRA adapters for all specified tones.

    Returns a dict mapping tone name to the path where the adapter was saved.
    """
    tones = tones or config.lora.tones
    base_model_name = config.models.base_llm
    output_dir = output_dir or config.paths.adapter_dir
    epochs = epochs or config.training.lora_epochs
    batch_size = batch_size or config.training.lora_batch_size
    learning_rate = learning_rate or config.training.lora_lr
    gradient_accumulation_steps = (
        gradient_accumulation_steps or config.training.gradient_accumulation_steps
    )

    logger.info("Loading tokenizer for %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    saved: dict[str, Path] = {}

    for tone in tones:
        try:
            dataset = load_tone_dataset(tone, data_dir)
        except FileNotFoundError:
            logger.warning("Skipping tone '%s' — no dataset found.", tone)
            continue

        adapter_path = train_single_tone(
            tone=tone,
            dataset=dataset,
            tokenizer=tokenizer,
            base_model_name=base_model_name,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_fraction=eval_fraction,
        )
        saved[tone] = adapter_path

    logger.info("All adapters trained: %s", list(saved.keys()))
    return saved


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA adapters per tone")
    parser.add_argument("--tones", nargs="+", default=None,
                        help="Subset of tones to train (default: all)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory containing lora_<tone> datasets")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Root directory for adapter output")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    train_all(
        tones=args.tones,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        eval_fraction=args.eval_fraction,
    )
