"""Fine-tune flan-t5-small to extract structured persona profiles from chat history.

Input  : conversation text
Output : structured profile (traits, tone, style, emotion, summary)

Uses Seq2SeqTrainer from HuggingFace Transformers.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from config import config

logger = logging.getLogger(__name__)

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256


def load_data(data_path: Optional[Path] = None) -> Dataset:
    """Load the persona-extraction dataset produced by ``prepare_data.py``."""
    path = data_path or config.paths.personachat_dir / "persona_extraction"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run prepare_data.py first."
        )
    ds = load_from_disk(str(path))
    logger.info("Loaded persona-extraction dataset: %d examples", len(ds))
    return ds


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
) -> Dataset:
    """Tokenize input-target pairs for Seq2Seq training."""

    def _tokenize(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    return tokenized


def train(
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    eval_fraction: float = 0.1,
) -> Path:
    """Run the full T5 persona-generator fine-tuning loop.

    Returns the path to the saved model directory.
    """
    model_name = model_name or config.models.persona_model
    output_dir = output_dir or config.paths.models_dir / "persona_t5"
    epochs = epochs or config.training.persona_epochs
    batch_size = batch_size or config.training.persona_batch_size
    learning_rate = learning_rate or config.training.persona_lr

    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    dataset = load_data(data_path)
    tokenized = tokenize_dataset(dataset, tokenizer)

    split = tokenized.train_test_split(test_size=eval_fraction, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    logger.info("Train: %d, Eval: %d", len(train_ds), len(eval_ds))

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting T5 persona training …")
    trainer.train()

    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    logger.info("Best model saved to %s", best_dir)

    return best_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune T5 for persona extraction")
    parser.add_argument("--data-path", type=Path, default=None,
                        help="Path to persona_extraction dataset folder")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to save the fine-tuned model")
    parser.add_argument("--model", type=str, default=None,
                        help="Base T5 model name (default: flan-t5-small)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_fraction=args.eval_fraction,
    )
