"""Evaluation suite: BLEU, ROUGE-L, BERTScore, persona consistency,
context relevance, and engagement.

Run as a script to benchmark a model on a test set and print a summary table.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nltk

nltk.download("punkt_tab", quiet=True)

from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from rouge_score import rouge_scorer

from config import config
from evaluation.persona_consistency import (
    PersonaConsistencyScorer,
    engagement_score,
)
from persona_generator.persona_schema import PersonaProfile

logger = logging.getLogger(__name__)


def compute_bleu(
    references: list[str],
    hypotheses: list[str],
    n: int = 4,
) -> dict[str, float]:
    """Corpus-level and average sentence-level BLEU up to *n*-grams."""
    smooth = SmoothingFunction().method1
    ref_tokens = [[r.split()] for r in references]
    hyp_tokens = [h.split() for h in hypotheses]

    weights_map = {
        1: (1.0,),
        2: (0.5, 0.5),
        3: (1 / 3, 1 / 3, 1 / 3),
        4: (0.25, 0.25, 0.25, 0.25),
    }
    weights = weights_map.get(n, (0.25, 0.25, 0.25, 0.25))

    corpus = corpus_bleu(ref_tokens, hyp_tokens, weights=weights, smoothing_function=smooth)

    sent_scores = [
        sentence_bleu(ref, hyp, weights=weights, smoothing_function=smooth)
        for ref, hyp in zip(ref_tokens, hyp_tokens)
    ]
    avg_sent = sum(sent_scores) / len(sent_scores) if sent_scores else 0.0

    return {
        "bleu_corpus": round(corpus, 4),
        "bleu_avg_sentence": round(avg_sent, 4),
    }


def compute_rouge_l(
    references: list[str],
    hypotheses: list[str],
) -> dict[str, float]:
    """Average ROUGE-L F1 score across pairs."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, hyp)["rougeL"].fmeasure for ref, hyp in zip(references, hypotheses)]
    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "rouge_l_f1": round(avg, 4),
        "rouge_l_scores": [round(s, 4) for s in scores],
    }


def compute_bert_score(
    references: list[str],
    hypotheses: list[str],
    lang: str = "en",
) -> dict[str, float]:
    """BERTScore (precision, recall, F1) averaged over pairs."""
    P, R, F1 = bert_score_fn(hypotheses, references, lang=lang, verbose=False)
    return {
        "bert_score_precision": round(P.mean().item(), 4),
        "bert_score_recall": round(R.mean().item(), 4),
        "bert_score_f1": round(F1.mean().item(), 4),
    }


def compute_persona_consistency(
    persona: PersonaProfile,
    responses: list[str],
    scorer: Optional[PersonaConsistencyScorer] = None,
) -> dict[str, float]:
    """Average persona-consistency cosine similarity."""
    scorer = scorer or PersonaConsistencyScorer()
    scores = scorer.score_batch(persona, responses)
    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "persona_consistency_avg": round(avg, 4),
        "persona_consistency_scores": [round(s, 4) for s in scores],
    }


def compute_context_relevance(
    queries: list[str],
    hypotheses: list[str],
) -> dict[str, float]:
    """Average semantic similarity between user queries and responses.

    This is the custom Context Relevance Score that measures how well
    each response addresses the corresponding query.
    """
    from context_analyzer.relevance import RelevanceChecker

    checker = RelevanceChecker()
    scores = [
        checker.score_relevance(q, h) for q, h in zip(queries, hypotheses)
    ]
    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "context_relevance_avg": round(avg, 4),
        "context_relevance_scores": [round(s, 4) for s in scores],
    }


def compute_engagement(responses: list[str]) -> dict[str, float]:
    """Average engagement metrics across all responses."""
    all_scores = [engagement_score(r) for r in responses]
    if not all_scores:
        return {"engagement_composite": 0.0}

    keys = all_scores[0].keys()
    averages = {}
    for key in keys:
        vals = [s[key] for s in all_scores]
        averages[f"engagement_{key}"] = round(sum(vals) / len(vals), 4)
    return averages


@dataclass
class EvaluationResult:
    bleu: dict[str, float] = field(default_factory=dict)
    rouge: dict[str, float] = field(default_factory=dict)
    bert: dict[str, float] = field(default_factory=dict)
    persona: dict[str, float] = field(default_factory=dict)
    context_relevance: dict[str, float] = field(default_factory=dict)
    engagement: dict[str, float] = field(default_factory=dict)
    num_samples: int = 0

    def summary(self) -> dict[str, float]:
        """Flat dict of headline numbers for easy comparison."""
        out: dict[str, float] = {"num_samples": self.num_samples}
        for section in [
            self.bleu, self.rouge, self.bert,
            self.persona, self.context_relevance, self.engagement,
        ]:
            for k, v in section.items():
                if isinstance(v, float):
                    out[k] = v
        return out

    def print_table(self) -> None:
        """Pretty-print a summary table to stdout."""
        summary = self.summary()
        max_key = max(len(k) for k in summary)
        print("\n" + "=" * (max_key + 18))
        print(f" {'Metric':<{max_key}}   {'Value':>10}")
        print("-" * (max_key + 18))
        for key, val in summary.items():
            if isinstance(val, float):
                print(f" {key:<{max_key}}   {val:>10.4f}")
            else:
                print(f" {key:<{max_key}}   {val!s:>10}")
        print("=" * (max_key + 18) + "\n")

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        logger.info("Results saved to %s", path)


def evaluate(
    references: list[str],
    hypotheses: list[str],
    queries: Optional[list[str]] = None,
    persona: Optional[PersonaProfile] = None,
    skip_bert_score: bool = False,
    skip_context_relevance: bool = False,
) -> EvaluationResult:
    """Run all metrics on reference-hypothesis pairs.

    Args:
        references: Ground-truth responses.
        hypotheses: Model-generated responses.
        queries: Original user queries (for context relevance scoring).
        persona: If provided, persona-consistency is computed.
        skip_bert_score: Skip BERTScore (slow on CPU).
        skip_context_relevance: Skip context relevance scoring.

    Returns:
        An ``EvaluationResult`` with all metric dictionaries populated.
    """
    assert len(references) == len(hypotheses), "Mismatched reference/hypothesis counts"

    result = EvaluationResult(num_samples=len(references))

    logger.info("Computing BLEU ...")
    result.bleu = compute_bleu(references, hypotheses)

    logger.info("Computing ROUGE-L ...")
    result.rouge = compute_rouge_l(references, hypotheses)

    if not skip_bert_score:
        logger.info("Computing BERTScore ...")
        result.bert = compute_bert_score(references, hypotheses)
    else:
        logger.info("Skipping BERTScore (--skip-bert-score flag set)")

    if persona is not None:
        logger.info("Computing persona consistency ...")
        result.persona = compute_persona_consistency(persona, hypotheses)

    if queries and not skip_context_relevance:
        assert len(queries) == len(hypotheses), "Mismatched query/hypothesis counts"
        logger.info("Computing context relevance ...")
        result.context_relevance = compute_context_relevance(queries, hypotheses)
    else:
        logger.info("Skipping context relevance (no queries provided or flag set)")

    logger.info("Computing engagement ...")
    result.engagement = compute_engagement(hypotheses)

    return result


def load_eval_file(path: Path) -> list[dict]:
    """Load a JSON-lines or JSON-array evaluation file.

    Expected fields per entry: ``reference``, ``hypothesis``,
    and optionally ``persona`` (a dict matching PersonaProfile),
    ``query`` (the original user input).
    """
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate generated responses")
    p.add_argument("eval_file", type=Path,
                   help="JSON/JSONL file with reference and hypothesis fields")
    p.add_argument("--output", type=Path, default=None,
                   help="Path to save the results JSON")
    p.add_argument("--skip-bert-score", action="store_true",
                   help="Skip BERTScore computation (faster)")
    p.add_argument("--skip-context-relevance", action="store_true",
                   help="Skip context relevance scoring")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    entries = load_eval_file(args.eval_file)
    references = [e["reference"] for e in entries]
    hypotheses = [e["hypothesis"] for e in entries]

    queries = None
    if "query" in entries[0]:
        queries = [e["query"] for e in entries]

    persona = None
    if "persona" in entries[0]:
        persona = PersonaProfile.from_dict(entries[0]["persona"])

    result = evaluate(
        references,
        hypotheses,
        queries=queries,
        persona=persona,
        skip_bert_score=args.skip_bert_score,
        skip_context_relevance=args.skip_context_relevance,
    )
    result.print_table()

    if args.output:
        result.to_json(args.output)
