"""GSM8K loading helpers."""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset

from grpo_reasoner.extract import extract_gold_answer

DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"


@dataclass(frozen=True)
class EvalExample:
    source_index: int
    question: str
    reference: str
    gold: str


def load_gsm8k_examples(
    split: str = "test",
    limit: int = 50,
    seed: int = 42,
    dataset_name: str = DATASET_NAME,
    dataset_config: str = DATASET_CONFIG,
) -> list[EvalExample]:
    """Load a deterministic subset of GSM8K with parsed gold answers."""
    if limit <= 0:
        raise ValueError("limit must be positive")

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    indexed = dataset.add_column("__source_index", list(range(len(dataset))))
    subset = indexed.shuffle(seed=seed).select(range(min(limit, len(indexed))))

    examples: list[EvalExample] = []
    for row in subset:
        question = row["question"].strip()
        reference = row["answer"].strip()
        gold = extract_gold_answer(reference)
        if gold is None:
            raise ValueError(f"Could not extract gold answer at index {row['__source_index']}")
        examples.append(
            EvalExample(
                source_index=int(row["__source_index"]),
                question=question,
                reference=reference,
                gold=gold,
            )
        )
    return examples
