"""Evaluate a model or LoRA checkpoint on a GSM8K subset."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from grpo_reasoner.data import EvalExample, load_gsm8k_examples
from grpo_reasoner.extract import answers_match, extract_answer
from grpo_reasoner.metrics import summarize
from grpo_reasoner.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_prompt

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT = Path("results/baseline.json")


@dataclass(frozen=True)
class EvalConfig:
    model: str = DEFAULT_MODEL
    adapter: str | None = None
    split: str = "test"
    limit: int = 50
    seed: int = 42
    max_new_tokens: int = 512
    batch_size: int = 1
    temperature: float = 0.0
    device: str = "auto"
    dtype: str = "auto"
    output: Path = DEFAULT_OUTPUT


def _resolve_device(requested: str):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(requested: str, device) -> Any:
    import torch

    if requested != "auto":
        return getattr(torch, requested)
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(config: EvalConfig):
    """Load the base model plus an optional PEFT adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _resolve_device(config.device)
    dtype = _resolve_dtype(config.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(config.model, dtype=dtype)
    if config.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, config.adapter)
    model.to(device)
    model.eval()
    return model, tokenizer, device, dtype


def generate_completions(
    model,
    tokenizer,
    questions: list[str],
    *,
    device,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    """Generate one completion per question using the fixed prompt."""
    import torch

    prompts = [build_prompt(question, tokenizer) for question in questions]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs.update(
            do_sample=True,
            temperature=temperature,
        )
    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generation_kwargs)

    generated_ids = output_ids[:, encoded["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def run_evaluation(config: EvalConfig, *, verbose: bool = False) -> dict[str, Any]:
    """Run the full evaluation and return a JSON-serializable result."""
    import torch

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    examples = load_gsm8k_examples(
        split=config.split,
        limit=config.limit,
        seed=config.seed,
    )
    model, tokenizer, device, dtype = load_model_and_tokenizer(config)
    print(
        f"Loaded {len(examples)} examples from GSM8K[{config.split}] "
        f"(seed={config.seed}, limit={config.limit})"
    )
    print(f"Model: {config.model} | device={device} | dtype={dtype}")

    records: list[dict[str, Any]] = []
    started = datetime.now(UTC)

    for start in range(0, len(examples), config.batch_size):
        batch: list[EvalExample] = examples[start : start + config.batch_size]
        completions = generate_completions(
            model,
            tokenizer,
            [example.question for example in batch],
            device=device,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )
        for example, completion in zip(batch, completions, strict=True):
            prediction = extract_answer(completion)
            correct = answers_match(prediction, example.gold)
            record = {
                "source_index": example.source_index,
                "question": example.question,
                "reference": example.reference,
                "gold": example.gold,
                "completion": completion,
                "prediction": prediction,
                "correct": correct,
            }
            records.append(record)
            position = len(records)
            marker = "OK " if correct else "MISS"
            print(
                f"[{position:>4}/{len(examples)}] {marker} "
                f"pred={prediction!s:<12} gold={example.gold}",
                flush=True,
            )
            if verbose:
                print(f"    completion: {completion.strip()[:500]!r}", flush=True)

    summary = summarize(records)
    elapsed = (datetime.now(UTC) - started).total_seconds()
    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": config.model,
        "adapter": config.adapter,
        "split": config.split,
        "seed": config.seed,
        "limit": config.limit,
        "prompt_version": PROMPT_VERSION,
        "prompt": SYSTEM_PROMPT,
        "generation": {
            "max_new_tokens": config.max_new_tokens,
            "batch_size": config.batch_size,
            "temperature": config.temperature,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
        },
        "elapsed_seconds": round(elapsed, 2),
        "summary": summary,
        "examples": records,
    }
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path")
    parser.add_argument("--adapter", default=None, help="Optional PEFT/LoRA adapter path")
    parser.add_argument("--split", default="test", help="GSM8K split (default: test)")
    parser.add_argument("--limit", type=int, default=50, help="Number of test examples")
    parser.add_argument("--seed", type=int, default=42, help="Subset/sampling seed")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy decoding")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")
    parser.add_argument("--dtype", default="auto", help="auto, float16, bfloat16, or float32")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verbose", action="store_true", help="Print full completions")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = EvalConfig(
        model=args.model,
        adapter=args.adapter,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
        device=args.device,
        dtype=args.dtype,
        output=args.output,
    )
    result = run_evaluation(config, verbose=args.verbose)

    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(result, indent=2) + "\n")

    summary = result["summary"]
    print(
        f"\nAccuracy: {summary['correct']}/{summary['total']} "
        f"= {summary['accuracy']:.2%} in {result['elapsed_seconds']:.1f}s"
    )
    print(f"Saved results to {config.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
