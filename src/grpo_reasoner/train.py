"""GRPO training entry point with YAML configuration."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, load_dataset

from grpo_reasoner.data import DATASET_CONFIG, DATASET_NAME
from grpo_reasoner.extract import extract_gold_answer
from grpo_reasoner.prompts import build_messages
from grpo_reasoner.rewards import (
    correctness_reward,
    format_reward,
    length_penalty_reward,
)

DEFAULT_CONFIG = Path("configs/train_smoke.yaml")


@dataclass(frozen=True)
class ModelSettings:
    name: str
    dtype: str = "auto"


@dataclass(frozen=True)
class LoraSettings:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(frozen=True)
class DataSettings:
    split: str = "train"
    limit: int | None = None
    seed: int = 42


@dataclass(frozen=True)
class TrainingSettings:
    output_dir: str = "outputs/grpo"
    learning_rate: float = 1.0e-5
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = 200
    num_generations: int = 8
    max_completion_length: int = 512
    temperature: float = 1.0
    beta: float = 0.04
    logging_steps: int = 1
    save_steps: int = 50
    save_total_limit: int | None = 2
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    use_cpu: bool = False
    fp16: bool = False
    bf16: bool = False
    report_to: str | list[str] = "none"
    project: str = "grpo-small-reasoner"
    run_name: str | None = None
    log_completions: bool = True
    seed: int = 42


@dataclass(frozen=True)
class LengthPenaltySettings:
    enabled: bool = False
    target_chars: int = 2000
    weight: float = 0.1


@dataclass(frozen=True)
class RewardSettings:
    format_weight: float = 0.5
    length_penalty: LengthPenaltySettings = field(default_factory=LengthPenaltySettings)


@dataclass(frozen=True)
class HubSettings:
    push: bool = False
    repo_id: str | None = None
    private: bool = True


@dataclass(frozen=True)
class UnslothSettings:
    enabled: bool = False
    load_in_4bit: bool = False
    max_seq_length: int = 2048


@dataclass(frozen=True)
class TrainConfig:
    model: ModelSettings
    lora: LoraSettings = field(default_factory=LoraSettings)
    data: DataSettings = field(default_factory=DataSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    rewards: RewardSettings = field(default_factory=RewardSettings)
    hub: HubSettings = field(default_factory=HubSettings)
    unsloth: UnslothSettings = field(default_factory=UnslothSettings)


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping")
    return value


def load_config(path: str | Path) -> TrainConfig:
    """Load and validate a YAML training configuration."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("Training config must be a YAML mapping")

    rewards_raw = _section(raw, "rewards")
    length_raw = rewards_raw.get("length_penalty", {})
    if not isinstance(length_raw, dict):
        raise ValueError("rewards.length_penalty must be a mapping")

    config = TrainConfig(
        model=ModelSettings(**_section(raw, "model")),
        lora=LoraSettings(**_section(raw, "lora")),
        data=DataSettings(**_section(raw, "data")),
        training=TrainingSettings(**_section(raw, "training")),
        rewards=RewardSettings(
            format_weight=rewards_raw.get("format_weight", 0.5),
            length_penalty=LengthPenaltySettings(**length_raw),
        ),
        hub=HubSettings(**_section(raw, "hub")),
        unsloth=UnslothSettings(**_section(raw, "unsloth")),
    )
    _validate_config(config)
    return config


def _validate_config(config: TrainConfig) -> None:
    training = config.training
    if not config.model.name.strip():
        raise ValueError("model.name must not be empty")
    if training.num_generations < 2:
        raise ValueError("training.num_generations must be at least 2 for GRPO")
    if training.max_steps < 1:
        raise ValueError("training.max_steps must be positive")
    if training.per_device_train_batch_size < 1:
        raise ValueError("training.per_device_train_batch_size must be positive")
    if training.gradient_accumulation_steps < 1:
        raise ValueError("training.gradient_accumulation_steps must be positive")
    generation_batch_size = (
        training.per_device_train_batch_size * training.gradient_accumulation_steps
    )
    if generation_batch_size % training.num_generations != 0:
        raise ValueError(
            "per_device_train_batch_size * gradient_accumulation_steps must be divisible "
            f"by num_generations ({generation_batch_size} vs {training.num_generations})"
        )
    if training.fp16 and training.bf16:
        raise ValueError("fp16 and bf16 cannot both be enabled")
    if config.data.limit is not None and config.data.limit < 1:
        raise ValueError("data.limit must be positive or null")
    if config.rewards.format_weight < 0:
        raise ValueError("rewards.format_weight must be non-negative")
    length_penalty = config.rewards.length_penalty
    if length_penalty.target_chars <= 0:
        raise ValueError("rewards.length_penalty.target_chars must be positive")
    if length_penalty.weight < 0:
        raise ValueError("rewards.length_penalty.weight must be non-negative")
    if config.hub.push and not config.lora.enabled:
        raise ValueError("hub.push requires lora.enabled=true (only adapters are pushed)")
    if config.hub.push and not config.hub.repo_id:
        raise ValueError("hub.push requires hub.repo_id")


def build_train_dataset(config: TrainConfig) -> Dataset:
    """Load GSM8K and convert it to TRL prompt/gold records."""
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=config.data.split)
    dataset = dataset.shuffle(seed=config.data.seed)
    if config.data.limit is not None:
        dataset = dataset.select(range(min(config.data.limit, len(dataset))))

    records: list[dict[str, Any]] = []
    for row in dataset:
        gold = extract_gold_answer(row["answer"])
        if gold is None:
            raise ValueError("Could not extract a gold answer from a GSM8K row")
        records.append(
            {
                "prompt": build_messages(row["question"]),
                "gold": gold,
            }
        )
    return Dataset.from_list(records)


def build_reward_funcs(
    config: TrainConfig,
) -> tuple[list[Callable[..., list[float]]], list[float]]:
    """Return reward callables and their weights for GRPOConfig."""
    reward_funcs: list[Callable[..., list[float]]] = [correctness_reward, format_reward]
    weights = [1.0, config.rewards.format_weight]
    if config.rewards.length_penalty.enabled:
        penalty = config.rewards.length_penalty
        reward_funcs.append(
            partial(
                length_penalty_reward,
                target_chars=penalty.target_chars,
                weight=penalty.weight,
            )
        )
        weights.append(1.0)
    return reward_funcs, weights


def _resolve_dtype(name: str, *, fp16: bool, bf16: bool):
    import torch

    if name != "auto":
        try:
            return getattr(torch, name)
        except AttributeError as exc:
            raise ValueError(f"Unknown dtype: {name}") from exc
    if bf16 and torch.cuda.is_available():
        return torch.bfloat16
    if fp16 and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def build_grpo_config(config: TrainConfig):
    import torch
    from trl import GRPOConfig

    training = config.training
    if (training.fp16 or training.bf16) and not torch.cuda.is_available():
        raise RuntimeError(
            "fp16/bf16 training needs CUDA. On this machine set both to false "
            "(the smoke config does this)."
        )
    if training.use_cpu:
        training_fp16 = training_bf16 = False
    else:
        training_fp16 = training.fp16
        training_bf16 = training.bf16
    dtype = _resolve_dtype(
        config.model.dtype,
        fp16=training_fp16,
        bf16=training_bf16,
    )

    _, reward_weights = build_reward_funcs(config)

    return GRPOConfig(
        output_dir=training.output_dir,
        learning_rate=training.learning_rate,
        per_device_train_batch_size=training.per_device_train_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        max_steps=training.max_steps,
        num_generations=training.num_generations,
        max_completion_length=training.max_completion_length,
        temperature=training.temperature,
        beta=training.beta,
        logging_steps=training.logging_steps,
        logging_first_step=True,
        save_steps=training.save_steps,
        save_total_limit=training.save_total_limit,
        gradient_checkpointing=training.gradient_checkpointing,
        optim=training.optim,
        use_cpu=training.use_cpu,
        fp16=training_fp16,
        bf16=training_bf16,
        report_to=training.report_to,
        project=training.project,
        run_name=training.run_name,
        log_completions=training.log_completions,
        seed=training.seed,
        data_seed=config.data.seed,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
        model_init_kwargs={"dtype": dtype},
        reward_weights=reward_weights,
    )


def build_model_and_peft(config: TrainConfig):
    """Return (model argument, processing class, PEFT config) for GRPOTrainer."""
    if not config.unsloth.enabled:
        from peft import LoraConfig

        peft_config = None
        if config.lora.enabled:
            peft_config = LoraConfig(
                r=config.lora.r,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                target_modules=list(config.lora.target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            )
        return config.model.name, None, peft_config

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError(
            "unsloth.enabled=true but unsloth is not installed. Install it on the "
            "Colab/Kaggle Linux GPU machine first."
        ) from exc

    dtype = (
        None
        if config.model.dtype == "auto"
        else _resolve_dtype(
            config.model.dtype,
            fp16=config.training.fp16,
            bf16=config.training.bf16,
        )
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.name,
        max_seq_length=config.unsloth.max_seq_length,
        dtype=dtype,
        load_in_4bit=config.unsloth.load_in_4bit,
    )
    if config.lora.enabled:
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=list(config.lora.target_modules),
            bias="none",
            use_gradient_checkpointing=(
                "unsloth" if config.training.gradient_checkpointing else False
            ),
        )
    else:
        FastLanguageModel.for_training(model)
    return model, tokenizer, None


def _save_artifacts(trainer, config: TrainConfig) -> Path:
    output_dir = Path(config.training.output_dir)
    final_dir = output_dir / ("final_adapter" if config.lora.enabled else "final_model")
    final_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_dir))
    processing_class = getattr(trainer, "processing_class", None)
    if processing_class is not None:
        processing_class.save_pretrained(final_dir)

    (output_dir / "resolved_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (output_dir / "train_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2, default=str) + "\n"
    )
    return final_dir


def _push_adapter(final_dir: Path, config: TrainConfig) -> None:
    from huggingface_hub import HfApi

    assert config.hub.repo_id is not None
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(
        repo_id=config.hub.repo_id,
        private=config.hub.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(final_dir),
        repo_id=config.hub.repo_id,
        commit_message="Upload GRPO LoRA adapter",
        allow_patterns=[
            "README.md",
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "chat_template.jinja",
            "special_tokens_map.json",
            "tokenizer*",
        ],
    )
    print(f"Pushed adapter to https://huggingface.co/{config.hub.repo_id}")


def train(config: TrainConfig) -> Path:
    from trl import GRPOTrainer

    dataset = build_train_dataset(config)
    reward_funcs, reward_weights = build_reward_funcs(config)
    model, processing_class, peft_config = build_model_and_peft(config)
    grpo_config = build_grpo_config(config)

    reward_names = [
        func.func.__name__ if isinstance(func, partial) else func.__name__ for func in reward_funcs
    ]

    print(f"Train examples: {len(dataset)}")
    print(f"Reward functions: {reward_names}")
    print(f"Reward weights: {reward_weights}")
    print(f"Generation batch: {grpo_config.generation_batch_size} completions")
    print(f"Output directory: {config.training.output_dir}")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=processing_class,
    )
    trainer.train()

    final_dir = _save_artifacts(trainer, config)
    print(f"Saved final artifacts to {final_dir}")

    if config.hub.push:
        _push_adapter(final_dir, config)
    return final_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config path (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
