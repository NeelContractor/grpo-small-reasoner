from pathlib import Path

import pytest

from grpo_reasoner.train import build_reward_funcs, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_configs_load():
    smoke = load_config(ROOT / "configs" / "train_smoke.yaml")
    assert smoke.model.name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert smoke.training.num_generations == 2
    assert smoke.training.max_steps == 2
    assert smoke.training.per_device_train_batch_size == 2
    assert smoke.training.fp16 is False
    assert smoke.training.report_to == "none"

    full = load_config(ROOT / "configs" / "train_full.yaml")
    assert full.training.num_generations == 8
    assert full.training.max_steps == 200
    assert full.lora.enabled is True
    assert full.training.report_to == "wandb"


def test_reward_weights_without_length_penalty():
    config = load_config(ROOT / "configs" / "train_smoke.yaml")
    funcs, weights = build_reward_funcs(config)
    assert [func.__name__ for func in funcs] == ["correctness_reward", "format_reward"]
    assert weights == [1.0, 0.5]


def test_length_penalty_adds_third_reward(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: test/model
training:
  num_generations: 2
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 1
rewards:
  format_weight: 0.25
  length_penalty:
    enabled: true
    target_chars: 1000
    weight: 0.2
"""
    )
    config = load_config(config_path)
    funcs, weights = build_reward_funcs(config)
    assert len(funcs) == 3
    assert weights == [1.0, 0.25, 1.0]


def test_generation_batch_must_be_divisible_by_num_generations(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: test/model
training:
  num_generations: 8
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 1
"""
    )
    with pytest.raises(ValueError, match="divisible"):
        load_config(config_path)


def test_hub_push_requires_repo_id(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: test/model
training:
  num_generations: 2
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 1
hub:
  push: true
"""
    )
    with pytest.raises(ValueError, match="repo_id"):
        load_config(config_path)
