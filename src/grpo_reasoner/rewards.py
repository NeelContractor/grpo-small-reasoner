"""Pure, verifiable reward functions for GRPO."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from grpo_reasoner.extract import answers_match, extract_answer

_FORMAT_RE = re.compile(
    r"\A\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*\Z",
    re.DOTALL,
)


def completion_text(completion: Any) -> str:
    """Normalize TRL string completions or chat-message completions to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        content = completion.get("content", "")
        return content if isinstance(content, str) else str(content)
    if isinstance(completion, Sequence):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, Mapping):
                content = item.get("content", "")
                parts.append(content if isinstance(content, str) else str(content))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(completion)


def has_required_format(text: str) -> bool:
    """Return True when text is exactly one non-empty think block and answer block."""
    for tag in ("<think>", "</think>", "<answer>", "</answer>"):
        if text.count(tag) != 1:
            return False
    match = _FORMAT_RE.match(text)
    if match is None:
        return False
    return bool(match.group("think").strip()) and bool(match.group("answer").strip())


def correctness_reward(completions: Sequence[Any], gold: Sequence[str], **_: Any) -> list[float]:
    """1.0 when the extracted final answer equals the gold answer numerics."""
    if len(gold) != len(completions):
        raise ValueError("gold and completions must have the same length")
    rewards: list[float] = []
    for completion, expected in zip(completions, gold, strict=True):
        prediction = extract_answer(completion_text(completion))
        rewards.append(1.0 if answers_match(prediction, expected) else 0.0)
    return rewards


def format_reward(completions: Sequence[Any], **_: Any) -> list[float]:
    """1.0 when the completion follows <think>...</think><answer>...</answer>."""
    return [
        1.0 if has_required_format(completion_text(completion)) else 0.0
        for completion in completions
    ]


def length_penalty_reward(
    completions: Sequence[Any],
    *,
    target_chars: int = 2000,
    weight: float = 0.1,
    **_: Any,
) -> list[float]:
    """Small penalty for completions longer than ``target_chars``.

    The penalty grows linearly from 0 to ``weight`` as length goes from
    ``target_chars`` to ``2 * target_chars``, then stays capped at ``weight``.
    Short completions are never rewarded or penalized.
    """
    if target_chars <= 0:
        raise ValueError("target_chars must be positive")
    if weight < 0:
        raise ValueError("weight must be non-negative")

    rewards: list[float] = []
    for completion in completions:
        length = len(completion_text(completion))
        excess = max(0, length - target_chars)
        rewards.append(-weight * min(1.0, excess / target_chars))
    return rewards
