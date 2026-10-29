import json
from pathlib import Path

import pytest

from grpo_reasoner.report import (
    build_comparison_markdown,
    compare_settings,
    format_adherence,
    lenient_correct_count,
    mcnemar_exact_p,
    paired_outcomes,
    plot_reward_curves,
)


def _result(*, adapter=None, seed=42, completions, corrects):
    examples = []
    for index, (completion, correct) in enumerate(zip(completions, corrects, strict=True)):
        examples.append(
            {
                "source_index": index,
                "question": f"Question {index}?",
                "reference": f"Work\n#### {index}",
                "gold": str(index),
                "completion": completion,
                "prediction": str(index) if correct else "999",
                "correct": correct,
            }
        )
    total = len(examples)
    correct_count = sum(corrects)
    return {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter": adapter,
        "split": "test",
        "seed": seed,
        "limit": total,
        "prompt_version": "v1",
        "prompt": "fixed prompt",
        "generation": {
            "max_new_tokens": 512,
            "batch_size": 1,
            "temperature": 0.0,
        },
        "summary": {
            "total": total,
            "correct": correct_count,
            "accuracy": correct_count / total,
        },
        "examples": examples,
    }


def test_format_adherence():
    result = _result(
        completions=[
            "<think>x</think><answer>0</answer>",
            "plain reasoning with no tags",
        ],
        corrects=[True, False],
    )
    assert format_adherence(result) == 0.5


def test_paired_outcomes():
    baseline = _result(completions=["a", "b", "c", "d"], corrects=[False, True, False, True])
    trained = _result(
        completions=["a", "b", "c", "d"],
        corrects=[True, False, False, True],
    )
    assert paired_outcomes(baseline, trained) == {
        "improved": 1,
        "regressed": 1,
        "both_correct": 1,
        "both_wrong": 1,
    }


def test_lenient_correct_count():
    result = _result(
        completions=["wrong but mentions 0", "wrong with no gold", "too long: 00"],
        corrects=[False, False, False],
    )
    assert lenient_correct_count(result) == 1


def test_mcnemar_exact_p():
    assert mcnemar_exact_p(8, 2) == pytest.approx(0.109375)
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(5, 0) == pytest.approx(0.0625)


def test_compare_settings_detects_mismatch():
    baseline = _result(completions=["a", "b"], corrects=[False, False])
    trained = _result(completions=["a", "b"], corrects=[False, False], seed=7)
    assert compare_settings(baseline, trained) == [("seed", 42, 7)]


def test_compare_settings_accepts_adapter_difference():
    baseline = _result(completions=["a", "b"], corrects=[False, False])
    trained = _result(
        adapter="outputs/grpo/final_adapter",
        completions=["a", "b"],
        corrects=[False, False],
    )
    assert compare_settings(baseline, trained) == []


def test_comparison_skips_ambiguous_baseline_samples():
    baseline = _result(
        completions=["wrong but ends with 0 in 1 day", "wrong with no number"],
        corrects=[False, False],
    )
    trained = _result(
        adapter="outputs/grpo/final_adapter",
        completions=[
            "<think>0</think><answer>0</answer>",
            "<think>1</think><answer>1</answer>",
        ],
        corrects=[True, True],
    )
    markdown = build_comparison_markdown(baseline, trained, samples=1)
    assert "### Example 1 (test index 1)" in markdown
    assert "test index 0" not in markdown


def test_comparison_markdown_contains_table_and_improved_sample():
    baseline = _result(
        completions=["no tags, wrong", "no tags, wrong"],
        corrects=[False, False],
    )
    trained = _result(
        adapter="outputs/grpo/final_adapter",
        completions=[
            "<think>0</think><answer>0</answer>",
            "<think>1</think><answer>1</answer>",
        ],
        corrects=[True, True],
    )
    markdown = build_comparison_markdown(baseline, trained, samples=1)
    assert "| Baseline (untrained) | 0/2 | 0.0% | 0.0% |" in markdown
    assert "| GRPO adapter | 2/2 | 100.0% | 100.0% |" in markdown
    assert "| Change | +2 | +100.0 pp | +100.0 pp |" in markdown
    assert "## Extraction sensitivity" in markdown
    assert "## Paired outcomes" in markdown
    assert "| Improved | 2 |" in markdown
    assert "| Regressed | 0 |" in markdown
    assert "McNemar exact two-sided p = 0.500" in markdown
    assert "### Example 1 (test index 0)" in markdown
    assert "**Baseline** — prediction `999` (wrong)" in markdown
    assert "**GRPO** — prediction `0` (correct)" in markdown
    assert "<think>0</think>" in markdown


def test_plot_reward_curves(tmp_path):
    history = [
        {
            "step": 1,
            "reward": 0.1,
            "reward_std": 0.2,
            "completions/mean_length": 100,
            "completions/min_length": 80,
            "completions/max_length": 128,
            "kl": 0.0,
            "rewards/correctness_reward/mean": 0.1,
            "rewards/format_reward/mean": 0.0,
        },
        {
            "step": 2,
            "reward": 0.3,
            "reward_std": 0.4,
            "completions/mean_length": 120,
            "completions/min_length": 100,
            "completions/max_length": 128,
            "kl": 0.01,
            "rewards/correctness_reward/mean": 0.2,
            "rewards/format_reward/mean": 0.2,
        },
    ]
    output = tmp_path / "curves.png"
    assert plot_reward_curves(history, output) == output
    assert output.read_bytes().startswith(b"\x89PNG")


def test_plot_requires_history(tmp_path):
    with pytest.raises(ValueError, match="no entries"):
        plot_reward_curves([{"loss": 1.0}], tmp_path / "curves.png")


def test_load_fixture_roundtrip(tmp_path):
    baseline = _result(completions=["a", "b"], corrects=[False, False])
    path = tmp_path / "result.json"
    path.write_text(json.dumps(baseline))
    assert json.loads(path.read_text())["summary"]["accuracy"] == 0.0


def test_baseline_fixture_shape():
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "results" / "baseline.json").read_text())
    assert baseline["limit"] == 50
    assert baseline["summary"]["total"] == 50
