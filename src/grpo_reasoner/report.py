"""Build baseline-vs-GRPO reports and reward-curve plots."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from grpo_reasoner.rewards import has_required_format

_SETTING_PATHS = (
    "model",
    "split",
    "limit",
    "seed",
    "prompt_version",
    "prompt",
    "generation.max_new_tokens",
    "generation.batch_size",
    "generation.temperature",
)


def _nested(data: Mapping[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def compare_settings(
    baseline: Mapping[str, Any], trained: Mapping[str, Any]
) -> list[tuple[str, Any, Any]]:
    """Return setting mismatches that would make the comparison unfair."""
    mismatches: list[tuple[str, Any, Any]] = []
    for path in _SETTING_PATHS:
        baseline_value = _nested(baseline, path)
        trained_value = _nested(trained, path)
        if baseline_value != trained_value:
            mismatches.append((path, baseline_value, trained_value))
    return mismatches


def format_adherence(result: Mapping[str, Any]) -> float:
    examples = result.get("examples", [])
    if not examples:
        return 0.0
    formatted = sum(has_required_format(example.get("completion") or "") for example in examples)
    return formatted / len(examples)


def _accuracy(result: Mapping[str, Any]) -> float:
    return float(result.get("summary", {}).get("accuracy", 0.0))


def _correct(result: Mapping[str, Any]) -> int:
    return int(result.get("summary", {}).get("correct", 0))


def _total(result: Mapping[str, Any]) -> int:
    return int(result.get("summary", {}).get("total", 0))


def _fmt_pct(value: float) -> str:
    return f"{value:.1%}"


def _fmt_signed_pct(value: float) -> str:
    return f"{value:+.1f} pp"


def _one_line(text: str, max_chars: int = 500) -> str:
    compact = " ".join(text.split())
    if len(compact) > max_chars:
        return compact[:max_chars].rstrip() + "..."
    return compact


def _code_block(text: str, max_chars: int = 900) -> str:
    snippet = text.strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "\n[...]"
    snippet = snippet.replace("```", "'''")
    return f"```text\n{snippet}\n```"


def _select_samples(
    baseline: Mapping[str, Any],
    trained: Mapping[str, Any],
    count: int,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    trained_by_index = {example["source_index"]: example for example in trained.get("examples", [])}
    pairs = [
        (example, trained_by_index[example["source_index"]])
        for example in baseline.get("examples", [])
        if example["source_index"] in trained_by_index
    ]
    improved = [pair for pair in pairs if pair[1].get("correct") and not pair[0].get("correct")]
    selected = improved[:count]
    if len(selected) < count:
        selected_ids = {baseline_example["source_index"] for baseline_example, _ in selected}
        for pair in pairs:
            if len(selected) >= count:
                break
            if pair[0]["source_index"] not in selected_ids:
                selected.append(pair)
    return selected


def build_comparison_markdown(
    baseline: Mapping[str, Any],
    trained: Mapping[str, Any],
    *,
    samples: int = 3,
) -> str:
    """Create a markdown table plus side-by-side completion samples."""
    mismatches = compare_settings(baseline, trained)
    if mismatches:
        details = ", ".join(path for path, _, _ in mismatches)
        raise ValueError(f"Evaluation settings differ: {details}")

    baseline_accuracy = _accuracy(baseline)
    trained_accuracy = _accuracy(trained)
    baseline_format = format_adherence(baseline)
    trained_format = format_adherence(trained)
    delta_correct = _correct(trained) - _correct(baseline)
    delta_pp = (trained_accuracy - baseline_accuracy) * 100
    delta_format_pp = (trained_format - baseline_format) * 100

    lines = [
        "# Baseline vs GRPO: GSM8K results",
        "",
        f"Generated {datetime.now(UTC).isoformat()}.",
        "",
        "## Evaluation settings",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| Base model | `{baseline.get('model')}` |",
        f"| Adapter | `{trained.get('adapter') or 'none'}` |",
        f"| Split | `{baseline.get('split')}` |",
        f"| Examples | {baseline.get('limit')} |",
        f"| Seed | {baseline.get('seed')} |",
        f"| Prompt version | `{baseline.get('prompt_version')}` |",
        f"| Max new tokens | {_nested(baseline, 'generation.max_new_tokens')} |",
        f"| Temperature | {_nested(baseline, 'generation.temperature')} |",
        "",
        "Settings match between both runs: **yes**.",
        "",
        "## Accuracy",
        "",
        "| Stage | Correct | Accuracy | Format adherence |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Baseline (untrained) | {_correct(baseline)}/{_total(baseline)} | "
            f"{_fmt_pct(baseline_accuracy)} | {_fmt_pct(baseline_format)} |"
        ),
        (
            f"| GRPO adapter | {_correct(trained)}/{_total(trained)} | "
            f"{_fmt_pct(trained_accuracy)} | {_fmt_pct(trained_format)} |"
        ),
        (
            f"| Change | {delta_correct:+d} | {_fmt_signed_pct(delta_pp)} | "
            f"{_fmt_signed_pct(delta_format_pp)} |"
        ),
        "",
        "Format adherence is the fraction of completions matching the exact "
        "`<think>...</think><answer>...</answer>` template.",
        "",
        "## Side-by-side samples",
        "",
    ]

    sample_pairs = _select_samples(baseline, trained, samples)
    if not sample_pairs:
        lines.append("No matching examples found.")
    for position, (baseline_example, trained_example) in enumerate(sample_pairs, start=1):
        lines.extend(
            [
                f"### Example {position} (test index {baseline_example['source_index']})",
                "",
                f"**Question:** {_one_line(str(baseline_example.get('question', '')))}",
                "",
                f"**Gold answer:** `{baseline_example.get('gold')}`",
                "",
                (
                    f"**Baseline** — prediction `{baseline_example.get('prediction')}` "
                    f"({'correct' if baseline_example.get('correct') else 'wrong'})"
                ),
                "",
                _code_block(str(baseline_example.get("completion", ""))),
                "",
                (
                    f"**GRPO** — prediction `{trained_example.get('prediction')}` "
                    f"({'correct' if trained_example.get('correct') else 'wrong'})"
                ),
                "",
                _code_block(str(trained_example.get("completion", ""))),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def plot_reward_curves(
    history: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    title: str = "GRPO training",
) -> Path:
    """Save reward, completion-length, KL, and stability curves as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = [entry for entry in history if "step" in entry]
    entries.sort(key=lambda entry: entry["step"])
    if not entries:
        raise ValueError("Training history has no entries with a step")

    steps = [entry["step"] for entry in entries]

    def series(*keys: str) -> tuple[list[int], list[float]]:
        xs: list[int] = []
        ys: list[float] = []
        for step, entry in zip(steps, entries, strict=True):
            value = next((entry[key] for key in keys if key in entry), None)
            if value is not None:
                xs.append(step)
                ys.append(float(value))
        return xs, ys

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    fig.suptitle(title)

    ax = axes[0, 0]
    plotted = False
    for keys, label in (
        (("reward",), "Total reward"),
        (("rewards/correctness_reward/mean",), "Correctness"),
        (("rewards/format_reward/mean",), "Format"),
    ):
        xs, ys = series(*keys)
        if xs:
            ax.plot(xs, ys, marker="o", label=label)
            plotted = True
    ax.set_title("Reward")
    ax.set_ylabel("reward")
    if plotted:
        ax.legend()

    ax = axes[0, 1]
    length_plotted = False
    for keys, label in (
        (("completions/mean_length",), "Mean"),
        (("completions/min_length",), "Min"),
        (("completions/max_length",), "Max"),
    ):
        xs, ys = series(*keys)
        if xs:
            ax.plot(xs, ys, marker="o", label=label)
            length_plotted = True
    ax.set_title("Completion length")
    ax.set_ylabel("tokens")
    if length_plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Length not logged", ha="center", va="center")

    ax = axes[1, 0]
    xs, ys = series("kl")
    if xs:
        ax.plot(xs, ys, marker="o", color="tab:red")
    else:
        ax.text(0.5, 0.5, "KL not logged", ha="center", va="center")
    ax.set_title("KL to reference")
    ax.set_ylabel("KL")
    ax.set_xlabel("step")

    ax = axes[1, 1]
    xs, ys = series("reward_std")
    if xs:
        ax.plot(xs, ys, marker="o", color="tab:purple")
    else:
        ax.text(0.5, 0.5, "Reward std not logged", ha="center", va="center")
    ax.set_title("Reward std (within group)")
    ax.set_ylabel("std")
    ax.set_xlabel("step")

    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("results/baseline.json"))
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/comparison.md"))
    parser.add_argument("--plot", type=Path, default=Path("results/reward_curves.png"))
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args(argv)

    baseline = _load_json(args.baseline)
    trained = _load_json(args.trained)
    mismatches = compare_settings(baseline, trained)
    if mismatches:
        details = ", ".join(
            f"{path} ({baseline_value!r} vs {trained_value!r})"
            for path, baseline_value, trained_value in mismatches
        )
        raise SystemExit(
            f"Evaluation settings differ: {details}\n"
            "Re-run the trained evaluation with the baseline's flags."
        )

    markdown = build_comparison_markdown(baseline, trained, samples=args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)
    print(f"Wrote comparison to {args.output}")

    if args.history is not None:
        history = _load_json(args.history)
        plot_path = plot_reward_curves(history, args.plot)
        print(f"Wrote reward curves to {plot_path}")

    baseline_summary = baseline["summary"]
    trained_summary = trained["summary"]
    print(f"Accuracy: {baseline_summary['accuracy']:.1%} -> {trained_summary['accuracy']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
