# Phase 1 baseline

Untrained `Qwen/Qwen2.5-0.5B-Instruct` on a fixed 50-example GSM8K test subset.

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Dataset | `openai/gsm8k`, `test` split |
| Subset | 50 examples, shuffled with `seed=42` |
| Prompt | fixed `<think>...</think><answer>...</answer>` system prompt |
| Decoding | greedy, `max_new_tokens=512` |
| Device | Apple MPS, float16 |
| Correct | 17 / 50 |
| **Accuracy** | **34.0%** |

Command:

```bash
uv run grpo-eval --output results/baseline.json
```

Notes:

- This is the number the trained model must beat under identical settings.
- None of the 50 untrained completions used the requested `<think>` or `<answer>`
  tags, so this baseline measures raw instruction-following plus GSM8K accuracy.
- Full per-example predictions are in `results/baseline.json`.
