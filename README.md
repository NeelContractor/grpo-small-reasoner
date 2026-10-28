# grpo-small-reasoner

Improving a small language model's math reasoning with GRPO and verifiable rewards.
Work in progress; this README will be expanded after training.

## Status

- [x] Phase 1: baseline evaluation
- [ ] Phase 2: reward functions
- [ ] Phase 3: training
- [ ] Phase 4: results
- [ ] Phase 5: final documentation

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 or 3.12.

```bash
uv sync --extra dev
uv run pre-commit install
```

## Evaluation

Evaluate any Hugging Face model or local checkpoint on a fixed, deterministic GSM8K
test subset:

```bash
uv run grpo-eval --model Qwen/Qwen2.5-0.5B-Instruct --limit 50 --output results/baseline.json
```

If the machine has MPS or CUDA, the script selects it automatically; otherwise it
runs on CPU. Use `--device` and `--dtype` to override. Add `--adapter PATH` for a
LoRA adapter.

The fixed prompt asks for `<think>...</think><answer>...</answer>`. Accuracy is
exact numeric match against the GSM8K reference answer.

```bash
make test
make lint
make baseline
```
