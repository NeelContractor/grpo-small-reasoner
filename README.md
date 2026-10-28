# grpo-small-reasoner

Improving a small language model's math reasoning with GRPO and verifiable rewards.
Work in progress; this README will be expanded after training.

## Status

- [x] Phase 1: baseline evaluation
- [x] Phase 2: reward functions
- [x] Phase 3: training
- [ ] Phase 4: results (tooling ready; needs a full T4 run)
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

## Rewards

Each reward is a pure function in `src/grpo_reasoner/rewards.py`:

- `correctness_reward`: 1.0 for exact numeric match with the gold answer.
- `format_reward`: 1.0 only for exactly one non-empty `<think>` block followed by
  one non-empty `<answer>` block, with no extra text.
- `length_penalty_reward`: optional, opt-in penalty capped at `weight` for
  completions longer than `target_chars`.

## Training

Two YAML configs are provided:

| Config | Purpose |
| --- | --- |
| `configs/train_smoke.yaml` | Laptop smoke test: 2 steps, 2 generations, 128 tokens |
| `configs/train_full.yaml` | Colab/Kaggle T4: 200 steps, 8 generations per prompt, LoRA |

```bash
# Local pipeline check on Apple MPS/CPU (a few minutes)
make smoke

# Full run (CUDA GPU; fine-tunes a LoRA adapter)
make train
```

In TRL, `per_device_train_batch_size` counts completions. With
`num_generations: 8` and batch size 8, each optimizer step uses one prompt and
its 8 completions. To cut generation memory, use batch size 4 with gradient
accumulation 2.

`train.py` logs reward, reward std, completion length, and KL to Weights & Biases
when `report_to: wandb`. The full config saves checkpoints, writes
`resolved_config.json` and `train_history.json`, and saves a final LoRA adapter to
`<output_dir>/final_adapter`. Set `hub.push: true` and `hub.repo_id` to upload the
adapter; provide `HF_TOKEN`.

The Colab/Kaggle wrapper is `notebooks/train_colab.ipynb`. It clones the repo,
installs dependencies, reads `WANDB_API_KEY`/`HF_TOKEN` from Colab secrets, runs
`train.py`, then runs the Phase 4 evaluation and report in the same runtime.

## Results (Phase 4)

After a full training run, evaluate the adapter with the exact baseline settings
and build the report:

```bash
make trained-eval
make report
```

Equivalent explicit commands:

```bash
uv run grpo-eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter outputs/grpo-qwen2.5-0.5b/final_adapter \
  --split test --limit 50 --seed 42 --max-new-tokens 512 \
  --batch-size 1 --temperature 0.0 \
  --output results/trained.json

uv run grpo-report \
  --baseline results/baseline.json \
  --trained results/trained.json \
  --history outputs/grpo-qwen2.5-0.5b/train_history.json \
  --output results/comparison.md \
  --plot results/reward_curves.png
```

`grpo-report` refuses to compare runs whose seed, subset size, prompt, decoding
settings, or base model differ. It writes a before/after accuracy and format table,
side-by-side sample completions, and reward/KL/completion-length plots. The real
numbers are pending a full Colab/Kaggle T4 run; the notebook creates all artifacts
in one pass.

## Troubleshooting

On some Colab images, PEFT fails at LoRA injection with:

```text
ImportError: Found an incompatible version of torchao. Found version 0.10.0,
but only versions above 0.16.0 are supported
```

Torchao is not used by this project. Remove Colab's old copy and rerun:

```bash
pip uninstall -y torchao
```

```bash
make test
make lint
make baseline
```
