# grpo-small-reasoner

Improving a small language model's math reasoning with GRPO and verifiable rewards.

## Status

- [x] Phase 1: baseline evaluation
- [x] Phase 2: reward functions
- [x] Phase 3: training
- [x] Phase 4: results
- [ ] Phase 5: final documentation

## Results at a glance

A 200-step GRPO run on a Colab T4 improved strict GSM8K accuracy on the fixed
50-example test subset from **34.0% to 46.0%** (+12.0 pp). The result is directional,
not statistically significant (McNemar exact p = 0.109), and the adapter did **not**
learn the requested `<think>`/`<answer>` format. See [Results](#results-phase-4) for
the full caveats.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+ (tested with 3.12 locally
and 3.13 on Colab).

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

Full run: `Qwen/Qwen2.5-0.5B-Instruct` + LoRA, 200 GRPO steps on one Colab T4,
1h45m. The adapter is published at
[`NeelContractor/grpo-qwen2.5-0.5b-gsm8k`](https://huggingface.co/NeelContractor/grpo-qwen2.5-0.5b-gsm8k).

Evaluation is the fixed 50-example GSM8K test subset (`seed=42`, greedy decoding,
`max_new_tokens=512`) used for the Phase 1 baseline.

| Stage | Correct | Accuracy | Format adherence |
| --- | ---: | ---: | ---: |
| Baseline (untrained) | 17/50 | 34.0% | 0.0% |
| GRPO adapter | 23/50 | 46.0% | 0.0% |
| Change | +6 | +12.0 pp | +0.0 pp |

**Interpretation.** The adapter improved the strict final-answer score, but the
format reward stayed at zero for every training step, so the model never learned
the requested `<think>...</think><answer>...</answer>` template. The gain comes
from the correctness reward on plain reasoning.

**The result is fragile.** Strict scoring uses the extracted final answer (the last
number when the model emits no answer tag); a lenient upper-bound check (gold value
mentioned anywhere) gives 25/50 (50.0%) for the baseline and 27/50 (54.0%) for the
adapter, a +4.0 pp gain. The headline +12.0 pp therefore shrinks to +4.0 pp when
intermediate mentions are allowed, and the strict metric is sensitive to
final-answer formatting. The paired outcomes are 8 improved, 2 regressed, 15 both
correct, and 25 both wrong; McNemar exact two-sided p = 0.109. With 50 examples,
this is not statistically significant at 0.05. Treat it as directional evidence,
not a benchmark claim.

Training curves: correctness reward averaged 0.40 over the first 10 steps and 0.46
over the last 20; KL rose from 0.0007 to 0.0063; mean completion length stayed near
239 tokens.

**Artifacts**

- [`results/comparison.md`](results/comparison.md): full table, sensitivity check,
  paired outcomes, and side-by-side samples.
- [`results/reward_curves.png`](results/reward_curves.png): reward, length, KL, and
  within-group reward std.
- [`results/baseline.json`](results/baseline.json) and
  [`results/trained.json`](results/trained.json): raw generations and predictions.
- [`results/train_history.json`](results/train_history.json) and
  [`results/train_config_resolved.json`](results/train_config_resolved.json):
  run metadata needed to regenerate the report.

### Reproduce the results

Evaluate the published adapter with the exact baseline settings:

```bash
uv run grpo-eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter NeelContractor/grpo-qwen2.5-0.5b-gsm8k \
  --split test --limit 50 --seed 42 --max-new-tokens 512 \
  --batch-size 1 --temperature 0.0 \
  --output results/trained.json
```

Regenerate the report and plot from the committed artifacts (no GPU needed):

```bash
make report
```

`grpo-report` refuses to compare runs whose seed, subset size, prompt, decoding
settings, or base model differ. It writes the accuracy/sensitivity/paired tables,
side-by-side samples, and reward/KL/completion-length plots. The Colab notebook
runs training, evaluation, and report generation in one pass.

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
