.PHONY: install lint format test check baseline eval smoke train trained-eval report

MODEL ?= Qwen/Qwen2.5-0.5B-Instruct
TRAIN_OUT ?= outputs/grpo-qwen2.5-0.5b

install:
	uv sync --extra dev

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

check: lint test

# Untrained baseline on the fixed 50-example GSM8K test subset.
baseline:
	uv run grpo-eval --output results/baseline.json

# Usage: make eval MODEL=Qwen/Qwen2.5-0.5B-Instruct ADAPTER=outputs/grpo/final
eval:
	uv run grpo-eval --model "$(MODEL)" $(if $(ADAPTER),--adapter "$(ADAPTER)") \
		--output "$(if $(OUTPUT),$(OUTPUT),results/eval.json)"

# Tiny local smoke run (Apple MPS / CPU).
smoke:
	uv run python train.py --config configs/train_smoke.yaml

# Full run; intended for Colab/Kaggle or a CUDA machine.
train:
	uv run python train.py --config configs/train_full.yaml

# Phase 4: identical eval settings to the Phase 1 baseline.
trained-eval:
	uv run grpo-eval --model "$(MODEL)" \
		--adapter "$(or $(ADAPTER),$(TRAIN_OUT)/final_adapter)" \
		--split test --limit 50 --seed 42 --max-new-tokens 512 \
		--batch-size 1 --temperature 0.0 \
		--output "$(or $(OUTPUT),results/trained.json)"

# Build results/comparison.md and results/reward_curves.png.
report:
	uv run grpo-report --baseline results/baseline.json \
		--trained "$(or $(TRAINED),results/trained.json)" \
		--history "$(or $(HISTORY),$(TRAIN_OUT)/train_history.json)" \
		--output results/comparison.md --plot results/reward_curves.png
