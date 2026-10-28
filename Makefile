.PHONY: install lint format test check baseline eval smoke train

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
