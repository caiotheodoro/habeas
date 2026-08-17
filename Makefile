PY ?= uv run

.PHONY: sync validate study bench

sync:
	cd forge && uv sync --no-editable --extra dev && chflags -R nohidden .venv

validate:
	cd forge && uv run python -m pytest -q

study:
	cd forge && $(PY) python -m i9forge_forge.cli pilot --seed 7 --n 400

bench:
	cd model && $(PY) python -m i9forge_model.benchmark_eval --tasks-file data/benchmark.jsonl
