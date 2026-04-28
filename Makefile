SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

# Default conservative model for an 8 GB M1. Override with:
# make checkpoint1 MODEL=mlx-community/Meta-Llama-3-8B-Instruct-4bit
MODEL ?= mlx-community/Llama-3.2-1B-Instruct-4bit
QUANTIZATION ?= 4bit
RUNS ?= 3
MAX_TOKENS ?= 128
PROMPT_TOKENS ?= 384,1536,3072
BASELINE_DIR ?= logs/baseline
BASELINE_LOG ?= $(BASELINE_DIR)/baseline_$(shell date +%Y%m%d_%H%M%S).jsonl
OBS_MODEL ?= models/Llama-3.2-1B-4bit
CACHE_SAMPLE_DIR ?= data/cache_samples
FIGURES_DIR ?= reports/figures
PCA_MAX_SAMPLES ?= 50000

.PHONY: help checkpoint1 checkpoint1-prepare checkpoint1-deps checkpoint1-dirs checkpoint1-preflight checkpoint1-baseline checkpoint1-report checkpoint1-check checkpoint2 checkpoint2-prepare checkpoint2-deps checkpoint2-dirs checkpoint2-extract checkpoint2-analyze checkpoint2-check telemetry-check

help:
	@echo "Lumina checkpoint automation"
	@echo ""
	@echo "Targets:"
	@echo "  make checkpoint1          Run the full Checkpoint 1 baseline flow"
	@echo "  make checkpoint1-check    Validate scripts and configuration without downloading a model"
	@echo "  make checkpoint1-deps     Install Python dependencies in .venv"
	@echo "  make checkpoint1-report   Rebuild reports from logs/baseline/*.jsonl"
	@echo "  make telemetry-check      Print current vm_stat/memory_pressure telemetry snapshot"
	@echo "  make checkpoint2          Extract KV cache and run offline PCA observatory"
	@echo "  make checkpoint2-check    Validate Checkpoint 2 scripts without loading a model"
	@echo ""
	@echo "Config:"
	@echo "  MODEL=$(MODEL)"
	@echo "  OBS_MODEL=$(OBS_MODEL)"
	@echo "  RUNS=$(RUNS)"
	@echo "  MAX_TOKENS=$(MAX_TOKENS)"
	@echo "  PROMPT_TOKENS=$(PROMPT_TOKENS)"

checkpoint1: checkpoint1-prepare checkpoint1-preflight checkpoint1-baseline checkpoint1-report
	@echo "Checkpoint 1 complete. Report: reports/baseline_report.md"

checkpoint1-prepare: checkpoint1-dirs checkpoint1-deps

checkpoint1-dirs:
	@mkdir -p $(BASELINE_DIR) reports scripts configs data checkpoints notebooks src

.venv/bin/python:
	python3 -m venv .venv

checkpoint1-deps: .venv/bin/python
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install --upgrade mlx mlx-lm

checkpoint1-preflight: checkpoint1-dirs
	@{ \
		echo "# Checkpoint 1 preflight"; \
		date '+timestamp=%Y-%m-%dT%H:%M:%S%z'; \
		echo "model=$(MODEL)"; \
		echo "runs=$(RUNS)"; \
		echo "prompt_tokens=$(PROMPT_TOKENS)"; \
		echo ""; \
		df -h; \
		echo ""; \
		sysctl vm.swapusage; \
		echo ""; \
		top -l 1 | grep "PhysMem" || true; \
		echo ""; \
		memory_pressure 2>/dev/null | tail -n 20 || true; \
	} > $(BASELINE_DIR)/preflight_$(shell date +%Y%m%d_%H%M%S).txt

checkpoint1-baseline:
	$(PYTHON) scripts/run_baseline.py \
		--model "$(MODEL)" \
		--quantization "$(QUANTIZATION)" \
		--runs "$(RUNS)" \
		--max-tokens "$(MAX_TOKENS)" \
		--prompt-tokens "$(PROMPT_TOKENS)" \
		--output "$(BASELINE_LOG)"

checkpoint1-report:
	$(PYTHON) scripts/generate_baseline_report.py \
		--logs "$(BASELINE_DIR)" \
		--report reports/baseline_report.md \
		--summary-csv reports/baseline_summary.csv \
		--memory-svg reports/baseline_memory_vs_prompt.svg \
		--throughput-svg reports/baseline_tokens_per_sec_vs_prompt.svg

checkpoint1-check: .venv/bin/python checkpoint1-dirs
	$(PYTHON) -m py_compile src/telemetry.py scripts/run_baseline.py scripts/generate_baseline_report.py
	$(PYTHON) scripts/run_baseline.py \
		--dry-run \
		--model "$(MODEL)" \
		--quantization "$(QUANTIZATION)" \
		--runs "$(RUNS)" \
		--max-tokens "$(MAX_TOKENS)" \
		--prompt-tokens "$(PROMPT_TOKENS)" \
		--output "$(BASELINE_LOG)"

telemetry-check: .venv/bin/python
	$(PYTHON) -c 'import json, os; from dataclasses import asdict; from src.telemetry import snapshot; print(json.dumps(asdict(snapshot(os.getpid())), indent=2, sort_keys=True))'

checkpoint2: checkpoint2-prepare checkpoint2-extract checkpoint2-analyze
	@echo "Checkpoint 2 observatory complete. Report: reports/memory_observatory.md"

checkpoint2-prepare: checkpoint2-dirs checkpoint2-deps

checkpoint2-dirs:
	@mkdir -p $(CACHE_SAMPLE_DIR) $(FIGURES_DIR) reports data

checkpoint2-deps: .venv/bin/python
	$(PIP) install --upgrade numpy

checkpoint2-extract:
	$(PYTHON) scripts/extract_cache.py \
		--model "$(OBS_MODEL)" \
		--prompt-tokens "$(PROMPT_TOKENS)" \
		--output-root "$(CACHE_SAMPLE_DIR)"

checkpoint2-analyze:
	$(PYTHON) scripts/analyze_cache_pca.py \
		--samples "$(CACHE_SAMPLE_DIR)" \
		--figures "$(FIGURES_DIR)" \
		--report reports/memory_observatory.md \
		--max-samples "$(PCA_MAX_SAMPLES)"

checkpoint2-check: .venv/bin/python checkpoint2-dirs
	$(PYTHON) -m py_compile src/telemetry.py scripts/extract_cache.py scripts/analyze_cache_pca.py
	$(PYTHON) scripts/extract_cache.py \
		--dry-run \
		--model "$(OBS_MODEL)" \
		--prompt-tokens "$(PROMPT_TOKENS)" \
		--output-root "$(CACHE_SAMPLE_DIR)"
