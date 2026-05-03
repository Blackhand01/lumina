PYTHON ?= python

.PHONY: check test probe-mlx

check:
	PYTHONPATH=src $(PYTHON) -m compileall src tests

test:
	$(PYTHON) -m pytest

probe-mlx:
	PYTHONPATH=src $(PYTHON) -m lumina.backend.mlx_probe
