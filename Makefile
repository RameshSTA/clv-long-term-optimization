# =============================================================================
# CLV Long-Term Optimization — Development Makefile
# =============================================================================
# Usage:
#   make help          Show this help message
#   make install       Create venv and install dependencies
#   make pipeline      Run the full end-to-end pipeline (default budget £5,000)
#   make test          Run unit test suite
#   make coverage      Run tests with coverage report
#   make lint          Run ruff linter
#   make format        Auto-format with ruff
#   make mlflow-ui     Launch MLflow tracking UI
#   make clean         Remove generated data and report artifacts
# =============================================================================

PYTHON      := .venv/bin/python
PIP         := .venv/bin/pip
PYTEST      := .venv/bin/pytest
RUFF        := .venv/bin/ruff
CONFIG_DIR  := config
BUDGET      := 5000

.PHONY: help install pipeline test coverage lint format mlflow-ui clean

# Default target
help:
	@echo ""
	@echo "CLV Long-Term Optimization — available targets:"
	@echo ""
	@echo "  install       Create .venv and install all dependencies"
	@echo "  pipeline      Run full end-to-end pipeline (BUDGET=5000)"
	@echo "  test          Run pytest unit test suite"
	@echo "  coverage      Run tests with HTML coverage report"
	@echo "  lint          Lint source with ruff"
	@echo "  format        Auto-format source with ruff"
	@echo "  mlflow-ui     Open MLflow experiment tracking UI (port 5000)"
	@echo "  clean         Remove generated artifacts (keeps raw data)"
	@echo ""

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✓  Environment ready. Activate with: source .venv/bin/activate"

pipeline:
	$(PYTHON) -m src.pipelines.weekly_scoring_pipeline \
		--config-dir $(CONFIG_DIR) \
		--budget $(BUDGET)
	@echo "✓  Pipeline complete. Results in data/processed/ and reports/"

test:
	$(PYTEST) tests/ -v --tb=short

coverage:
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "✓  Coverage report written to htmlcov/index.html"

lint:
	$(RUFF) check src/ tests/

format:
	$(RUFF) check --fix src/ tests/
	$(RUFF) format src/ tests/

mlflow-ui:
	@echo "Launching MLflow UI at http://localhost:5000 ..."
	$(PYTHON) -m mlflow ui --port 5000

clean:
	rm -f data/interim/*.parquet
	rm -f data/processed/*.parquet
	rm -f data/processed/*.joblib
	rm -f reports/figures/*.png
	rm -f reports/tables/*.csv
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓  Artifacts removed. Raw data preserved at data/raw/"
