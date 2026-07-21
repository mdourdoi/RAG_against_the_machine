RUN := uv run python -B
PYPROJECT_TOML := pyproject.toml

ARGS ?= index
FLAKE8 := $(RUN) -m flake8
MYPY := $(RUN) -m mypy

INSTALL_STAMP := .installed
VLLM_DIR := data/raw/vllm-0.10.1
VLLM_ZIP := data/raw/vllm-0.10.1.zip

all: install

$(INSTALL_STAMP): $(PYPROJECT_TOML)
	uv sync
	@touch $(INSTALL_STAMP)

$(VLLM_DIR): $(VLLM_ZIP)
	@unzip -o $(VLLM_ZIP) -d data/raw

install: $(INSTALL_STAMP) $(VLLM_DIR)

run: $(INSTALL_STAMP)
	@$(RUN) -m src $(ARGS)

debug: $(INSTALL_STAMP)
	@$(RUN) -m pdb -m src $(ARGS)

lint: $(INSTALL_STAMP)
	$(FLAKE8) src
	$(MYPY) src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

clean:
	@rm -rf $(INSTALL_STAMP) $(VLLM_DIR)
	@rm -rf .mypy_cache .venv data/processed data/output
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

.PHONY: install run debug clean lint