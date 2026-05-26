RUN := uv run python3 -B
PYPROJECT_TOML := pyproject.toml

ARGS ?= index
FLAKE8 := $(RUN) -m flake8
MYPY := $(RUN) -m mypy

INSTALL_STAMP := .installed

$(INSTALL_STAMP): $(PYPROJECT_TOML)
	uv sync
	@touch $(INSTALL_STAMP)

install: $(INSTALL_STAMP)

run: $(INSTALL_STAMP)
	@$(RUN) -m src $(ARGS)

debug: $(INSTALL_STAMP)
	@$(RUN) -m pdb -m src $(ARGS)

lint: $(INSTALL_STAMP)
	$(FLAKE8) src
	$(MYPY) src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: $(INSTALL_STAMP)
	$(FLAKE8) src
	$(MYPY) src --strict

clean:
	@rm -rf $(INSTALL_STAMP)
	@find .  -type d -name __pycache__ -delete
	@rm -rf .mypy_cache .venv data/processed

.PHONY: install run debug clean lint lint-strict
