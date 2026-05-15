RUN := uv run python3 -B
PYPROJECT_TOML := pyproject.toml

FLAKE8 := $(RUN) -m flake8
MYPY := $(RUN) -m mypy

INSTALL_STAMP := .installed

$(INSTALL_STAMP): $(PYPROJECT_TOML)
	uv sync
	@touch $(INSTALL_STAMP)

install: $(INSTALL_STAMP)

run: $(INSTALL_STAMP)
	@$(RUN) -m rag_against_the_machine

debug: $(INSTALL_STAMP)
	@$(RUN) -m pdb -m rag_against_the_machine

lint: $(INSTALL_STAMP)
	$(FLAKE8) . --exclude .venv
	$(MYPY) . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: $(INSTALL_STAMP)
	$(FLAKE8) . --exclude .venv
	$(MYPY) . --strict

clean:
	@rm -rf $(INSTALL_STAMP)
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	@rm -rf .mypy_cache .pytest_cache build dist *.egg-info .venv uv.lock data/

.PHONY: install run debug clean lint lint-strict
