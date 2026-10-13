# Contributing

Thanks for your interest in improving the Intelligent Document Engine.

## Development setup

```bash
python -m venv .venv
# activate it, then
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

## Before you open a pull request

```bash
ruff format .                 # formatting
ruff check .                  # lint (pyflakes, bugbear, pyupgrade, naming, pytest style, ...)
python -m pytest -q           # unit + integration tests
idengine eval                 # quality regression gate against the committed baseline
```

All four must pass; CI runs the same commands.

## Guidelines

- **Typed, small modules.** Public functions and methods have type hints. Keep packages within
  their layer (see ARCHITECTURE.md): primitives must not import services or the API.
- **Typed errors.** Raise an `idengine.errors.EngineError` subclass with a clear message; never
  return error strings or raise bare `Exception` for expected failures.
- **Validate at boundaries.** API schemas (`api/schemas.py`) forbid unknown fields; library entry
  points validate sizes and ranges.
- **Tests with every change.** Unit tests live in `tests/unit`, end-to-end tests (engine, API,
  CLI) in `tests/integration`. Generate binary fixtures with `tests/builders.py` rather than
  committing files. Tests must not use the network; mock HTTP with `httpx.MockTransport`.
- **Determinism.** Anything random takes a seed (HNSW levels, k-means, SVD, splits) so tests and
  evaluation results are reproducible.
- **Quality changes.** If you change retrieval, reranking or answering, run `idengine eval
  --output report.json`, review the per-query details, and only then refresh the baseline with
  `idengine eval --update-baseline`, explaining the change in the pull request.
- **Dependencies.** Prefer the standard library and numpy. A new runtime dependency must be pure
  Python or ship wheels for all platforms; pin it in `pyproject.toml` and regenerate
  `requirements.lock`.
- **Commits.** Small, focused commits with imperative messages ("Add IVF adaptive probing").

## Reporting issues

Include the version (`idengine --version`), the command or request, the error envelope (with its
`request_id`) and relevant log lines. Never include real API keys or confidential documents.
