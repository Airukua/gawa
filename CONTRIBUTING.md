# Contributing to GAWA

Thanks for your interest in contributing to **GAWA**! This guide explains how to set up the project, propose changes, and submit a pull request.

---

## Quick Start

```bash
git clone https://github.com/AiRukua/gawa.git
cd gawa
pip install -e ".[dev]"
```

---

## Project Layout

- `model/`: Core encoder/decoder modules
- `training/`: Training loop, config, scheduler, checkpoints
- `data/`: Dataset preparation utilities
- `eval/`: Evaluation and encoding helpers
- `scripts/`: CLI entrypoints
- `configs/`: YAML configs for experiments
- `tests/`: Unit and e2e tests

---

## Development Workflow

1. Create a feature branch off `main`.
2. Keep changes focused and small when possible.
3. Update or add tests for behavior changes.
4. Update docs if user-facing behavior changes.

---

## Running Tests

```bash
pytest
```

To run a subset:

```bash
pytest tests/unit/test_training_loop.py
```

---

## Code Style

- Prefer clear, explicit code over clever shortcuts.
- Add short comments only when the intent is not obvious.
- Keep docstrings accurate and concise.
- Avoid breaking public APIs without discussion.

---

## Reporting Issues

Please include:

- A clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Your environment (Python version, OS, GPU/CPU)

---

## Pull Request Checklist

- Tests pass locally
- New tests added for new behavior
- README/docs updated if needed
- No unrelated changes bundled in the PR

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
