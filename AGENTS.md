# Project Guidelines

## Python & Package Management
- Always use the `uv` package manager for all Python execution, testing, and package management commands in this project:
  - Run scripts or one-liners with `uv run python ...` (do not invoke `python`, `python3`, or `.venv/bin/python` directly).
  - Run tests with `uv run pytest ...`.
  - Manage dependencies and virtual environments with `uv pip ...` or `uv add ...` (do not use `pip` directly).
