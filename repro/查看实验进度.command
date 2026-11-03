#!/bin/zsh
cd -- "$(dirname -- "$0")/.." || exit 1
exec .venv/bin/python -m repro.watch_progress "$@"
