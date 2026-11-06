#!/bin/zsh
cd -- "$(dirname -- "$0")/.." || exit 1
if [[ -f results/repro/v4-batch-status.json ]]; then
    exec .venv/bin/python -m repro.v4_batch --stop
fi
if [[ -f results/repro/v3-batch-status.json ]]; then
    exec .venv/bin/python -m repro.v3_batch --stop
fi
exec .venv/bin/python -m repro.auto_resume --stop
