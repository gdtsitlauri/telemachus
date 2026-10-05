#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-tested.txt
.venv/bin/python -m pip install --no-deps -e '.[modern,dev]'
.venv/bin/python -m pytest -q
.venv/bin/telemachus episode --seed 100 --config configs/main.json --output results/local_demo.json
