#!/usr/bin/env bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra cuda
source .venv/bin/activate
