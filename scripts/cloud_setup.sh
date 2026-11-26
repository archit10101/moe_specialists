#!/usr/bin/env bash
# Paste into the cloud environment's "Setup script" box (claude.ai/code → environment settings).
# Runs as root before the session starts; the result is cached when it finishes in under ~5 minutes.
# Needs "download.pytorch.org" in the environment's Allowed domains (CPU torch wheels come from there).
set -uo pipefail
python3 -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu || true
python3 -m pip install --quiet numpy scipy scikit-learn pytest tiktoken transformers huggingface_hub || true
python3 -c "import torch, numpy, sklearn; print('cloud setup OK: torch', torch.__version__)" || true
