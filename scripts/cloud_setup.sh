#!/usr/bin/env bash
# Paste into the cloud environment's "Setup script" box (claude.ai/code → environment settings).
# Runs as root before the session starts; the result is cached when it finishes in under ~5 minutes.
# Allowed domains needed for the small CPU torch build: download.pytorch.org AND download-r2.pytorch.org
# (the first redirects to the second). If they are blocked, it falls back to the PyPI wheel (works, but
# it bundles CUDA libraries and is several GB larger).
set -uo pipefail
python3 -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu \
  || python3 -m pip install --quiet torch || true
python3 -m pip install --quiet numpy scipy scikit-learn pytest tiktoken transformers huggingface_hub || true
python3 -c "import torch, numpy, sklearn; print('cloud setup OK: torch', torch.__version__)" || true
