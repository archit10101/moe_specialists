#!/usr/bin/env bash
# One-time setup on the ORCD LOGIN node (it has internet; compute nodes are treated as offline).
#   1. conda env + torch (CUDA build) + requirements
#   2. downloads: 5 M2D2 arXiv archives (~11.7 GB), GPT-2 BPE, e5-base-v2 and MiniLM weights -> ./cache
#   3. offline check (what compute nodes will see) + CPU tests
# Safe to re-run. Takes ~20-40 min, mostly the download.
set -euo pipefail
REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPRO_ROOT
source "$REPRO_ROOT/slurm/config.sh"
cd "$REPRO_ROOT"
mkdir -p cache/hf cache/tiktoken "$REPRO_WORK/logs"

set +u
if ! type module >/dev/null 2>&1; then
  for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do [ -f "$f" ] && source "$f" && break; done
fi
module load "$CONDA_MODULE"
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then echo "== conda env $CONDA_ENV exists"
else echo "== creating conda env $CONDA_ENV"; conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"; fi
conda activate "$CONDA_ENV"
set -u
if python -c "import torch,sys; sys.exit(torch.__version__.split('+')[0] != '$TORCH_VERSION')" 2>/dev/null; then
  echo "== torch $TORCH_VERSION present"
else
  python -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"
fi
python -m pip install -r requirements.txt

echo "== downloads (online)"
export HF_HOME="$REPRO_ROOT/cache/hf" TIKTOKEN_CACHE_DIR="$REPRO_ROOT/cache/tiktoken" HF_HUB_DISABLE_PROGRESS_BARS=1
python -m repro.run prepare --download --profile full
python - <<'PY'
from huggingface_hub import snapshot_download
for m in ("intfloat/e5-base-v2", "sentence-transformers/all-MiniLM-L6-v2"):
    print("==", m, "->", snapshot_download(m, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"]))
PY

echo "== offline verification"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'PY'
from repro.config import FULL
from repro.data.prepare import download
download(FULL)
from transformers import AutoModel, AutoTokenizer
for m in ("intfloat/e5-base-v2", "sentence-transformers/all-MiniLM-L6-v2"):
    AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)
print("offline OK")
PY
python -m pytest -q
du -sh cache 2>/dev/null || true
echo
echo "setup complete. Next:"
echo "  ./slurm/submit_all.sh --smoke     # ~15 min end-to-end check on a GPU node"
echo "  ./slurm/submit_all.sh --bench     # measured GPU-hours for the real runs"
echo "  ./slurm/submit_all.sh             # the core reproduction"
