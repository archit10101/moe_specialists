# Source this file from the project root: `source ./env.sh`. It keeps every cache inside the root
# and disables telemetry. It does NOT grant execution approval (see MOE_RESEARCH_APPROVE).
umask 077
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [ "$_ROOT" != "/home/zapan4/moe-research" ]; then echo "env.sh: unexpected root $_ROOT" >&2; return 1 2>/dev/null || exit 1; fi
export MOE_ROOT="$_ROOT"
export TMPDIR="$_ROOT/.tmp" XDG_CACHE_HOME="$_ROOT/.cache/xdg"
export HF_HOME="$_ROOT/.cache/hf" HUGGINGFACE_HUB_CACHE="$_ROOT/.cache/hf/hub" HF_HUB_CACHE="$_ROOT/.cache/hf/hub" HF_DATASETS_CACHE="$_ROOT/.cache/hf/datasets"
export UV_CACHE_DIR="$_ROOT/.cache/uv" PIP_CACHE_DIR="$_ROOT/.cache/pip" TORCH_HOME="$_ROOT/.cache/torch"
export TRITON_CACHE_DIR="$_ROOT/.cache/triton" TORCHINDUCTOR_CACHE_DIR="$_ROOT/.cache/inductor" CUDA_CACHE_PATH="$_ROOT/.cache/cuda"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false MAX_JOBS=4
export WANDB_DISABLED=true WANDB_MODE=disabled HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TRANSFORMERS_NO_ADVISORY_WARNINGS=1 DISABLE_TELEMETRY=1
export PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 GIT_CONFIG_NOSYSTEM=1 GIT_TERMINAL_PROMPT=0
export CUDA_VISIBLE_DEVICES="${MOE_CUDA_VISIBLE_DEVICES:-}"   # CUDA hidden unless a Stage-B guard sets it to 0
export PYTHONPATH="$_ROOT/src"
unset _ROOT
