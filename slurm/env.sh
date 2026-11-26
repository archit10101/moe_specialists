# Sourced by every script and job. Needs REPRO_ROOT. Loads config.sh, sets offline caches,
# defines repro_activate (conda env).
: "${REPRO_ROOT:?REPRO_ROOT is not set}"
source "$REPRO_ROOT/slurm/config.sh"
export REPRO_ROOT REPRO_WORK REPRO_PROFILE REPRO_MAX_MINUTES REPRO_CKPT_MINUTES JOB_PREFIX
export HF_HOME="$REPRO_ROOT/cache/hf"
export TIKTOKEN_CACHE_DIR="$REPRO_ROOT/cache/tiktoken"
export PYTHONUNBUFFERED=1
if [ "${REPRO_ONLINE:-0}" != "1" ]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
[ -n "${SLURM_CPUS_PER_TASK:-}" ] && export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

repro_activate() {
  [ "${REPRO_NO_CONDA:-0}" = 1 ] && return 0     # use whatever python is on PATH (laptops, cloud, dry runs)
  set +u
  if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
      [ -f "$f" ] && source "$f" && break
    done
  fi
  type module >/dev/null 2>&1 && module load "$CONDA_MODULE" >/dev/null 2>&1
  command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found (module load $CONDA_MODULE)" >&2; set -u; return 1; }
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV" || { echo "ERROR: conda env $CONDA_ENV missing - run slurm/setup_login.sh" >&2; set -u; return 1; }
  set -u
}
