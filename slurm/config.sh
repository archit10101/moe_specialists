# Cluster settings for MIT ORCD (Engaging). Every value can be overridden from the environment:
#     GPU_TYPE=h200 ./slurm/submit_all.sh
# Experiment settings do NOT live here - they are all in repro/config.py.
#
# ORCD facts this relies on (same as subject-moe):
#   login:  ssh <kerberos>@orcd-login.mit.edu  (password + Duo). Login nodes have internet.
#   compute nodes are treated as offline: setup_login.sh downloads everything into ./cache first.
#   mit_normal_gpu: 6 h max per job, 2 GPUs per user at a time. ALWAYS request a GPU type -
#   an untyped request lands on a slow L40S.   mit_normal: CPU only, 12 h max.

: "${GPU_PARTITION:=mit_normal_gpu}"
: "${CPU_PARTITION:=mit_normal}"
: "${GPU_TYPE:=h100}"                  # h100 or h200
: "${SLURM_ACCOUNT:=}"                 # optional -A account
: "${MAIL_USER:=}"                     # e-mail for FAIL (and END of the report job); empty = none
: "${JOB_PREFIX:=repro_}"              # prefix used to find this project's jobs in squeue
: "${EXTRA_SBATCH:=}"                  # extra flags for every job, e.g. "--qos=foo"

: "${GPU_CPUS:=8}"
: "${GPU_MEM:=64G}"
: "${CPU_CPUS:=16}"
: "${CPU_MEM:=48G}"

: "${CONDA_MODULE:=miniforge}"
: "${CONDA_ENV:=repro_moe}"
: "${PYTHON_VERSION:=3.11}"
: "${TORCH_VERSION:=2.14.1}"           # the version subject-moe ran on ORCD H100s
: "${TORCH_INDEX_URL:=https://download.pytorch.org/whl/cu126}"

# Where data / checkpoints / results go. ~60 GB for everything; prefer pool or scratch over home.
: "${REPRO_WORK:=$REPRO_ROOT/work}"
: "${REPRO_PROFILE:=full}"
: "${REPRO_MAX_MINUTES:=340}"          # a training job checkpoints and exits (code 3) after this long
: "${REPRO_CKPT_MINUTES:=15}"
