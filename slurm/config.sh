# Cluster settings for MIT ORCD (Engaging). Every value can be overridden from the environment:
#     GPU_TYPE=h200 ./slurm/submit_all.sh
# Experiment settings do NOT live here - they are all in repro/config.py.
#
# ORCD facts this relies on (same as subject-moe):
#   login:  ssh <kerberos>@orcd-login.mit.edu  (password + Duo). Login nodes have internet.
#   compute nodes are treated as offline: setup_login.sh downloads everything into ./cache first.
#   mit_normal_gpu: 6 h max per job, 2 GPUs per user at a time. ALWAYS request a GPU type -
#   an untyped request lands on a slow L40S.   mit_normal: CPU only, 12 h max.
#   GPUs there: 1 H100 node (4 GPUs) vs 8 H200 nodes (8 GPUs each), so H200 is the default.

: "${GPU_PARTITION:=mit_normal_gpu}"
: "${CPU_PARTITION:=mit_normal}"
# One type ("h200") or a comma list ("h200,h100" = take whichever is free first). A list becomes
#   --gres=gpu:1 --constraint="h200|h100" when sinfo shows those names as node features of the
#   partition; otherwise submit_all.sh falls back to the FIRST type with -G <type>:1 (never an L40S).
#   H100 and H200 are both Hopper (same kernels, bf16); the H200 has more memory and bandwidth.
: "${GPU_TYPE:=h200,h100}"
# The smoke test (--smoke) only checks that every stage runs on tiny data, so it takes the far less
# contended L40S nodes and a short time limit: short limits let SLURM backfill the jobs into gaps.
: "${SMOKE_GPU_TYPE:=l40s}"
: "${SMOKE_TIME:=00:30:00}"
# Scheduling order (each Experiment has a priority 0..3 in repro/config.py): a job is submitted with
#   --nice = PRIORITY_NICE * priority + SEED_NICE * (index of its seed)
# so important experiments run first and their first seeds before their later seeds. Under the per-user GPU
# cap this decides what is finished first. PRIORITY_NICE=0 SEED_NICE=0 gives everything equal priority.
: "${PRIORITY_NICE:=10000}"
: "${SEED_NICE:=6000}"
: "${SLURM_ACCOUNT:=}"                 # optional -A account
: "${MAIL_USER:=}"                     # e-mail for FAIL (and END of the report job); empty = none
: "${JOB_PREFIX:=repro_}"              # prefix used to find this project's jobs in squeue
: "${EXTRA_SBATCH:=}"                  # extra flags for every job, e.g. "--qos=foo"

: "${GPU_CPUS:=8}"
: "${GPU_MEM:=64G}"
: "${CPU_CPUS:=4}"                     # CPU jobs: data preparation (if not done on the login node) and the report;
: "${CPU_MEM:=16G}"                    # small requests start sooner on the busy CPU partition

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
