#!/usr/bin/env bash
# Print prompt N from CLOUD_PROMPT.md (works from any directory; fails loudly if the prompt is empty).
#   claude --cloud "$(./scripts/cloud_prompt.sh 1)"
set -euo pipefail
n="${1:?usage: cloud_prompt.sh <prompt number>}"
file="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/CLOUD_PROMPT.md"
text="$(awk -v n="$n" '
  /^## Prompt / { inside = ($3 == n":") ; next }
  inside && /^```/ { if (fence) exit; fence = 1; next }
  inside && fence { print }' "$file")"
[ -n "$text" ] || { echo "cloud_prompt.sh: prompt $n not found in $file" >&2; exit 1; }
printf '%s\n' "$text"
