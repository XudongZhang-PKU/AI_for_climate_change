#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v sbatch >/dev/null 2>&1; then
  echo "[error] sbatch not found. Run this script from the Slurm login node, not the data node." >&2
  exit 1
fi

mapfile -t JOBS < <(find sensitivity/jobs_core -maxdepth 1 -name '*.sbatch' | sort)
if [ "${#JOBS[@]}" -eq 0 ]; then
  echo "[error] no core sensitivity jobs found" >&2
  exit 1
fi

MAX_PARALLEL=3
declare -a LAST_IDS
for ((i=0; i<MAX_PARALLEL; i++)); do LAST_IDS[$i]=""; done

for i in "${!JOBS[@]}"; do
  slot=$((i % MAX_PARALLEL))
  job="${JOBS[$i]}"
  if [ -n "${LAST_IDS[$slot]}" ]; then
    submit_out=$(sbatch --dependency=afterany:${LAST_IDS[$slot]} "$job")
  else
    submit_out=$(sbatch "$job")
  fi
  job_id=$(echo "$submit_out" | awk '{print $NF}')
  LAST_IDS[$slot]="$job_id"
  echo "[submitted] slot=$slot job_id=$job_id script=$job"
done

echo "[queue] submitted ${#JOBS[@]} jobs with at most $MAX_PARALLEL concurrent GPU jobs."
