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

for job in "${JOBS[@]}"; do
  submit_out=$(sbatch "$job")
  job_id=$(echo "$submit_out" | awk '{print $NF}')
  echo "[submitted] job_id=$job_id script=$job"
done

echo "[queue] submitted ${#JOBS[@]} independent jobs. Each job requests one GPU; Slurm will start jobs as GPUs become available."
