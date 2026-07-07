#!/bin/bash
set -euo pipefail
mkdir -p logs

cd /lustre/home/2501111516/work/co2-0204-best

SEED=1234
SPLIT_SEED=1234
CW=1e-4

submit_one() {
  local job_name="$1"
  local cmd="$2"

  sbatch \
    --job-name="$job_name" \
    --partition=GPU40G \
    --qos=normal \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --gres=gpu:1 \
    --time=5-00:00:00 \
    --output="logs/${job_name}.%j.out" \
    --error="logs/${job_name}.%j.err" \
    --export=ALL,TRAIN_CMD="$cmd" \
    --wrap='
      set -euo pipefail
      source /lustre/home/2501111516/software/miniconda3/etc/profile.d/conda.sh
      conda activate co2
      cd /lustre/home/2501111516/work/co2-0204-best
      echo "[JOB] $SLURM_JOB_NAME  id=$SLURM_JOB_ID"
      nvidia-smi
      eval "$TRAIN_CMD"
    '
}

# 1) 去除 TCN
submit_one "co2_notcn" \
  "python train_val_test.py --ablation Full --denoiser UGnetNoTCN --split_seed ${SPLIT_SEED} --seed ${SEED}"


# 3) 训练端 constraint loss
submit_one "co2_constraint_loss_on" \
  "python train_val_test.py --ablation Full --denoiser UGnet --enable_constraint_loss 1 --constraint_weight ${CW} --split_seed ${SPLIT_SEED} --seed ${SEED}"

# 4) Tiny
submit_one "co2_tiny" \
  "python train_val_test.py --ablation Full --denoiser TinyUGnet --split_seed ${SPLIT_SEED} --seed ${SEED}"

# 2) 分别去除 condition embedding
submit_one "co2_no_cat_cond" \
  "python train_val_test.py --ablation Full --denoiser UGnet --no_cat_cond 1 --split_seed ${SPLIT_SEED} --seed ${SEED}"
submit_one "co2_no_region_cond" \
  "python train_val_test.py --ablation Full --denoiser UGnet --no_region_cond 1 --split_seed ${SPLIT_SEED} --seed ${SEED}"
submit_one "co2_no_var_cond" \
  "python train_val_test.py --ablation Full --denoiser UGnet --no_var_cond 1 --split_seed ${SPLIT_SEED} --seed ${SEED}"


echo "Submitted all jobs in single-GPU mode."
