#!/usr/bin/env bash
#SBATCH -J co2_datascale_f025_s3407
#SBATCH -o sensitivity/logs/%x_%j.out
#SBATCH -e sensitivity/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail
cd "/lustre/home/2501111516/work/碳排放/co2-0204-best-sensitivity"

source /lustre/home/2501111516/software/miniconda3/etc/profile.d/conda.sh
conda activate co2

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

OUT_DIR="fig2_data_prepare_Full_3407_sens_datascale_frac025_seed3407"
if [ -s "$OUT_DIR/训练记录.csv" ] && tail -n +2 "$OUT_DIR/训练记录.csv" | awk -F, '$2=="val" && $3>=300 {found=1} END {exit !found}'; then
  echo "[skip] completed $OUT_DIR"
  exit 0
fi

echo "[run] datascale_f025_s3407 train_fraction=0.25 seed=3407"
echo "[env] $(hostname) $(date)"
python train_val_test.py \
  --ablation Full \
  --denoiser UGnet \
  --hidden_size 64 \
  --N 200 \
  --beta_end 0.1 \
  --sample_steps 45 \
  --n_blocks 2 \
  --lr 0.00005 \
  --train_batch_size 32 \
  --gen_batch_size 32 \
  --n_samples 64 \
  --epoch 300 \
  --export_fig2 0 \
  --lambda_region 0.002 \
  --lambda_var 0.002 \
  --lambda_nn 0.004 \
  --use_soft_guidance 1 \
  --soft_guidance_schedule exp \
  --soft_eta_min 0.0 \
  --soft_eta_max 0.001 \
  --constraint_target_ratio 0.35 \
  --constraint_ratio_smooth 0.70 \
  --constraint_gain_min 0.5 \
  --constraint_gain_max 4.0 \
  --constraint_warmup_steps 50 \
  --reverse_projection_scale 2 \
  --train_fraction 0.25 \
  --seed 3407 \
  --split_seed 3407 \
  --save_suffix sens_datascale_frac025_seed3407 \
  --exp_tag sens_datascale_frac025_seed3407

echo "[done] datascale_f025_s3407 $(date)"
