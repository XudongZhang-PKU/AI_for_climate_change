#!/usr/bin/env bash
#SBATCH -J co2_sens_lambda_0p5x
#SBATCH -p GPU80G
#SBATCH --qos=high
#SBATCH -o sensitivity/logs/%x_%j.out
#SBATCH -e sensitivity/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00

set -euo pipefail
cd "/lustre/home/2501111516/work/碳排放/co2-0204-best-sensitivity"

source /lustre/home/2501111516/software/miniconda3/etc/profile.d/conda.sh
conda activate co2

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

OUT_DIR="fig2_data_prepare_Full_2026_sens_lambda_0p5x"
if [ -s "$OUT_DIR/训练记录.csv" ] && tail -n +2 "$OUT_DIR/训练记录.csv" | awk -F, '$2=="val" && $3>=300 {found=1} END {exit !found}'; then
  echo "[skip] completed $OUT_DIR"
  exit 0
fi

echo "[run] group=lambda label=0p5x seed=2026 split_seed=2026"
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
  --lambda_region 0.001 \
  --lambda_var 0.001 \
  --lambda_nn 0.002 \
  --use_soft_guidance 1 \
  --soft_guidance_schedule exp \
  --soft_eta_min 0.0 \
  --soft_eta_max 0.001 \
  --constraint_target_ratio 0.35 \
  --constraint_ratio_smooth 0.7 \
  --constraint_gain_min 0.5 \
  --constraint_gain_max 4.0 \
  --constraint_warmup_steps 50 \
  --reverse_projection_scale 2 \
  --train_fraction 1.0 \
  --seed 2026 \
  --split_seed 2026 \
  --save_suffix sens_lambda_0p5x \
  --exp_tag sens_lambda_0p5x

echo "[done] group=lambda label=0p5x $(date)"
