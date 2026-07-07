#!/bin/bash
#SBATCH -J co2_notcn_tmp
#SBATCH -p GPU40G
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH -t 5-00:00:00
#SBATCH -o logs/co2_notcn_tmp.%j.out
#SBATCH -e logs/co2_notcn_tmp.%j.err

set -euo pipefail
mkdir -p logs

echo "JobID: $SLURM_JOB_ID"
echo "NodeList: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

cd /lustre/home/2501111516/work/co2-0204-best

source /lustre/home/2501111516/software/miniconda3/etc/profile.d/conda.sh
conda activate co2

which python
python -V
nvidia-smi

srun python train_val_test.py \
  --ablation Full \
  --denoiser UGnetNoTCN \
  --split_seed 1234 \
  --seed 1234
