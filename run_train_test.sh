#!/bin/bash
#SBATCH -J co2_gen
#SBATCH -p GPU40G
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH -t 5-00:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err

set -euo pipefail
mkdir -p logs

echo "JobID: $SLURM_JOB_ID"
echo "NodeList: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

cd /lustre/home/2501111516/work/co2-0204-best

# 关键：让 conda 命令可用（仅在主节点打印信息）
which conda
source /lustre/home/2501111516/software/miniconda3/etc/profile.d/conda.sh
conda activate co2
which python
python -V
nvidia-smi

# 多任务并行：每任务 1 GPU，SLURM_PROCID 区分 ablation；各任务 stdout/err 单独落盘避免混在一起
srun --output=logs/%x.%j.task_%t.out --error=logs/%x.%j.task_%t.err bash -c '
  source /lustre/home/2501111516/software/miniconda3/etc/profile.d/conda.sh
  conda activate co2
  cd /lustre/home/2501111516/work/co2-0204-best
  case $SLURM_PROCID in
    0) python train_val_test.py --ablation Full --split_seed 1234 --seed 1234 ;;
    1) python train_val_test.py --ablation NoGraph --split_seed 1234 --seed 1234 ;;
    2) python train_val_test.py --ablation NoConstraint --split_seed 1234 --seed 1234 ;;
  esac
'

# srun python train_val_test.py --ablation Full --split_seed 1999 --seed 1999
# srun python train_val_test.py --ablation NoGraph --split_seed 1999 --seed 1999
# srun python train_val_test.py --ablation NoConstraint --split_seed 1999 --seed 1999
# srun python train_val_test.py --ablation LinearDenoise --split_seed 1999 --seed 1999

# srun python train_val_test.py --ablation Full --split_seed 42 --seed 42
# srun python train_val_test.py --ablation NoGraph --split_seed 42 --seed 42