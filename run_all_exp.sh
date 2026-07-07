#!/bin/bash
#SBATCH -J co2_new_ablation
#SBATCH -p GPU40G
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
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

which conda
source /lustre/home/2501111516/software/miniconda3/etc/profile.d/conda.sh
conda activate co2
which python
python -V
nvidia-smi

# 固定对照随机种子
SEED=1234
SPLIT_SEED=1234
CW=1e-4


srun --output=logs/%x.%j.task_%t.out --error=logs/%x.%j.task_%t.err bash -c '
  source /lustre/home/2501111516/software/miniconda3/etc/profile.d/conda.sh
  conda activate co2
  cd /lustre/home/2501111516/work/co2-0204-best

  case $SLURM_PROCID in
    # 0) Full baseline (UGnet)
    0) python train_val_test.py --ablation Full --denoiser UGnet --split_seed '"$SPLIT_SEED"' --seed '"$SEED"' ;;
    # 1) 去除 Graph
    1) python train_val_test.py --ablation NoGraph --denoiser UGnet --split_seed '"$SPLIT_SEED"' --seed '"$SEED"' ;;
    # 2) 去除 Constraint（采样阶段）
    2) python train_val_test.py --ablation NoConstraint --denoiser UGnet --split_seed '"$SPLIT_SEED"' --seed '"$SEED"' ;;
    # 3) 去除 TCN
    3) python train_val_test.py --ablation Full --denoiser UGnetNoTCN --split_seed '"$SPLIT_SEED"' --seed '"$SEED"' ;;
    # 4-6) 分别去除 condition embedding（单变量消融）
    4) python train_val_test.py --ablation Full --denoiser UGnet --no_cat_cond 1    --split_seed '"$SPLIT_SEED"' --seed '"$SEED"' ;;
    5) python train_val_test.py --ablation Full --denoiser UGnet --no_region_cond 1 --split_seed '"$SPLIT_SEED"' --seed '"$SEED"' ;;
    6) python train_val_test.py --ablation Full --denoiser UGnet --no_var_cond 1    --split_seed '"$SPLIT_SEED"' --seed '"$SEED"' ;;
    # 7) 训练端 constraint loss 开
    7) python train_val_test.py --ablation Full --denoiser UGnet --enable_constraint_loss 1 --constraint_weight '"$CW"' --split_seed '"$SPLIT_SEED"' --seed '"$SEED"' ;;
  esac
'

# 可选手动命令示例
# srun python train_val_test.py --ablation Full --denoiser UGnet --split_seed 1999 --seed 1999
# srun python train_val_test.py --ablation Full --denoiser UGnetNoTCN --split_seed 1999 --seed 1999
# srun python train_val_test.py --ablation Full --denoiser UGnet --enable_constraint_loss 1 --constraint_weight 1e-4 --split_seed 1999 --seed 1999
# srun python train_val_test.py --ablation Full --denoiser TinyUGnet --split_seed 1999 --seed 1999
