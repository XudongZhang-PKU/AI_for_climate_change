#!/bin/bash
#SBATCH -J plotfigb
#SBATCH -p C064M0256G
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 5-00:00:00
#SBATCH -o logs_figs/%x.%j.out
#SBATCH -e logs_figs/%x.%j.err

set -euo pipefail
mkdir -p logs_figs

echo "JobID: $SLURM_JOB_ID"
echo "NodeList: $SLURM_NODELIST"

cd /lustre/home/2501111516/work/co2-0204-best/experiments

# 关键：让 conda 命令可用
# source ~/.bashrc
which conda
# source /lustre/home/2501111516/software/miniconda3/bin/conda.sh
source /lustre/home/2501111516/software/miniconda3/etc/profile.d/conda.sh
conda activate co2

which python
python -V

# 建议用 srun 包起来
srun python plot_fig2b.py