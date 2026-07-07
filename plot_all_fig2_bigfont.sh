#!/bin/bash
#SBATCH -J plot_all_fig2_bigfont
#SBATCH -p C064M0256G
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 5-00:00:00
#SBATCH -o /lustre/home/2501111516/work/碳排放/co2-0204-best/logs_figs/%x.%j.out
#SBATCH -e /lustre/home/2501111516/work/碳排放/co2-0204-best/logs_figs/%x.%j.err

set -euo pipefail

PROJECT_DIR="/lustre/home/2501111516/work/碳排放/co2-0204-best"
cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/logs_figs"

echo "JobID: ${SLURM_JOB_ID:-local}"
echo "NodeList: ${SLURM_NODELIST:-local}"
echo "WorkDir: $PROJECT_DIR"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-64}"

source /lustre/home/2501111516/software/miniconda3/etc/profile.d/conda.sh
conda activate co2

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MPLBACKEND=Agg

which python
python -V

run_plot() {
  local name="$1"
  shift
  echo
  echo "========== [$name] start: $(date '+%F %T') =========="
  srun --ntasks=1 python "$@"
  echo "========== [$name] done : $(date '+%F %T') =========="
}

# Fig.2(a): QQ overlay, category QQ, and dimension reduction.
run_plot "fig2a" experiments/plot_fig2a_my.py

# Fig.2(b): sector circular heatmap. Inner-ring values are listed below the plot.
run_plot "fig2b" experiments/plot_fig2b_new.py

# Fig.2(d): reuse cached bootstrap stats when available.
run_plot "fig2d" experiments/plot_tray.py --reuse_cached_stats

# Fig.2(f): semantic validity confusion/ribbon figure.
run_plot "fig2f" experiments/plot_fig2f.py

# Fig.2(g): ablation radar. Reuses figs/fig2g summary CSV unless --force_recompute is added.
run_plot "fig2g" experiments/plot_fig2g.py

echo
echo "All Fig.2 plots finished: $(date '+%F %T')"
