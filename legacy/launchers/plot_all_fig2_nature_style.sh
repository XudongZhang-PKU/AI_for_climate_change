#!/bin/bash
#SBATCH -J plot_all_fig2_ns
#SBATCH -p C064M1024G
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 5-00:00:00
#SBATCH -o /lustre/home/2501111516/work/碳排放/co2-0204-best/logs_figs/%x.%j.out
#SBATCH -e /lustre/home/2501111516/work/碳排放/co2-0204-best/logs_figs/%x.%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lustre/home/2501111516/work/碳排放/co2-0204-best}"
PLOT_DIR="${PLOT_DIR:-${PROJECT_DIR}/experiments_new_nature_style}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/figs_nature}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs_figs}"

# 设为 1 时强制重算 fig2d bootstrap stats / fig2g summary CSV
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-0}"

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

echo "JobID: ${SLURM_JOB_ID:-local}"
echo "NodeList: ${SLURM_NODELIST:-local}"
echo "Project: ${PROJECT_DIR}"
echo "PlotDir: ${PLOT_DIR}"
echo "OutRoot: ${OUT_ROOT}"
echo "Force recompute: ${FORCE_RECOMPUTE}"

source /lustre/home/2501111516/software/miniconda3/etc/profile.d/conda.sh
conda activate co2

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MPLBACKEND=Agg

which python
python -V

FAILED=()
SKIPPED=()

score_data_dir() {
  local d="$1"
  local score=0
  [[ -f "${d}/fig2a_real_syn.npz" ]] && score=$((score + 20))
  [[ -f "${d}/fig2_raw_test.npz" ]] && score=$((score + 10))
  [[ -f "${d}/corr_real.npy" && -f "${d}/corr_syn.npy" ]] && score=$((score + 15))
  [[ -f "${d}/X_train.npy" && -f "${d}/y_train.npy" && -f "${d}/X_syn.npy" && -f "${d}/y_syn.npy" ]] && score=$((score + 12))
  [[ -f "${d}/embeddings.npy" ]] && score=$((score + 4))
  [[ -f "${d}/physics_eq_residual.npy" ]] && score=$((score + 4))
  [[ -f "${d}/convergence.csv" ]] && score=$((score + 4))
  [[ -f "${d}/metrics_dist_full.csv" ]] && score=$((score + 2))
  [[ -f "${d}/corr_distance.txt" ]] && score=$((score + 1))
  echo "${score}"
}

resolve_primary_data_dir() {
  if [[ -n "${DATA_DIR:-}" ]]; then
    echo "${DATA_DIR}"
    return 0
  fi

  local best_dir="" best_score=-1
  local candidate score

  candidate="${PROJECT_DIR}/fig2_data_prepare_Full_1234/fig2_data"
  if [[ -d "${candidate}" ]]; then
    score="$(score_data_dir "${candidate}")"
    if (( score > best_score )); then
      best_score="${score}"
      best_dir="${candidate}"
    fi
  fi

  shopt -s nullglob
  for candidate in "${PROJECT_DIR}"/fig2_data_prepare_*/fig2_data; do
    [[ -d "${candidate}" ]] || continue
    score="$(score_data_dir "${candidate}")"
    if (( score > best_score )); then
      best_score="${score}"
      best_dir="${candidate}"
    fi
  done
  shopt -u nullglob

  if [[ -n "${best_dir}" && "${best_score}" -ge 0 ]]; then
    echo "${best_dir}"
    return 0
  fi
  return 1
}

ensure_canonical_data_link() {
  local resolved_dir="$1"
  local canonical="${PROJECT_DIR}/fig2_data_prepare_Full_1234/fig2_data"

  if [[ "${resolved_dir}" == "${canonical}" ]]; then
    return 0
  fi

  mkdir -p "$(dirname "${canonical}")"
  ln -sfn "${resolved_dir}" "${canonical}"
  echo "[data] linked ${canonical} -> ${resolved_dir}"
}

has_fig2d_cache() {
  local stat_dir="${OUT_ROOT}/fig2d/stats"
  [[ -f "${stat_dir}/n_list.npy" ]] \
    && [[ -f "${stat_dir}/mean_center.npy" ]] \
    && [[ -f "${stat_dir}/mean_ci_width.npy" ]] \
    && [[ -f "${stat_dir}/q95_center.npy" ]] \
    && [[ -f "${stat_dir}/q95_ci_width.npy" ]]
}

resolve_big_npy() {
  if [[ -n "${BIG_NPY:-}" && -f "${BIG_NPY}" ]]; then
    echo "${BIG_NPY}"
    return 0
  fi

  local candidate
  for candidate in \
    "${PROJECT_DIR}/output/forecast/最终生成_250.npy" \
    "/lustre/home/2501111516/work/碳排放/co2-0204-best-for-gen/output/forecast/最终生成_250.npy" \
    "/lustre/home/2501111516/work/co2-0204-best-for-gen/output/forecast/最终生成_250.npy"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

run_plot() {
  local name="$1"
  shift
  echo
  echo "========== [${name}] start: $(date '+%F %T') =========="
  if srun --ntasks=1 python "$@"; then
    echo "========== [${name}] done : $(date '+%F %T') =========="
    return 0
  fi
  echo "========== [${name}] FAIL : $(date '+%F %T') =========="
  FAILED+=("${name}")
  return 1
}

skip_plot() {
  local name="$1"
  local reason="$2"
  echo
  echo "========== [${name}] SKIP: ${reason} =========="
  SKIPPED+=("${name}: ${reason}")
}

cd "${PROJECT_DIR}"

if ! DATA_DIR="$(resolve_primary_data_dir)"; then
  echo "[warn] 未找到 fig2_data 目录；fig2a/fig2b/fig2f 可能无法运行。"
  DATA_DIR="${PROJECT_DIR}/fig2_data_prepare_Full_1234/fig2_data"
else
  echo "[data] primary fig2_data = ${DATA_DIR} (score=$(score_data_dir "${DATA_DIR}"))"
  ensure_canonical_data_link "${DATA_DIR}"
fi

# Fig.2(a): distribution fidelity (Nature style)
if [[ -f "${DATA_DIR}/fig2a_real_syn.npz" ]]; then
  run_plot "fig2a" "${PLOT_DIR}/plot_fig2a_my.py"
else
  skip_plot "fig2a" "缺少 ${DATA_DIR}/fig2a_real_syn.npz"
fi

# Fig.2(b): sector circular heatmap (Nature style)
if [[ -f "${DATA_DIR}/corr_real.npy" && -f "${DATA_DIR}/corr_syn.npy" ]]; then
  run_plot "fig2b" "${PLOT_DIR}/plot_fig2b_new.py"
else
  skip_plot "fig2b" "缺少 corr_real.npy / corr_syn.npy"
fi

# Fig.2(d): sampling necessity; 优先复用 figs/fig2d/stats 缓存
FIG2D_ARGS=(--out_dir "${OUT_ROOT}/fig2d")
FIG2D_RUN=0
if [[ "${FORCE_RECOMPUTE}" == "1" ]]; then
  if BIG_NPY="$(resolve_big_npy)"; then
    FIG2D_ARGS+=(--force_recompute_metrics --big_npy "${BIG_NPY}")
    FIG2D_RUN=1
    echo "[data] fig2d force recompute from big_npy=${BIG_NPY}"
  else
    skip_plot "fig2d" "FORCE_RECOMPUTE=1 但未找到 big_npy"
  fi
elif has_fig2d_cache; then
  FIG2D_ARGS+=(--reuse_cached_stats)
  FIG2D_RUN=1
  echo "[cache] fig2d stats hit: ${OUT_ROOT}/fig2d/stats"
else
  if BIG_NPY="$(resolve_big_npy)"; then
    FIG2D_ARGS+=(--big_npy "${BIG_NPY}")
    FIG2D_RUN=1
    echo "[data] fig2d will compute from big_npy=${BIG_NPY}"
  else
    skip_plot "fig2d" "无 stats 缓存且未找到 big_npy"
  fi
fi
if [[ "${FIG2D_RUN}" == "1" ]]; then
  run_plot "fig2d" "${PLOT_DIR}/plot_tray.py" "${FIG2D_ARGS[@]}"
fi

# Fig.2(f): semantic validity ribbons
if [[ -f "${DATA_DIR}/X_train.npy" && -f "${DATA_DIR}/y_train.npy" && -f "${DATA_DIR}/X_syn.npy" && -f "${DATA_DIR}/y_syn.npy" ]]; then
  run_plot "fig2f" "${PLOT_DIR}/plot_fig2f.py" --data_dir "${DATA_DIR}"
else
  skip_plot "fig2f" "缺少 X_train/y_train/X_syn/y_syn"
fi

# Fig.2(g): ablation radar; 默认复用 figs/fig2g/fig2g_ablation_metrics_summary.csv
FIG2G_ARGS=(--out "${OUT_ROOT}/fig2g/fig2g_ablation_radar_nature_polished.svg" --summary_csv "${OUT_ROOT}/fig2g/fig2g_ablation_metrics_summary.csv")
if [[ "${FORCE_RECOMPUTE}" == "1" ]]; then
  FIG2G_ARGS+=(--force_recompute)
else
  if [[ -f "${OUT_ROOT}/fig2g/fig2g_ablation_metrics_summary.csv" ]]; then
    echo "[cache] fig2g summary hit: ${OUT_ROOT}/fig2g/fig2g_ablation_metrics_summary.csv"
  fi
fi
run_plot "fig2g" "${PLOT_DIR}/plot_fig2g.py" "${FIG2G_ARGS[@]}"

# Fig.1 右下角示意图（不依赖 fig2_data）
mkdir -p "${OUT_ROOT}/schematic"
run_plot "schematic" "${PLOT_DIR}/plot_a_右下角.py" --out "${OUT_ROOT}/schematic/schematic_feasible_pathways_6x5cm_transparent.png"

echo
echo "========== Summary =========="
echo "Output root: ${OUT_ROOT}"
echo "Data dir   : ${DATA_DIR}"
if ((${#SKIPPED[@]})); then
  echo "Skipped (${#SKIPPED[@]}):"
  printf '  - %s\n' "${SKIPPED[@]}"
fi
if ((${#FAILED[@]})); then
  echo "Failed (${#FAILED[@]}):"
  printf '  - %s\n' "${FAILED[@]}"
  exit 1
fi
echo "All requested Fig.2 plots finished: $(date '+%F %T')"
