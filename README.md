# CO2 physics-guided diffusion experiments

Codebase for the CO2 climate transition pathway generation experiments, including the sensitivity experiment pipeline used for the manuscript.

## Sensitivity experiments

The core sensitivity jobs are stored in `sensitivity/jobs_core/`.

Submit all independent GPU jobs from the Slurm login node:

```bash
cd /lustre/home/2501111516/work/碳排放/co2-0204-best-sensitivity
bash sensitivity/submit_core_sensitivity_all.sh
```

Each job requests one `GPU80G` GPU with high QoS. Results can be collected after completion with:

```bash
python3 sensitivity/scripts/collect_core_sensitivity_results.py
```

The generated summary is written to `sensitivity/results/core_sensitivity_summary.csv` and is intentionally ignored by git.

## Notes

Large data, checkpoints, generated figures, logs and experiment result folders are excluded from version control through `.gitignore`.
