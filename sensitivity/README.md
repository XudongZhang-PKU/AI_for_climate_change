# CO2 sensitivity experiments

First round: data scaling sensitivity for the paper.

Design: 4 training fractions x 3 seeds = 12 Slurm jobs.

- `train_fraction`: 0.25, 0.50, 0.75, 1.00
- seeds and split seeds: 1234, 2026, 3407
- model: Full USTGnet, default physics guidance and projection settings
- evaluation: held-out validation metrics in normalized model space, exported in each `训练记录.csv`

Submit from the Slurm login node:

```bash
cd /lustre/home/2501111516/work/碳排放/co2-0204-best-sensitivity
bash sensitivity/submit_datascaling_max3.sh
```

The submitter uses three dependency chains, so at most three GPU jobs run concurrently.

Collect results after jobs finish:

```bash
cd /lustre/home/2501111516/work/碳排放/co2-0204-best-sensitivity
python3 sensitivity/scripts/collect_sensitivity_results.py
```

Output summary: `sensitivity/results/datascaling_summary.csv`.

Recommended later rounds: physics guidance strength (`soft_eta_max`), reverse sampling steps (`sample_steps`), projection strength (`reverse_projection_scale`), and broader seed stability.
