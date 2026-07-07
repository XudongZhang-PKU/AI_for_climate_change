#!/usr/bin/env python3
import csv
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = root / 'sensitivity' / 'results' / 'datascaling_summary.csv'
rows = []
for csv_path in sorted(root.glob('fig2_data_prepare_*sens_datascale*/训练记录.csv')):
    with csv_path.open(newline='') as f:
        data = list(csv.DictReader(f))
    val = [r for r in data if r.get('split') == 'val']
    if not val:
        continue
    def epoch_num(r):
        try:
            return float(r.get('epoch', 'nan'))
        except ValueError:
            return -1
    best = max(val, key=epoch_num)
    best['result_dir'] = csv_path.parent.name
    rows.append(best)

preferred = [
    'result_dir', 'exp_tag', 'split', 'epoch', 'seed', 'split_seed', 'save_suffix',
    'train_fraction', 'sample_steps', 'soft_eta_max', 'reverse_push', 'n_samples',
    'macro_MAE', 'macro_RMSE', 'macro_MMD2', 'vet_pass_rate', 'ckpt', 'train_loss_avg'
]
extra = []
for r in rows:
    for k in r.keys():
        if k not in preferred and k not in extra:
            extra.append(k)
fields = preferred + extra
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in sorted(rows, key=lambda x: (float(x.get('train_fraction') or 0), int(float(x.get('seed') or 0)))):
        writer.writerow({k: r.get(k, '') for k in fields})
print(f'[collect] wrote {out} with {len(rows)} rows')
