#!/usr/bin/env python3
import csv
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = root / 'sensitivity' / 'results' / 'core_sensitivity_summary.csv'
rows = []
for csv_path in sorted(root.glob('fig2_data_prepare_*sens_*/*记录.csv')):
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
    row = max(val, key=epoch_num)
    row['result_dir'] = csv_path.parent.name
    suffix = row.get('save_suffix', '')
    parts = suffix.split('_')
    row['sens_group'] = parts[1] if len(parts) > 1 else ''
    row['sens_label'] = '_'.join(parts[2:]) if len(parts) > 2 else ''
    rows.append(row)

preferred = [
    'sens_group', 'sens_label', 'result_dir', 'exp_tag', 'split', 'epoch', 'seed', 'split_seed', 'save_suffix',
    'train_fraction', 'sample_steps', 'beta_end', 'soft_eta_max', 'reverse_push',
    'lambda_region', 'lambda_var', 'lambda_nn', 'n_samples',
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
    for r in sorted(rows, key=lambda x: (x.get('sens_group',''), x.get('sens_label',''), int(float(x.get('seed') or 0)))):
        writer.writerow({k: r.get(k, '') for k in fields})
print(f'[collect] wrote {out} with {len(rows)} rows')
