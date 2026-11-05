# Training data

## Source and access

The source is the [AR6 Scenario Explorer and Database](https://iiasa.ac.at/models-tools-data/ar6-scenario-explorer-and-database), release 1.1. Download the World and R10 scenario tables and the metadata indicators from the [official downloads page](https://data.ece.iiasa.ac.at/ar6/#/downloads):

- `AR6_Scenarios_Database_World_v1.1.csv`
- `AR6_Scenarios_Database_R10_regions_v1.1.csv`
- `AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx`

Citation: Byers, E. et al. *AR6 Scenarios Database*, version 1.1. Integrated Assessment Modeling Consortium & International Institute for Applied Systems Analysis (2022). https://doi.org/10.5281/zenodo.7197970

The processed training bundle has been assembled and checked locally; it is not included in this Git checkout. The [AR6 source license](https://data.ene.iiasa.ac.at/ar6/static/License/Legal.html) restricts redistribution of substantial portions of the database. Public distribution of the full processed bundle requires a confirmed basis under those terms. The repository's MIT license applies to code, not the AR6 data.

## Processed inputs

The bundle contains 1,188 model–scenario pathways from 41 model versions and eight climate categories. Of these, 842 include regional observations. Each pathway has 528 region–variable groups (11 regions × 48 variables) and ten time points (2010–2100 at ten-year intervals).

| File under `SPM/` | Contents |
|---|---|
| `spm.npz` | Physical-value input arrays and observation masks |
| `region_baseline.csv` | Regional aggregation constraint baselines |
| `variable_baseline.csv` | Cross-variable constraint baselines |
| `变量编码对应表/dim_Model_encoding.csv` | 41 model-version codes |
| `变量编码对应表/dim_Category_encoding.csv` | Eight climate-category codes |
| `变量编码对应表/dim_Region_encoding.csv` | World and ten R10 region codes |
| `变量编码对应表/dim_Variable_encoding.csv` | 48 variable codes |
| `变量编码对应表/variables48.csv` | Variable names and units |

`spm.npz` contains `padded` (float32, shape `1188 × 528 × 10 × 5`) and `mask` (uint8, shape `1188 × 528`). The five channels contain model, category, region and variable codes, followed by the value. A mask of 1 marks an observed series; 0 marks an unobserved series. Arrays can be loaded with `numpy.load(..., allow_pickle=False)`.

The eight files total 19,533,005 bytes (18.63 MiB). The compressed NPZ preserves the original array contents exactly. File sizes and SHA-256 hashes are listed in the [manifest](training_data_manifest.json).

## Training

Place the eight files in `data/SPM/`, preserving the directory structure, then run:

```bash
python scripts/run_model.py train \
  --data-root data/SPM \
  --device cuda --epochs 300 \
  --seed 2026 --split-seed 2026 \
  --output-dir outputs/train
```

Training starts from scratch and does not require a pretrained checkpoint. The input bundle precedes splitting and normalization: the training command uses a fixed 80:10:10 path split, giving 950/119/119 paths before training-set resampling, and fits normalization on the training partition. Inference requires a compatible trained checkpoint; see the [usage guide](usage.md).

The historical preprocessing steps are recorded in [prepare_spm.ipynb](../preprocessing/prepare_spm.ipynb). They select and harmonize variables, filter paths by missingness, interpolate time points, and encode the fixed region–variable grid. Re-running the notebook also requires its study-specific `Variables48.csv` and `pairs_metadata.csv` inputs and the directory layout used there; the three source downloads alone are not a complete notebook input bundle.
