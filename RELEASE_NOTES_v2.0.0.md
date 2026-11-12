# Release v2.0.0 — Unified reanalysis (2026-11-12)

## What changed

All corpora were re-extracted from raw waveforms by one profile-likelihood
pipeline and merged into a single master table. The v1 four-channel
manuscript is archived in place; the v2 unified manuscript supersedes its
numbers.

- `01_manuscript/unified_2026-11-12/`: v2 manuscript + SI (LaTeX + PDF,
  13 + 13 pages)
- `02_figures/unified_v2/`: six v2 main figures
- `03_data_csv/unified_v2/`: `master_unified_v1.csv` (487,940 rows x 22
  columns), `cipa2_cells.csv`, frozen four-calibre verdict table, CiPA IC50
  anchor comparison, `reproduce_check.py`
- `04_code/unified_collection_2026-11-12/`: 126 scripts in 11 stages with
  per-script README and third-party REPRODUCE guide
- Fixed: three symlinks under `04_code/code_package_v03/` replaced by the
  real artifact files (model62 identifiability JSON set, construction-cell
  data, CiPA official anchors); the repository now clones self-contained

## Verification

`python reproduce_check.py` in `03_data_csv/unified_v2/`: 37/37 ALL PASS
(2026-11-12). Headline values: 63/64 activation groups conserved (median
ratio 2.47), profile-interval coverage 96.2% (n=255,411), CiPA tau ratio
0.948 (n=970), cross-laboratory prediction 67/68, ORd APD90 difference
0.36%, qNet rho 0.998.

## Data sources

All source datasets are public: Channelpedia (channelpedia.epfl.ch), CiPA
five-laboratory hERG (OSF osf.io/a6k5t), Beattie nine-cell (figshare
doi:10.6084/m9.figshare.4702546), Nav1.5 series (figshare
doi:10.6084/m9.figshare.27193878), CaV1.2 pharmacology (OSF g3msb), IKs
series (Zenodo 8226585), Lei five-temperature series (Biophys J 2019).
