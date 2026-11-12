# α-Ion-Channel Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23008848.svg)](https://doi.org/10.5281/zenodo.23008848)
[![Deposit](https://zenodo.org/badge/DOI/10.5281/zenodo.23042211.svg)](https://doi.org/10.5281/zenodo.23042211)

Mechanism-anchored, data-extracted kinetic parameters for cardiac ion
channels, with cross-laboratory validation and ORd host-model integration /
CiPA drug scoring.

## v2.0.0 — Unified reanalysis (2026-11-12)

The repository now carries the **unified reanalysis**: all corpora were
re-extracted from raw waveforms by one profile-likelihood pipeline and merged
into a single master table (487,940 measured quantities, 20,709 cells, 74
channel types, seven families), the basis of the v2 manuscript
*Gating time constants are the conserved coordinates of cardiac ion channels:
a unified reanalysis of seven voltage-clamp corpora*.

| New path | Content |
|---|---|
| `01_manuscript/unified_2026-11-12/` | v2 manuscript + SI (LaTeX sources and compiled PDFs, 13 + 13 pages) |
| `02_figures/unified_v2/` | The six v2 main figures (PNG, 300 dpi) |
| `03_data_csv/unified_v2/` | `master_unified_v1.csv` (487,940 rows), `cipa2_cells.csv`, the frozen four-calibre verdict table, the CiPA IC50 anchor comparison, and `reproduce_check.py` |
| `04_code/unified_collection_2026-11-12/` | All 126 pipeline scripts, classified into 11 stages, with `README.md` (per-script manifest) and `REPRODUCE.md` (third-party guide) |

### One-command verification

```bash
cd 03_data_csv/unified_v2
python reproduce_check.py
```

37 checks recompute every headline number from the deposited tables (master
table size and per-source row counts, profile-interval calibration 96.2%,
the five verdict tables 63/64 @ 2.47 and companions, CiPA tau ratio 0.948
n=970 with per-laboratory and blinded-arm values, Beattie anchor
-8.97 mV). Run record of this release: 37/37 ALL PASS.

Headline v2 results: gating time constants are conserved across cells while
voltage midpoints drift (63 of 64 activation groups, median scatter ratio
2.47; deactivation 43/48; inactivation 28/35; recovery 39/42; HCN 5/5;
9 of 10 published-model layers; 38 of 38 cross-species pairs). Temperature
moves kinetics by process-specific Q10 and leaves midpoints flat; drugs move
occupancy, not kinetics (median tau ratio 0.948, flat against block depth,
blinded arm concordant). Models assembled from the measured tables with one
free conductance per cell predict 67 of 68 held-out CiPA screening traces and
integrate into the O'Hara-Rudy host (APD90 difference 0.36%, qNet rho 0.998).

## v1 — Four-channel manuscript (archived)

The earlier four-channel study
(*Data-extracted gating kinetics yield parameter-efficient, audit-ready
models of four cardiac ion channels*; Research Square preprint
[doi:10.21203/rs.3.rs-11218939/v1](https://doi.org/10.21203/rs.3.rs-11218939/v1))
remains in place: `01_manuscript/` (v05.x files), `02_figures/`,
`03_data_csv/`, `04_code/code_package_v03/`, `05_verdict_cards/`,
`05_validation_records_index.md`. v1 numbers are superseded by the unified
reanalysis above, which re-extracts every corpus with the profile-likelihood
pipeline.

### Layout (v1 archive)

| Path | Content |
|---|---|
| `01_manuscript/` | v1 main text and SI (Markdown + Word) |
| `02_figures/` | v1 main and Extended Data figures (PNG + SVG) |
| `03_data_csv/` | v1 numeric outputs behind the figures/tables (CSV) |
| `04_code/code_package_v03/` | v1 pipeline: extraction, forward validation, population validation, drug-grammar scoring, figure generation; the previously symlinked identifiability artifacts (`06_identifiability_slowmemory/`) are now checked in as real files |
| `05_validation_records_index.md` | Registry of all validation cards |

## Reproduce

Python 3.10+; pandas, numpy, scipy, matplotlib (pynwb only for the
Channelpedia raw stage). For v2 see
`04_code/unified_collection_2026-11-12/REPRODUCE.md`. For v1, a full
local-machine re-run ledger reconciled against the sealed verdict values is
in `REPRODUCIBILITY.md`.

## License

For academic use; manuscript under review.
