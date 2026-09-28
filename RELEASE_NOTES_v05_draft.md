# Release Notes Draft — alpha-ion-channel-models v05 (2026-09-22)

Companion code release for the manuscript *"Alpha-function ion-channel models extracted directly from data"* (submission package v05).

## What this release contains

- **125 tracked analysis scripts**, renamed to ASCII English filenames, covering the full pipeline:
  - hERG alpha-model forward simulation and parameter extraction (code 13003 identity smoke test chain)
  - model62 identifiability audit (de173–de182 stack, Fisher analysis, nine-cell parallel adjudication)
  - IKs / INa / ICaL / Kv4.3 single-channel extraction arms
  - StageB CiPA rescue: 28-drug library, three-arm drug scoring (official / alpha dynamic / alpha static), ORd host-model integration (B2/B3)
  - figure suite for main text and Extended Data (v05)
- Result JSON/CSV artefacts for all headline numbers (APD90 difference 0.36%, retrospective agreement 1.1e-5, qNet 0.998, scoring agreement 0.833–0.857).
- Manuscript v05 + SI (Markdown and docx) under `01_manuscript/` (archived in the workspace; this repo carries the code and data layer).

## Reproducibility anchors

- `de182.py` digit-exact identity sanity: R² values reproduce judge-canonical references to |Δ| = 0.00e+00; `judge182.py` recomputation verdict archived (`judge182_verdict.json`).
- CiPA StageB arms re-executed locally 2026-09-26/27; B1 official static arm requires hours (documented).
- All random seeds frozen; see each script header.

## Provenance note

All scripts were translated to English filenames/comments in a single pass (2026-09-26); the de182 digit-exact sanity suite was re-run after renaming to verify no behavioural change. Chinese-named originals remain in the local archive.

## Citation

See the companion preprint (bioRxiv, pending) and the sister repository `cell-communication-audit` (tag v06).
