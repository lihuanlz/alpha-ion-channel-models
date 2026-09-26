# code_package_v03 · Four-channel data-computation code bundle

**Date**: 2026-09-21
**Convention**: this package is a **copy** (not a move) of all data-computation scripts from `04_细胞线4\α模型\`, `04_细胞线4\w182\`, and `04_细胞线4\文章\` — 84 scripts in 10 subfolders by business line. The originals remain in place; this package is the single entry point for audit and recomputation.
**Companion manuscript**: `Nature_main_v02_2026-09-21.md` + `Nature_SI_v02_2026-09-21.md`.
**Runtime**: Windows + Anaconda Python (numpy / scipy / pandas / matplotlib / pyabf / neo). Data roots are hard-coded as `D:\data` and workspace-relative paths; adjust them to your machine before recomputation.

Note: file names and JSON keys that reference archived result files are kept verbatim in Chinese so that every reference resolves one-to-one to the authors' archive.

---

## 01_hERG_Beattie9_extraction (24) — hERG nine-cell extraction · sealing · forward validation

Manuscript: Fig. 2a–d, Extended Data Fig. 1; SI S1.1, S2, S6.

| Script | Output | Manuscript anchor |
|---|---|---|
| 2026-09-13_alpha_amplitude_table_extraction.py | nine-cell amplitude/steady-state tables | S1.1c h_ss table |
| 2026-09-13_alpha_inact_gate_inact_protocol_seal_verdict.py | h_ss(V) 12-level seal | S1.1c; Fig. 2b |
| 2026-09-13_alpha_tauobs_verdict_repol_window_fast_relax_seal.py | tau_rec/tau_d repolarization-window four levels | S1.1b |
| 2026-09-13_alpha_tauobs_recon_testseg_early_phase.py | tau_obs upper-bound convention | S1.1d |
| 2026-09-13_alpha_act_envelope_tauact_seal_verdict.py | tau_act envelope seal / registry band | S1.1 tau_act section |
| 2026-09-13_alpha_act_kinetics_envelope_const_verdict.py | activation delay 50–150 ms | main-text hERG section |
| 2026-09-13_alpha_act_timecourse_ss_act_rise_verdict.py | m_ss anchors | S1.1f |
| 2026-09-13_alpha_pure_hssV_strip_verdict.py | B1 (h=1 simplification) rejection | main text "0.42 at −70 mV" |
| 2026-09-13_alpha_hssmeasured_table_rerun_AP_forwardverdict.py | AP forward 8/9 | Fig. 2d; S2 |
| 2026-09-13_alpha_hssmeasured_table_rerun_sineforward_verdict.py | sine forward 7/9 | Fig. 2c; S2 |
| 2026-09-13_alpha_AP_forwardverdict.py | earlier version of the above | archive |
| 2026-09-13_alpha_assembly_smoke_sineforward_verdict.py | assembly smoke | archive |
| 2026-09-13_alpha_rebound_hook_inact_recovery_const_verdict.py | tau_rec hook anchor | S1.1e G table anchor |
| 2026-09-13_alpha_forward_validation_tail_and_sine.py | tail whitening 28/33 | S2 |
| 2026-09-13_minus40_step_drift_assay.py | −40 mV level data-defect identification | S6 D registry |
| 2026-09-13_longtail_flatness_verdict_v5_nine_cells.py | long-tail verdict | S6 |
| 2026-09-14_alpha_tauobs_reanchor_AP_forwardverdict.py | re-anchoring sensitivity AP | S2 |
| 2026-09-14_alpha_tauobs_reanchor_sineforward_verdict.py | re-anchoring sensitivity sine | S2 |
| 2026-09-14_alpha_data_side_spike_baseline_recheck.py | 060/016 baseline shift >3x MAD | S6 data-side closure |
| 2026-09-14_alpha_B6_H1_highV_act_verdict.py | B6 candidate H1 verdict | S1.1 B6 section |
| 2026-09-14_alpha_B6_H2_quantitative_prediction_card.py | B6 H2 quantitative prediction | S1.1 B6 section |
| 2026-09-14_alpha_B6_H2_interbeat_accumulation_verdict.py | inter-beat accumulation main explanation | S1.1 B6 section |
| 2026-09-14_alpha_B6_H3_envelope_convention_audit.py | B6 H3 convention | S1.1 B6 section |
| 2026-09-14_alpha_B6_act_foot_k3_cascade_dual_protocol_rerun.py | activation-foot k3 cascade | S1.1 B6 section |

## 02_hERG_Lei_population_temperature (9) — Lei 211 population + five-temperature layer

Manuscript: main-text hERG population section (ref. 18/19), Extended Data Figs. 6–8; SI S2 population/temperature sections, S7.

| Script | Output | Manuscript anchor |
|---|---|---|
| lei211_J1J2_extraction.py | E_rev / h_ss / m_ss per-well extraction (211→670 wells) | S2 population section; ED7/ED8 a,b |
| lei211_J3_extraction.py | deactivation-ladder slow-component extraction | 0.74/3.03 s ladder sequence |
| lei211_TAUH_per_well.py | tau_rec per-well DoE fit (4-sigma noise gate) | S7 registry class-A source |
| lei211_TAUH_population_table.py | tau_rec population median table | ED6a; ED8c |
| lei211_J5_AP_forward.py | AP-clamp forward (0.5/1/2 Hz, A1–A3) | 91–99% amplitude; A1 45.8% |
| lei211_C1_synthetic_recovery.py | isochronous-bias synthetic-recovery control | S2 limitation (ii); S7 class C |
| lei211_time_table_build.py | five-temperature all-level median/CV/judgement time table | ED6 data source |
| lei211_temperature_summary.py | Q10 = 2.84 / R² = 0.991 summary | main-text temperature sentence; ED6 |
| lei211_population_plot.py | population figure | ED7 figure |

## 03_Nav1.5 (6)

Manuscript: Fig. 3a–b, ED3; SI S1.2, S2, S6.

| Script | Output | Manuscript anchor |
|---|---|---|
| 2026-09-15_Nalpha1_Nav15inact_fast_component_constancyverdict.py | tau_h(V) constancy 6/7; −30 mV CV 0.029 | S1.2a |
| 2026-09-15_Nalpha2_Nav15whole_channel_oneshot_validation.py | whole-trace forward 3/3 (<=3.4% RMS) | main-text Nav section; S2 |
| 2026-09-16_Nalpha3_Nav15act_foot_portability_verdict.py | foot spread CV 0.184 is real biology | main text; S6 Nα-3 |
| 2026-09-16_Nalpha4_single_channel_minus40_anchor_verdict.py | tau_h(−40) distribution (n=69, CV 0.772) -> N-arm closure | Fig. 3b; S1.2f |
| 2026-09-15_Nalpha_drug_Nav15drug_shape_modulation_verdict.py | two-component drug action (|kappa*|=0.20) | Fig. 4; S4 |
| 2026-09-15_Nalpha_drug_supplement_pack.py | external benchmarks 7/7 within the ±50% band | S4 |

## 04_CaV1.2 (6)

Manuscript: Fig. 3c, ED4; SI S1.3, S4, S6.

| Script | Output | Manuscript anchor |
|---|---|---|
| 2026-09-15_Calpha1_CaV12inact_temperature_law_carrier_verdict.py | f_inact family constant; sweep QC gates | S1.3a; S7 class A |
| 2026-09-15_Calpha1b_CaV12inact_charge_carrier_verdict_17mV.py | +17 mV anchor-segment recheck | S1.3a |
| 2026-09-15_Calpha2_Ca_microdomain_verdict_current_inact_corr.py | CDI not amplitude-graded (negative) | S1.3 equation section |
| 2026-09-15_Calpha3_CDI_structure_form_verdict.py | CDI additive fast component w_fast=0.289 | Fig. 3c |
| 2026-09-15_Calpha4_ramp_IV_temperature_law.py | Q10 not constant -> used as covariate | S1.3c |
| 2026-09-15_Calpha5_CaV12drug_shape_modulation_verdict.py | pure block rejected p=7.2e−16; two-pool opposing effects | Fig. 4; S4 |

## 05_IKs (6)

Manuscript: Fig. 3d, ED5; SI S1.4, S4.

| Script | Output | Manuscript anchor |
|---|---|---|
| 2026-09-15_Kalpha0_IKssmoke_GV_phenotype_cluster.py | G-V phenotype survey | archive |
| 2026-09-15_Kalpha1_IKsact_deact_constancy_verdict.py | tau_app rejected, magnitude anchored, spread real | S1.4d |
| 2026-09-15_Kalpha2_Fedida2024_Fig3B_digitised.py | tau_act(V) 25-point table (bell-shaped) | S1.4a; Fig. 3d |
| 2026-09-15_Kalpha2_Fedida2024_Fig3C_Deltatdigitised.py | delay table d(V) | S1.4b |
| 2026-09-15_Kalpha3_IKsforward_validation.py | activation forward 7/8 | S2 |
| 2026-09-15_Kalpha_drug_IKsdrug_table_decomposition_verdict.py | multi-table modulation; DIDS pool confound registered | Fig. 4; S4 |

## 06_identifiability_slowmemory (8) — Fisher audit + slow-memory spectrum

Manuscript: main-text Fisher section, Extended Data Fig. 2; SI S3.

| Script | Output | Manuscript anchor |
|---|---|---|
| de182.py / judge182.py (w182) | model62 13-parameter Fisher spectrum: chi=7.4e12, six null directions | Fig. ED2; S3 |
| 2026-09-13_rate_coupling_identifiability_math_validation.py | rate-conductance joint-scaling exact degeneracy | S3 exact degeneracy (i) |
| 2026-09-13_slow_memory_layer_identifiability_math_validation.py | alpha_amp·ALPHA·XMAX product degeneracy | S3 exact degeneracy (ii) |
| 2026-09-13_memory_kernel_identifiability_three_gate_validation.py | modal-coordinate projection | S3 |
| 2026-09-13_sameV_diff_history_coupling_verdictv4_nine_cells.py | same-voltage different-history 17–112x (9/9) | S3 slow-memory audit |
| 2026-09-13_slow_layer_discrete_const_verdictv5_noise_envelope_gate.py | memory spectrum 4.8 s–2000 s+ | S3 |
| 2026-09-13_slow_layer_master_curve_verdictP1.py | slow-layer master curve | S3 |

## 07_drug_grammar_CiPA (8) — drug grammar + CiPA 65/68

Manuscript: Fig. 2e, Fig. 4; SI S4.

| Script | Output | Manuscript anchor |
|---|---|---|
| 2026-09-14_CiPA_full_drug_library_downloader.py | CiPA three-lab 68-trace download | S4 data source |
| 2026-09-14_alpha_CiPA_crosslab_check_card.py | cross-lab comparison (20/20, 16/16, 29/32) | Fig. 2d |
| 2026-09-14_alpha_drug_card1_block_kinetics_state_preference.py | k_on/k_off; trapped state preference | S4 |
| 2026-09-14_alpha_drug_card2_pmz_non_single_step_binding_attribution.py | pimozide non-single-step binding | S4 |
| 2026-09-14_alpha_drug_card3_full_library_panorama_parameter_table.py | 136-dataset parameter panorama | S4 |
| 2026-09-14_alpha_drug_card4_AP_waveform_clinical_risk_crosscheck.py | AP-waveform risk crosscheck | S4 |
| 2026-09-14_alpha_drug_card5_shape_invariance_verdict.py | pure block rejected 30/30 | main-text hERG drug section |
| 2026-09-14_alpha_drug_card6_shape_deformation_decomposition.py | amplitude-type attribution (|kappa*−1|=0.020) | Fig. 2e |

## 08_StageB_ORd_host (7) — ORd host integration + three-arm drug scoring

Manuscript: Fig. 5; SI S5.

| Script | Output | Manuscript anchor |
|---|---|---|
| 2026-09-14_alpha_formal_assembly_forward_engine.py | alpha-model forward engine (shared by hosts) | S1/S5 |
| 2026-09-15_ORdthree_channel_alpha_assembly.py | IKr/IKs alpha-ised replacement inside ORd | Fig. 5a; J1–J6 |
| 2026-09-14_alpha_StageB_B0_ORd_port_foundation_recheck.py | port foundation recheck | S5 |
| 2026-09-14_alpha_StageB_B1_official_static_arm_24drugs.py | B1 arm rho=−0.523 | Fig. 5b |
| 2026-09-14_alpha_StageB_B2_alpha_substitution_24drugs.py | B2/B2s arms rho=−0.547/−0.506 | Fig. 5b |
| 2026-09-16_alpha_StageB_B3_dual_alpha_host_recheck_24drugs.py | dual-alpha recheck; tau_act sensitivity-arm range 0.016 | S5 robustness |
| 2026-09-16_B3_figure_redraw.py | Fig. 5 figure | Fig. 5 |

## 09_figures (7) — manuscript figure generation

| Script | Output |
|---|---|
| make_figures_v01.py | Fig. 1–5 + EDFig2 + EDFig3 (= manuscript ED7) PNG/PDF |
| make_EDFig6_temperature_Q10.py | EDFig6 temperature Q10 figure |
| make_percell_violin_2026-09-21.py | ED8 v01 six-panel per-cell violin figure (also generates the long-table CSV + failure registry) |
| make_percell_violin_v02_2026-09-21.py | ED8 v02: n<20 groups de-violinized to points only, all linear axes, IKs Chan anchor star, CaV literature anchor dashed line |
| make_percell_violin_v03_2026-09-21.py | ED8 v03: first nine-panel version (drug arms in figure), layout superseded by v04 |
| make_percell_violin_v04_2026-09-21.py | **ED8 v04 (current)**: unified box+strip style (median/IQR/10–90% whiskers + all scatter), per-row width by group count (f/i widened), wrapped annotations fixing the right-side blank; also idempotently generates `结果\per_cell_master_table_pharmacology_arms_2026-09-21.csv` (361 rows) |
| make_EDFig345_channel_cards_2026-09-21.py | **ED3/ED4/ED5 channel cards (current)**: Nav1.5/CaV1.2/IKs each 2x2, box+strip same style as ED8 v04; all numbers read directly from sealed JSONs / the master table (s_ref pool CV 0.184, w_fast 0.289 vs delta-f 0.293, forward 7/8, etc.); ED5d registers the convention difference between the per-file re-extraction (n=8) and the sealed J2 (n=5) |

## 10_verification_tau_rec (3) — tau_rec grid-quantization robustness checks (2026-09-21, post-seal verification layer)

| Script | Output |
|---|---|
| verify_tauh_continuous_refit_2026-09-21.py | continuous refinement v1 (archived mis-seeded version): record of the optimizer artifact with SSE worsening −3000%; the Q10≈0.9 artifact was discarded |
| verify_tauh_refit_v2_2026-09-21.py | **correctly seeded continuous refinement (current valid version)**: 1,340 fits, SSE improved only 1.7–4.1%, refined Q10 2.26 (−140) / 2.89 (−120) |
| verify_tauh_modelfree_2026-09-21.py | model-free clocks (t_half/t_ext/t10–90): Q10 2.1–3.8 (R²>=0.985) + grid-edge piling audit (37 °C −140 mV 55% censored) |

Conclusion in `结果\验证卡_taurec网格量化_连续精修复核_2026-09-21.md`: the sealed Q10=2.84 lies inside the refined/model-free bands and is not grid flattery.

---

## Related assets NOT in this package (pointers)

- Data-source download/curation scripts: `04_细胞线4\数据\Lei全量\download_lei*.py`, `_releakcorrect.py`; `04_细胞线4\公开数据\Nav15_Tarasov2026_dryad\*.py`.
- Manuscript version-generation scripts (v01→v02 anchor replacement): `文章\_make_v02_main.py`, `文章\_make_v02_si.py`.
- Early probe/debug scripts (`_debug_*`, `_probe_*`, `peek_*`, `scan_*`): left in `α模型\`; they are reconnaissance-layer, not production-layer.
