# Reproducibility record

Local-machine re-run of the full analysis chain, reconciled against the sealed
verdict values. All re-runs used the repository code as archived, on a single
Windows workstation (Anaconda + CUDA environment; managed Python 3.11 runtime;
20 worker processes where the script supports parallelism).

## StageB ORd host chain (re-run 2026-09-27/28, this machine)

| Step | Script | Key sealed value | Re-run value | Status |
|---|---|---|---|---|
| ORd three-channel alpha assembly | `2026-09-15_ORdthree_channel_alpha_assembly.py` | APD90 difference 0.36%; I_Kr peak ratio 2.06; J1/J2 pass | APD90 251.206 ms vs reference 252.114 ms (0.360%); I_Kr ratio 2.0592; J1 steady 0.005 ms; all three sensitivity arms pass | Reproduced |
| B2 alpha-substitution arm, 24 drugs | `2026-09-14_alpha_StageB_B2_alpha_substitution_24drugs.py` | dynamic rho -0.547, AUC 0.857; static rho -0.506, AUC 0.833 | dynamic rho -0.547 (p=5.67e-3), AUC 0.857; static rho -0.506, AUC 0.833; not inferior to B1 | Reproduced |
| B3 dual-alpha host recheck, 24 drugs | `2026-09-16_alpha_StageB_B3_dual_alpha_host_recheck_24drugs.py` | rho -0.555, AUC 0.881; qNet rho 0.998 | rho -0.555, AUC 0.881; qNet rho 0.998; per-drug median abs(delta qNet) 0.002; Kendall discordant pairs 2 | Reproduced |

| B1 official static arm, 24 drugs | `2026-09-14_alpha_StageB_B1_official_static_arm_24drugs.py` | rho -0.523, AUC 0.833 | rho -0.5232 (p=8.7e-3), AUC 0.8333; APD90 registry rho 0.630; 24/24 drugs, 0 EAD, 0 steady failures; full run 3.2 h, completed 2026-09-27 | Reproduced |

Registered notes carried by the re-run (unchanged from the sealed record):
- The V1 criterion (rho <= -0.60) is not met by any arm; this is the source of
  the manuscript statement that all arms share a 1xCmax static-protocol
  ceiling near -0.55. It is a registered result, not a run failure.
- The I_Na non-portability verdict stands (G x16 saturation arm, APD90 ~76 ms,
  no regenerative upstroke), consistent with the Nav1.5 model card section 3.4.

## Identifiability chain (re-run 2026-09-26, this machine)

| Script | Check | Result |
|---|---|---|
| `de182.py` (sanity arm) | Four-protocol digit-exact identity vs judge canonical R2 | pass (153 s) |
| `judge182.py` | Fisher / eigenvalue / Delta_s / R_ss recomputation vs frozen registry | recomputation complete; max relative diffs at 1e-6 to 1e-4 level against frozen criteria; verdict artifact `judge182_verdict.json` |

## Earlier chain links (re-run 2026-09-26/27, this machine)

Steps 1, 2, 7 and 8 of the eight-step chain (extraction, population anchors,
verification, figure regeneration) were re-run and reconciled in the same
campaign; per-step logs are retained outside the repository (project workspace,
`_rerun/logs_*`). No numerical conflict with the sealed values was found
anywhere in the campaign.
