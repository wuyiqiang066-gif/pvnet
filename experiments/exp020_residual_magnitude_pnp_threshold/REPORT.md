# EXP020: Keypoint Residual Magnitude vs PnP Stability Threshold Diagnosis

Branch: `exp020_residual_magnitude_pnp_threshold` (created from `exp019_residual_geometry_dose_response` @ `31b9a5d`)
Mode: **pure mechanism diagnosis** (synthetic isotropic residual magnitude sweep from GT keypoints; the ONLY change vs frozen baseline inputs is the 2D keypoint coordinates fed to the UNMODIFIED original PnP).
Cost: CPU-only, seed=0, no network forward, no GPU, no training; full run 2.6 s + analysis < 1 s (budget < 5 min met; no solver modification was ever needed).

## 1. Research Question

EXP017–019 established: D1 improves keypoint localization yet increases ~180° PnP flips (4/20 → 14/20); the dominant observable mechanism is reprojection-objective ambiguity; and D1-specific residual GEOMETRY received no causal support — matched-RMS isotropic zero-shift noise alone reproduced most of the real D1 flip burst (geometry mining formally CLOSED). EXP020 asks the one remaining basic question:

> Does the original PVNet PnP exhibit a keypoint residual MAGNITUDE → pose-ambiguity THRESHOLD / phase-transition regime, or does flip probability degrade smoothly with residual magnitude?

Not a claim of a universal threshold — only a diagnosis of whether a clear nonlinear transition (knee) exists on the fixed 20-image OCC set.

## 2. Hypothesis (pre-registered)

Pre-registered in `config.json` (`decision_rules_pre_registered`), written BEFORE any synthetic PnP call, never modified:

- **H1 / GO-A (transition)**: CLEAN18 flip150 vs ACTUAL RMS shows a qualifying nonlinear jump — ALL of (i) jump_max ≥ 0.20 within one adjacent step; (ii) delta_J sync: median delta_J non-decreasing across ≥ 9/11 adjacent steps AND ΔP(delta_J>0) across the transition pair ≥ +0.10; (iii) not 1–2 image driven (≥ 3 CLEAN18 images first-flip inside the transition pair; SOLVER_DIVERGENCE events not counted as drivers); (iv) real D1 median per-image RMS falls within [lower σ − 0.5, upper σ + 0.5] of the transition pair.
- **H0 / GO-B (smooth degradation)**: dose-response exists (flip150(σ_max) ≥ flip150(σ_min) + 0.30) but GO-A criteria fail → NO threshold wording.
- **GO-C (magnitude insufficient)**: otherwise → STOP, no geometry mining restart.
- Sign convention (EXP018-corrected, unchanged): `delta_J = J_gt − J_pred`; `delta_J > 0` ⇔ the returned pose fits the input keypoints strictly better than the correct pose.

## 3. Experimental Isolation

- **Data**: exact 20 OCC held-out images asserted identical to EXP019 `image_ids.json` (EXP017/018 cross-checked). Nothing else touched.
- **Inputs**: GT keypoints/poses and D0/D1 voted keypoints loaded bit-identically from EXP016's persisted `raw_predictions/raw_occ_D{0,1}.npz`.
- **Solver**: original `Evaluator.evaluate → pnp() → cv2.solvePnP(SOLVEPNP_ITERATIVE)`, original `linemod` intrinsics, original `VotingType.Farthest` 9 3D points. UNMODIFIED (no `solvePnPGeneric`, no flag/threshold/termination change, keypoint count unchanged).
- **GEOMETRY CLOSED (EXP019)**: residuals are isotropic Gaussian, zero coherent shift by construction; anisotropy/covariance shape NOT studied. Sample means are NOT adjusted (protocol section 7).
- **Oracle**: original `Projector.project_K` fits J_pred/J_gt on the same 9 3D points the solver consumed. A pre-existing `arccos` RuntimeWarning inside the ORIGINAL `evaluation_utils.py` (line 140) fires on divergent real D0/D1 poses — library behavior, no code modified.
- CPU-only, seed=0, no training, 20 images, cat only.

## 4. Exact Image IDs

`color_00024.png` … `color_00043.png` (20 OCC images; internal `image_id` 20–39), asserted == EXP019 == EXP017 == EXP018. Full list in `image_ids.json`. CLEAN18 (all except `color_00026/27`, fixed from EXP018/019 before any EXP020 result) is a reporting subset; nothing is deleted.

## 5. Baseline Reproduction

`results/reproduction_check.json`: `gate_ok = true` (run 0.9 s, before any synthetic condition).

| condition | flips >90° | verification |
|---|---|---|
| C0 GT keypoints | **0/20** | max rot 0.0000°, max official reproj 0.0001 px |
| C1 D0 real | **4/20** | flips/rot/reproj/trans exact vs EXP017; poses bit-match EXP016 (<1e-9); oracle J matches EXP018 (<1e-4) |
| C2 D1 real | **14/20** | same bit-level verifications |

Additionally `SIG_S000` (σ = 0) is asserted per-image identical to C0 (rot and J_pred < 1e-9). Any mismatch would have STOPPED the experiment.

## 6. Synthetic Residual Construction

Per image: `kp_i = GT_kp_i + r_i`, with `r_i(σ) = σ · z_i`, `z_i ~ N(0, I)`.
- ONE base realization `z` (shape (20, 9, 2), `numpy default_rng(0)`, checksum −11.621783) generated ONCE; every level is the deterministic transform `r = σ·z` of the SAME z (no per-σ re-sampling). σ is the only experimental variable.
- NO mean removal, NO normalization, NO covariance shaping.
- Only PnP input keypoint coordinates change; images/masks/vertices/voting/poses untouched.

## 7. Magnitude Levels

12 pre-registered levels (px): 0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00. Nominal σ ≠ actual sampled magnitude: base z per-image RMS spans 0.928–1.891 (median 1.3965), so actual per-image RMS = σ × (per-image z-RMS). ALL dose-response analysis uses ACTUAL RMS (per-condition mean over images; pooled variant also recorded). Per condition recorded: RMS, mean_norm, median_norm, p90_norm, coherent shift (≈0 confirmed; recorded, NOT studied).

## 8. Objective Definition

`J(pose) = mean_i ||project_K(kp3d_Farthest9, pose)_i − kp_input_i||` (px), original Projector, linemod intrinsics. `J_pred` = fit of solver-returned pose; `J_gt` = fit of GT pose; `delta_J = J_gt − J_pred` (EXP018 sign, not re-inverted): `delta_J > 0` ⇔ returned pose out-fits the correct pose on the input keypoints (objective ambiguity signal).

## 9. ALL20 Results

`results/dose_response.csv` (primary columns; full table in file):

| σ (nominal) | actual RMS | flip150 | flip170 | P(ΔJ>0)¹ | med ΔJ¹ | div |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | −0.000 | 0 |
| 0.25 | 0.351 | 0.00 | 0.00 | 1.00 | +0.052 | 0 |
| 0.50 | 0.703 | 0.00 | 0.00 | 1.00 | +0.103 | 0 |
| 0.75 | 1.054 | 0.15 | 0.15 | 0.85 | +0.130 | 0 |
| 1.00 | 1.406 | 0.40 | 0.35 | 0.60 | +0.167 | 3 |
| 1.25 | 1.757 | 0.50 | 0.50 | 0.70 | +0.194 | 1 |
| 1.50 | 2.109 | 0.55 | 0.50 | 0.65 | +0.238 | 2 |
| 1.75 | 2.460 | 0.60 | 0.55 | 0.75 | +0.311 | 2 |
| 2.00 | 2.812 | 0.60 | 0.60 | 0.80 | +0.365 | 1 |
| 2.50 | 3.514 | 0.60 | 0.60 | 0.90 | +0.465 | 1 |
| 3.00 | 4.217 | 0.65 | 0.65 | 0.95 | +0.561 | 1 |
| 4.00 | 5.623 | 0.65 | 0.60 | 0.90 | +0.743 | 2 |

¹ P(ΔJ>0) and med ΔJ on CLEAN18 (the primary subset); ALL20 variants in `dose_response.csv`.

Real controls on ALL20: C1_D0 flip150 0.10, C2_D1 flip150 0.60.

## 10. CLEAN18 Results

flip150: 0.00, 0.00, 0.00, **0.11, 0.39, 0.50, 0.56**, 0.56, 0.56, 0.56, 0.61, 0.61 (σ = 0 → 4.0). flip170 follows within 0.05 at every level. Median ΔJ rises monotonically +0.052 → +0.743 (11/11 non-decreasing steps). P(ΔJ>0) is NON-monotone: trivially 1.00 at σ ≤ 0.5 (in the correct basin the solver slightly out-fits GT — first_sigma_deltaJ_positive = 0.25 for ALL 20 images), dips to 0.61 at the transition (SF-type flips mixed in), then rises to 0.78–0.94 in the ambiguity regime (wrong pose out-fits GT).

## 11. Flip Dose-Response

Adjacent Δflip150 (CLEAN18): [0, 0, **+0.111, +0.278, +0.111**, +0.056, 0, 0, 0, +0.056, 0].
- jump_max = **+0.278** at the nominal pair [0.75, 1.0] px (actual RMS 1.054 → 1.406) — 2.5× the next-largest step.
- ALL20 shows the same structure: jump_max +0.25 at the SAME pair [0.75, 1.0] — the pattern is not a CLEAN18 artifact.
- Shape: flat (≤0.5 px) → steep rise (0.75–1.25 px) → plateau/saturation (~0.56–0.61 for RMS 2.1–5.6 px). Descriptively knee-LIKE, but see §16: the pre-registered transition criteria fail on the objective-sync and placement legs, so the conservative verdict is smooth-degradation wording.

## 12. Delta-J Dose-Response

Median ΔJ (CLEAN18) increases monotonically at every one of the 11 adjacent steps (finite-difference monotonicity 11/11): the objective's preference for the returned pose over GT grows continuously with magnitude — including in the pre-transition flat-flip region. P(ΔJ>0) is U-shaped-truncated (1.00 → 0.61 → 0.94): the dip is exactly at the steep flip segment (mixed SF/RA flips), the recovery is the ambiguity regime where the WRONG pose out-fits GT. ΔP_RA across the max-jump pair is **−0.278** (a DROP, not the pre-registered ≥ +0.10 rise) — the objective-ambiguity signal does not synchronize with the flip jump in the direction the GO-A criterion required.

## 13. Per-image Stability Margins

`results/per_image_transition.csv`; first flip150 σ per CLEAN18 image (no interpolation; NA if never):

| margin class | n | images |
|---|---|---|
| early (≤ 1.0 px) | 7 | 00032, 00034, 00036, 00038, 00039, 00040, 00043 |
| middle (1.25–2.5 px) | 3 | 00024, 00028, 00042 |
| late (> 2.5 px) | 1 | 00025 (first flip at σ = 3.0) |
| never (NA at σ ≤ 4.0) | 7 | 00029, 00030, 00031, 00033, 00035, 00037, 00041 |

First-flip σ distribution (all 20): 0.75×3, 1.0×5, 1.25×2, 1.5×1, 1.75×1, 3.0×1, NA×7. Stability margins are strongly image-dependent — spanning ≤ 1 px to > 4 px — which itself is a primary result: there is no universal pixel-error threshold; there is an image-dependent pose-stability margin.

## 14. D0/D1 Placement on Dose-Response

Real median per-image RMS anchors (numpy-only, pre-PnP; D1 asserted == EXP019's frozen anchor):

- **D1: 2.423 px** → nearest synthetic level σ = 2.50 (actual RMS 3.514): synthetic CLEAN18 flip150 = 0.556 vs real C2 flip150 = 0.611 (clean 0.611). **Real D1 flip behavior sits ON the synthetic isotropic curve at its own median magnitude — a strong bridge:** the real D1 flip burst is what matched-magnitude isotropic noise predicts.
- **D0: 3.422 px** → nearest level σ = 3.00 (actual RMS 4.217): synthetic flip150 = 0.611 vs real C1 flip150 = 0.10 (clean 0.111). **Real D0 flips FAR BELOW the synthetic curve at equal-or-larger magnitude.**
- Consequence (protocol section 22, recorded not mined): magnitude is a major driver but NOT the sole determinant — D0's real residual field (median RMS 3.42 px, anisotropy 4.30, non-zero coherent shift) flips less than matched-magnitude isotropic noise, while D1's field flips as predicted. Note D0 RMS > D1 RMS yet D0 flips less — the dose-response curve is family-dependent for REAL residual fields, even though synthetic isotropic noise at D1's magnitude reproduces D1's flips.
- Placement vs transition window: D1 RMS 2.423 ∉ [0.25, 1.5] (the max-jump pair ± 0.5) — GO-A criterion (iv) FAILS.

## 15. Pathological Cases

- `color_00026` (kept in ALL20): flips from σ = 1.75; at σ = 1.75 the row is flagged **SOLVER_DIVERGENCE** (rot 166.8°, J_pred/reproj ≫ 100 px) and is NOT read as a normal phase transition; σ ≥ 2.0 flips are ordinary ~176–178° ambiguity. Under CLEAN18 it is excluded throughout.
- `color_00027` (kept in ALL20): flips from σ = 0.75 onward, rot ≈ 178–180° at every level (an extremely low-margin image; consistent with its EXP018/019 **CATASTROPHIC_KEYPOINT** history). Excluded from CLEAN18.
- Neither image drives the CLEAN18 transition (both excluded there); the ALL20 jump occurs at the SAME pair [0.75, 1.0], so the main reading does not hinge on either.
- Divergence counts per synthetic level (ALL20): 0/0/0/0/3/1/2/2/1/1/1/2; PnP exceptions 0 everywhere.

## 16. Decision: GO-B (smooth degradation; NO threshold wording)

`results/decision.json`, mapped against the pre-registered rules:

| GO-A criterion | value | met? |
|---|---|---|
| (i) jump_max ≥ 0.20 (CLEAN18) | +0.278 @ [0.75, 1.0] | YES |
| (ii-a) median ΔJ monotone ≥ 9/11 steps | 11/11 | YES |
| (ii-b) ΔP(ΔJ>0) across pair ≥ +0.10 | −0.278 | **NO** |
| (iii) ≥ 3 drivers first-flip in pair | 5 (00032/36/38/39/40; +2 divergence-flagged not counted) | YES |
| (iv) D1 RMS in [σ_lo − 0.5, σ_hi + 0.5] | 2.423 ∉ [0.25, 1.5] | **NO** |

GO-A fails (criteria ii-b and iv). Overall dose-response exists: flip150 rises +0.61 from σ = 0 to σ = 4.0 (≥ 0.30) → **GO-B** per the pre-registered branch.

> **Conclusion (conservative, per pre-registration):** PnP stability degrades continuously with keypoint residual magnitude on this 20-image OCC set, WITHOUT evidence qualifying as a sharp threshold under the pre-registered criteria. The flip curve does contain a steep segment (nominal σ 0.75→1.25, actual RMS ~1.0→1.8, after which it saturates at ~0.6), and 5 CLEAN18 images first-flip inside the steepest step — but the objective-side signal does NOT synchronize with that jump as registered (P(ΔJ>0) drops across it), and the real D1 magnitude does not sit at the jump. The honest summary: a descriptive knee-like rise exists, but the pre-registered threshold test is failed on 2 of 5 criteria — do not use "phase transition / threshold" wording.

Statistical discipline (protocol section 26): n = 20; no significance claims; no universal / object-independent / dataset-independent / solver-independent threshold wording; diagnostic evidence on the fixed 20-image OCC set only.

## 17. What This Experiment Establishes

1. With geometry closed (isotropic, zero shift, shared realization, σ the only variable), flip150 rises from 0 (σ ≤ 0.5 px) to ~0.6 (σ ≥ 1.75 px) and saturates — residual magnitude is a REAL and dominant driver of PnP instability on this set.
2. The objective signal (median ΔJ) grows monotonically and continuously with magnitude (11/11 steps) — the returned pose increasingly out-fits GT regardless of flip status.
3. Per-image stability margins are strongly heterogeneous (≤ 1 px for 7 images, > 2.5 px for 1, never within 4 px for 7) — an image-dependent pose-stability margin, not a universal pixel threshold.
4. Real D1's flip burst (0.611 clean) sits ON the synthetic isotropic curve at its own median magnitude (2.42 px → synthetic 0.556): matched-magnitude isotropic noise predicts D1's flips.
5. Real D0 flips far below the synthetic curve at equal-or-larger median magnitude (0.111 vs 0.611) — magnitude alone does not explain the D0/D1 flip contrast (recorded per protocol section 22; geometry mining stays CLOSED).

## 18. What This Experiment Does NOT Establish

1. A sharp threshold / phase transition — the pre-registered GO-A test failed on the objective-sync and placement criteria; only descriptive knee-LIKE shape was observed.
2. Any universal, object-independent, dataset-independent, or solver-independent stability boundary.
3. An explanation of why real D0 (larger median RMS) flips less than matched-magnitude isotropic noise — identifying which property of real residual fields suppresses flips would require studying statistics other than magnitude, which EXP019 closed for D1-like geometry; this remains OPEN and is deliberately not mined here.
4. Anything about the solver's internal basin selection (EXP018's candidate-access limitation stands).
5. Statistical significance of any jump (n = 20, single shared realization; the σ-sweep reuses ONE noise field, so level-to-level differences are perfectly correlated by design — appropriate for dose-response shape, not for variance claims).

## 19. Next Experiment Recommendation

Per the pre-registered GO-B: no threshold claim; no further magnitude mining at finer granularity (the curve has saturated; denser σ would not change the verdict). Recommended successors, each with its own pre-registration:

1. **EXP021 (mechanism, highest value):** a NEW diagnostic branch using multi-hypothesis PnP (`solvePnPGeneric` — allowed ONLY there, never in the baseline) to expose the candidate landscape and measure, per image, the objective gap between the correct and the ~180° basin as a function of σ. This directly tests the basin-competition reading of the observed margin heterogeneity (why 00029–00041 never flip while 00032–00043 flip below 1 px).
2. **EXP022 (bridge completion):** explain the D0-vs-synthetic-isotropic gap (same median magnitude, 0.111 vs 0.611 flip rate) by testing REAL-D0 residual surrogates (e.g., rescaled real D0 fields preserving their spatial pattern) against matched isotropic noise — a magnitude-held-constant, pattern-varied contrast on REAL fields. NOTE: this revisits field structure that EXP019 closed for D1-LIKE geometry; it is justified here only because EXP020 exposed a real-field-specific suppression that magnitude cannot explain, and it must be pre-registered as a D0-pattern study, not a D1-geometry revival.
3. Practical downstream check (no new science): an input-jitter stability probe (re-run PnP under ±1 px jitter; flag disagreement) would catch most flips for the early-margin (≤ 1 px) images identified here — a deployment-relevant use of the per-image margin table.
