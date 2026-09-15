# EXP019: Residual-Geometry Dose–Response

Branch: `exp019_residual_geometry_dose_response` (created from `exp018_pose_hypothesis_oracle` @ `1a5ff8b`)
Mode: **pure mechanism diagnosis** (synthetic keypoint residual fields; the ONLY change vs the frozen baseline inputs is the 2D keypoint coordinates fed to the UNMODIFIED original PnP).
Cost: CPU-only, no network forward, no GPU, seed=0, no training; full run 2.4 s + analysis < 1 s.

## 1. Research Question

EXP017/018 established: D1 lowers OCC keypoint median error (3.150 → 2.143 px) yet the ORIGINAL standard PnP flips ~180° on 14/20 held-out images vs 4/20 for D0; and in the D1 flip subset the dominant phenomenon is that the WRONG pose fits the input keypoints better than the GT pose under the reprojection objective. EXP019 asks the stricter causal question:

> **Is D1-type keypoint residual GEOMETRY itself (anisotropy / coherent shift) causally sufficient to push the original PVNet reprojection objective into the ~180° ambiguity regime — at matched residual MAGNITUDE?**

Not "does D1 have larger error → more flips", but: with residual RMS held comparable, does changing only the residual geometry systematically change (i) `delta_J = J_gt − J_pred` and (ii) PnP flip probability? Only if yes can EXP017/018's association evidence be upgraded to causal dose–response evidence.

No solver, flag, threshold, seed, GT, intrinsics, 3D points, image set, voting, or architecture modification of any kind.

## 2. Causal Hypothesis (pre-registered)

Pre-registered in `config.json` (`decision_rules_pre_registered`), written BEFORE any synthetic PnP call, never modified:

- **GO-A (geometry-causal)** — at matched magnitude, D1-like geometry vs matched isotropic geometry shows (i) a stable flip150 gap `g_k = flip150(D1G,k) − flip150(ISO,k)`, all g_k ≥ 0 with Σg ≥ 3 (D1-promoting) or all g_k ≤ 0 with −Σg ≥ 3 (isotropy-promoting; symmetric form, direction reported); (ii) RA-type objective evidence `RA := flip150 ∧ delta_J > 0` ordered in the promoted direction at ≥ 2/3 scales; (iii) cross-scale punchline `flip150(promoted @1.0×) ≥ flip150(other @1.5×)`.
- **GO-B (magnitude-only)** — |g_k| ≤ 1 at every scale AND flip150 rises ≥ 3 from 0.5× to 1.5× within ≥ 1 geometry family → STOP, no geometry mining.
- **GO-C (mixed/inconclusive)** — otherwise; conservative wording only.
- Primary statistic: flip at rot > 150° on ALL 20; CLEAN 18 (excluding the two EXP018-confirmed pathological images `color_00026/27`, fixed BEFORE any EXP019 result) reported as required support; decision invalidated to GO-C if the CLEAN 18 gap sign contradicts ALL 20.
- Sign convention (EXP018-corrected, never inverted here): `delta_J = J_gt − J_pred`; `delta_J > 0` ⇔ the returned (wrong) pose fits the input keypoints strictly better than the correct pose (REPROJECTION_AMBIGUITY direction).

## 3. Pre-registered Config & Anti-circularity

`config.json` written at experiment start (before any synthetic PnP) and never modified. Anti-circularity protocol (section 22):

1. D1-like geometry targets were computed from the EXP016 raw D1 residuals with numpy ONLY (no PnP), then cross-checked against EXP017's `residual_geometry.csv` medians (assert < 1e-3), THEN written to `config.json`, THEN any synthetic PnP ran:
   - `anisotropy_median = 3.5666`, `coherent_shift_median = 1.2445 px`, `rms_anchor_median_per_image = 2.4235 px`, `meannorm_anchor = 2.1319 px`, unit direction `u = (−0.9991, 0.0421)` (mean of per-image residual mean vectors over the fixed CLEAN 18).
2. Anchor correction happened at this stage, BEFORE any synthetic result existed: the first draft used POOLED RMS/mean-vector anchors; pooled RMS = 139.78 px is dominated by `color_00025/26/27` and was REJECTED in favor of median per-image statistics (documented in `config.json: anchor_correction_note`).
3. One shared random realization: `z ~ N(0, I)` of shape (20, 9, 2), `numpy default_rng(0)`, generated ONCE after config write; EVERY synthetic condition is a deterministic transform of the SAME z (no per-condition re-sampling).
4. Normalization rule (per image): `w = A·z; w ← w − mean(w); mu = shift·scale·SHIFT_TARGET·u; s = sqrt(T² − ||mu||²)/rms(w); r = s·w + mu; T = scale·D1_RMS` (assert T > ||mu||); `kp = GT + r`. Achieved per-image RMS is EXACT by construction; coherent shift and anisotropy exact up to sample noise; `mean_norm` reported but not forced (geometry-coupled; RMS is the pre-registered magnitude statistic).
5. Only images' PnP INPUT keypoint coordinates change. Images, masks, vertices, poses, GT: untouched.

## 4. Exact Image IDs

`color_00024.png` … `color_00043.png` (20 OCC held-out images; internal `image_id` 20–39), asserted identical to EXP017/018's `image_ids.json`. Full list in `image_ids.json`. No image added or removed; the CLEAN 18 subset (all except `color_00026/27`) is a fixed reporting subset, never a deletion.

## 5. Experimental Isolation

- Keypoints/poses loaded bit-identically from EXP016's persisted `raw_predictions/raw_occ_D{0,1}.npz`; GT keypoints and poses identical across all conditions.
- Solver: original `Evaluator.evaluate → pnp() → cv2.solvePnP(SOLVEPNP_ITERATIVE)`, original `linemod` intrinsics, original `VotingType.Farthest` 9 3D points. UNMODIFIED (no `solvePnPGeneric`, no RANSAC on pose, no flag/threshold/termination change).
- Objective oracle uses the original `Projector.project_K`: `J(pose) = mean_i ||project_K(kp3d, pose)_i − kp_input_i||` (px) on the same 9 3D points the solver consumed.
- CPU-only, seed=0, no training, 20 images, cat only.

## 6. Baseline Gate (must pass before any synthetic run)

`results/reproduction_check.json`: `gate_ok = true`.

| condition | flips >90° | verification |
|---|---|---|
| C0 GT keypoints | **0/20** | rot err < 1e-3°, official reproj < 1e-2 px on all 20 |
| C1 D0 real | **4/20** | exact vs EXP017; poses bit-match EXP016 npz (< 1e-9); J matches EXP018 oracle (< 1e-4) |
| C2 D1 real | **14/20** | exact vs EXP017; same bit-level verifications |

Any failure would have STOPPED the experiment. The gate passed on the first run.

## 7. Condition Set

9-condition pre-registered grid + 2-condition mechanism check (protocol sections 10–11), plus 3 real/GT controls — 11 conditions × 20 images = 220 rows (`results/per_image.csv`):

- Controls: `C0_gt` (r = 0), `C1_D0_real`, `C2_D1_real`.
- Grid: {ISOTROPIC, D1-LIKE} × {0.5×, 1.0×, 1.5×} = `ISO_S05/S10/S15`, `D1G_S05/S10/S15`; 1.0× matches the real D1 median per-image RMS (2.4235 px).
- Mechanism 2×2 at 1.0× (run AFTER the grid conditions within the same script; axes reuse the frozen D1-like targets, no result-driven redefinition): `ISO_S10_SH` (isotropic + D1-like shift), `D1G_S10_NS` (D1-like anisotropy, zero shift).
- Condition aliases: C3 (isotropic matched) = `ISO_S10`; C4 (D1-like geometry) = `D1G_S10`.

## 8. Matched-Magnitude Audit

`results/matched_magnitude_check.json`:

- Per-image RMS == scale × 2.4235 px EXACTLY: max |achieved − target| = **8.88e-16 px** (machine epsilon). The magnitude confound is closed.
- `mean_norm/RMS`: ISO 0.8974, D1G 0.8666, real D1 0.8797 — all close; mean_norm not forced (it is geometry-coupled; RMS is the registered magnitude statistic).
- Achieved sample anisotropy: ISO 2.669 (finite-sample eigenvalue-ratio bias of n=9 2D samples under a population-isotropic generator), D1G 4.114 (target 3.5666). Sample-level geometry difference between the two families is real and in the designed direction.

## 9. Condition Summary (protocol section 14 table)

`results/condition_summary.csv`. ALL 20 primary; CLEAN 18 in last column (flip150). RA = flip150 ∧ delta_J > 0 (ambiguity-type); SF = flip150 ∧ delta_J ≤ 0 (selection-failure/divergence-type). Real-condition RMS is pooled over images (dominated by the catastrophic images; the real D1 MEDIAN per-image RMS is 2.4235 px — the anchor).

| condition | RMS | mnorm | aniso | f90 | f120 | f150 | f170 | RA | SF | medΔJ | P(ΔJ>0) | f150 clean18 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_gt | 0.000 | 0.000 | – | 0 | 0 | 0 | 0 | 0 | 0 | −0.000 | 0.00 | 0 |
| C1_D0_real | 47.96 | 47.41 | 4.30 | 4 | 3 | 2 | 2 | 2 | 0 | 2.519 | 0.95 | 2 |
| C2_D1_real | 47.74 | 47.14 | 3.57 | 14 | 13 | 12 | 10 | 9 | 3 | 0.996 | 0.80 | 11 |
| ISO_S05 | 1.212 | 1.087 | 2.67 | 5 | 5 | 4 | 3 | 0 | 4 | 0.079 | 0.75 | 3 |
| ISO_S10 | 2.424 | 2.175 | 2.67 | 13 | 13 | 13 | 12 | 6 | 7 | 0.144 | 0.65 | 11 |
| ISO_S15 | 3.635 | 3.262 | 2.67 | 13 | 13 | 13 | 13 | 10 | 3 | 0.322 | 0.85 | 11 |
| D1G_S05 | 1.212 | 1.050 | 4.11 | 1 | 1 | 1 | 1 | 0 | 1 | 0.225 | 0.95 | 1 |
| D1G_S10 | 2.424 | 2.100 | 4.11 | 11 | 11 | 11 | 9 | 6 | 5 | 0.401 | 0.75 | 10 |
| D1G_S15 | 3.635 | 3.150 | 4.11 | 12 | 12 | 12 | 12 | 10 | 2 | 0.716 | 0.90 | 11 |
| ISO_S10_SH | 2.424 | 2.145 | 2.67 | 11 | 11 | 11 | 11 | 8 | 3 | 0.446 | 0.85 | 10 |
| D1G_S10_NS | 2.424 | 2.143 | 4.11 | 11 | 11 | 11 | 11 | 7 | 4 | 0.180 | 0.80 | 10 |

## 10. Dose–Response Result

**Magnitude axis (within-family, same geometry, scale 0.5× → 1.5×):** strong, monotone dose–response. flip150 (ALL 20): ISO 4 → 13 → 13; D1G 1 → 11 → 12. Median delta_J rises monotonically in both families (ISO 0.079 → 0.144 → 0.322; D1G 0.225 → 0.401 → 0.716). Residual magnitude is a real, dominant driver of both flips and the objective's preference for the wrong basin.

**Geometry axis (across-family, matched RMS):** small, and NOT in the hypothesized D1-promoting direction. g = flip150(D1G) − flip150(ISO) = **{−3, −2, −1}** at {0.5×, 1.0×, 1.5×} (Σ = −6; isotropy-promoting separation). RA counts are IDENTICAL across families at every scale (D1G, ISO) = (0,0), (6,6), (10,10) — RA ordering wins 0/3. CLEAN 18 gaps: {−2, −1, 0} (Σ = −3), same sign as ALL 20 → consistent, no subset contradiction.

**The single most informative comparison:** matched-magnitude ISOTROPIC noise with ZERO coherent shift (`ISO_S10`, per-image RMS 2.4235 px) reproduces the real D1 flip burst almost exactly on its own: **13/20 flips (11/18 clean) vs real D1's 14/20 (12/18 clean)**. D1-like geometry at the same magnitude gives 11/20 (10/18). The real D1 flip burst therefore does NOT require D1-like residual geometry.

**Directional note (honesty over narrative):** the observed iso-promoting separation is directionally CONSISTENT with EXP017's within-D1 association (flip images had LOWER anisotropy and SMALLER coherent shift). But at n = 20 with max gap 3/20 and identical RA ordering, this cannot be elevated to a causal claim under the pre-registered rules.

## 11. Mechanism Check (2×2 anisotropy × shift, 1.0×; run after the grid)

| | shift = 0 | shift = D1-like |
|---|---|---|
| **isotropic** | 13 | 11 |
| **D1-like** | 11 | 11 |

Neither anisotropy nor coherent shift PROMOTES flips at matched magnitude; each single change slightly reduces flips (13 → 11) and the combination stays at 11. No interaction evidence. Anisotropy and shift are not confounded with flip promotion in either direction.

## 12. Per-image Analysis (protocol section 19; classification only — NO image removed)

Classes (an image can appear in more than one): geometry-sensitive = flips under D1G where ISO does not at some scale (n=1); geometry-insensitive = no flip in any grid condition (n=7); contradictory = flips under ISO where D1G does not at some scale (n=6).

- **Geometry-sensitive (1):** `color_00034` — both geometries flip at 1.0×/1.5×, but delta_J sign differs (ISO_S10 −0.07 vs D1G_S10 +0.07): the only image where geometry changes the objective's verdict at matched flip.
- **Geometry-insensitive (7):** `color_00029, 00030, 00031, 00033, 00035, 00037, 00041` — correct under D1G at all scales (e.g. 00029: rot 4.7°/7.1° at 1.0×/1.5×, delta_J ≈ 0.7/1.0 > 0 without flipping).
- **Contradictory (6):** `color_00025, 00026, 00027, 00032, 00038, 00042` — ISO flips where D1G does not (e.g. 00025: ISO_S10 180° vs D1G_S10 1.9°).

Forced-focus images (protocol section 19; all retained in every statistic):

- `color_00033`: never flips under ANY synthetic condition (rot ≤ 5°); delta_J grows with scale (0.11 → 0.33) but the solver stays in the correct basin. Stable-image archetype.
- `color_00039`: flips from 1.0× in BOTH geometries with dose-growing positive delta_J (0.30 → 0.72 ISO; 0.72 → 1.17 D1G) — cleanest magnitude-driven ambiguity archetype.
- `color_00042`: ISO_S05 divergence (J_pred 2033 px → SOLVER_DIVERGENCE), flips at higher scales in both families; D1G_S10 also divergent (1134 px). Mixed, instability-prone.
- `color_00026`: ISO_S10 divergence (J_pred 1536 px); under D1G it NEVER flips (rot 3.8°/7.7°/11.6° at 0.5/1.0/1.5×) — the synthetic residual actually REPLACES the real D1 catastrophic failure on this image (see §13).
- `color_00027`: ISO flips at ALL scales including 0.5× (rot 180°, delta_J −0.10 SF-type) — flips under nearly any perturbation; D1G flips at 1.0×/1.5× only.

## 13. Pathological Images and Subsets (protocol section 20)

- `color_00026` — real D1 keypoints are catastrophic (input kp median error ≫ 100 px). Under synthetic residuals at matched magnitude the image RECovers (never flips under D1G; one ISO divergence at 1.0×). The ISO_S10 divergence (J_pred = 1536 px ≫ 100 px) is flagged **SOLVER_DIVERGENCE** and is NOT counted as reprojection ambiguity (it lands in SF, delta_J ≈ −1534).
- `color_00027` — real D1 catastrophic keypoint baseline; flagged **CATASTROPHIC_KEYPOINT** on real-residual conditions. Under synthetic conditions it flips under almost any perturbation (ISO even at 0.5×).
- Both images stay in ALL 20 for every statistic; CLEAN 18 (all except 26/27, fixed from EXP018 before any EXP019 result) is reported alongside. Solver divergence counts per condition (ALL 20): C1 2, C2 3, ISO_S05 2, ISO_S10 2, ISO_S15 1, D1G_S05 0, D1G_S10 3, D1G_S15 2, ISO_S10_SH 1, D1G_S10_NS 2; PnP exceptions 0 everywhere.

## 14. Deviations from Protocol (documented, none result-driven)

1. **Anchor correction** (pre-synthetic): pooled RMS/mean anchors (pooled RMS 139.78 px, dominated by catastrophic images) rejected in favor of median per-image anchors BEFORE any synthetic PnP ran; recorded in `config.json: anchor_correction_note`.
2. **Analyzer CLEAN 18 sign fix** (post-run, pre-interpretation): the first analysis pass computed the CLEAN 18 consistency gap with a direction-weighted sign, printing "CLEAN 18 gap +3" (apparently contradicting ALL 20). Corrected to the raw D1G−ISO sum (−3, consistent with the ALL 20 direction) BEFORE any interpretation or commit. The DECISION IS INVARIANT under both versions (GO-C either way, because RA ordering wins 0/3); only the rationale wording changed. The first-run decision artifact was regenerated, not reported.
3. **Mechanism 2×2 timing**: registered in `config.json` before the run and executed after the 9 grid conditions within the same script (protocol section 11 allows it only after the main grid; no grid result influenced its parameters — both axes reuse the frozen D1-like targets).
4. **MODERATE anisotropy level** was reserved in `config.json` and NOT run in round 1, per the protocol's 9-condition grid spec (section 10 lists only isotropic and D1-like families).

## 15. Decision: GO-C (mixed / inconclusive; direction observed: isotropy-promoting)

`results/decision.json`, mapped against the pre-registered rules:

- **GO-A fails** — criterion (i) separation is met in the isotropy-promoting direction (all g_k ≤ 0, −Σ = 6 ≥ 3) and the punchline holds (ISO@1.0× = 13 ≥ D1G@1.5× = 12), but criterion (ii) RA-type objective evidence fails: RA(D1G) = RA(ISO) at every scale, 0/3 wins. Geometry-specific causality is NOT established in EITHER direction.
- **GO-B fails** — |g_k| ≤ 1 at every scale is violated (g = −3/−2/−1). A clean magnitude-only law cannot be formally asserted at this sample size either.
- → **GO-C** per the pre-registered "otherwise" branch. Conservative conclusion:

> EXP019 provides suggestive but insufficient evidence that residual geometry is causally related to PnP ~180° ambiguity. At matched residual magnitude, changing residual geometry from matched-isotropic to D1-like did NOT increase flip probability or flip-type objective preference (if anything the flip-count separation ran mildly the other way, −3/−2/−1, directionally consistent with EXP017's within-D1 association but indistinguishable from noise at n = 20). Residual MAGNITUDE shows the strong, monotone dose–response, and matched-magnitude isotropic noise alone reproduces the real D1 flip burst (13/20 vs 14/20; clean 11/18 vs 12/18). Per protocol section 17/18: STOP — no geometry mining.

**What this experiment rules out:** D1-type residual geometry (anisotropy / coherent shift) as a causally SUFFICIENT driver of the ~180° ambiguity regime at matched magnitude; and consequently the reading that EXP017's D1 flip burst requires D1-specific residual geometry — matched-magnitude isotropic noise suffices.

**What it does NOT prove:** (i) geometry has exactly zero effect (n = 20, max gap 3/20, single random realization — underpowered for gaps of this size); (ii) a strict magnitude-only law (GO-B's |g| ≤ 1 criterion failed); (iii) anything about objects/datasets beyond this cat/20-image mechanism diagnosis, or about the solver's internal basin selection (EXP018's candidate-access limitation stands). The synthetic grid applies a UNIFORM per-image dose (RMS = scale × median), whereas real D1 residuals are heteroscedastic with catastrophic outliers — the grid tests the median-magnitude regime, not the real dose distribution.

## 16. Next Experiment Recommendation

Per the pre-registered GO-C/STOP rule, geometry mining stops here. The accumulated EXP014–019 chain supports the following reading: D1's keypoint improvement moves inputs into a magnitude/geometry regime where the ORIGINAL single-hypothesis `SOLVEPNP_ITERATIVE` solver frequently selects the ~180° basin; the driver is dominated by residual magnitude acting on image-specific geometry (7 insensitive vs 6 contradictory images), not by D1-specific residual geometry. Recommended successors (each requiring its own pre-registration):

1. **EXP020 (power, optional):** replicate the ISO-vs-D1G contrast with more OCC images and/or objects if the small isotropy-promoting separation is to be resolved at all — otherwise leave it unresolved.
2. **EXP021 (mechanism, higher value):** a NEW diagnostic branch using multi-hypothesis PnP (`solvePnPGeneric`, allowed ONLY there, never in the baseline) to expose the candidate landscape EXP018 could not observe, and test whether the ~180° basin is a near-tie that seed/init perturbations switch — this targets the actual failure locus (solver selection) instead of residual statistics.
3. Practical downstream check (no new science): report whether a trivial input perturbation guard (e.g. re-run PnP after ±1 px keypoint jitter and flag disagreement) would have caught the real D1 flips — a deployment-relevant consequence of the magnitude-dominated flip mechanism established here.
