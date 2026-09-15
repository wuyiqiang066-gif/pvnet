# EXP017 — Pose Hypothesis Landscape Diagnosis (REPORT)

Branch `exp017_pose_hypothesis_landscape` (parent `exp016_pnp_stability_diagnosis` @ e4348c1).
Pure diagnosis. No training, no baseline/PVNet/vertex/voting/PnP modification, no solver
replacement, no tuning, no sigma sweep, no extra seeds/datasets, no EXP018 designed here.

## 1. Question

EXP016 showed D1's improved OCC keypoints push the ORIGINAL standard PnP into ~180° flip
basins far more often than D0 (14/20 vs 4/20 flips >90°), while keypoint error magnitude
does not separate flip from stable images. EXP017 determines whether this is:

- **A. intrinsic ambiguity** — the cat keypoint/PnP geometry itself admits the flipped
  solution even near zero keypoint error; or
- **B. residual-geometry-associated basin selection** — GT pose is stable, but D1's
  specific keypoint residual geometry pushes the original PnP into pre-existing wrong
  hypothesis basins; or
- **C. perturbation-sensitive numerical instability** — the original PnP is so sensitive
  to small perturbations that matched 1 px noise erases the D0/D1 difference; or
- **D. inconclusive.**

## 2. Fixed protocol

- Data: EXP016 `results/raw_predictions/raw_occ_D{0,1}.npz` reused **bit-identically**
  (voted keypoints, GT projected keypoints, GT poses) for the same 20 OCC held-out images
  as EXP015/016. Image IDs asserted equal (EXP015 `image_ids.json` == EXP016
  `per_image.csv` occ rows == npz ids 20–39); IDs saved to `results/image_ids.json`.
  Camera intrinsics (`linemod`), 3D keypoints (`VotingType.Farthest`), cat ply model,
  GT poses, evaluation protocol, and PnP solver are all the project's original artifacts,
  loaded through the unmodified `Evaluator`.
- PnP: original `lib.utils.evaluation_utils.pnp` via `Evaluator.evaluate`
  (cv2.SOLVEPNP_ITERATIVE, zero dist coeffs). No flag/initialization/iteration/RANSAC/
  refinement/multi-hypothesis/flip-correction changes; no GT-based solution selection.
- Conditions (the ONLY manipulated variable is the 2D keypoint vector):
  - **C0** GT projected keypoints (control)
  - **C1** EXP016 D0 voted keypoints
  - **C2** EXP016 D1 voted keypoints
  - **C3** C1 + ε, ε_x, ε_y ~ N(0, 1.0 px²) iid, seed=0, ONE run (no sweep)
  - **C4** C2 + the **same** per-image/keypoint ε realization as C3
- Flip definition (EXP016, not re-invented): rot err = acos((trace(R_pred R_gtᵀ)−1)/2);
  flip = rot > 90° (also 120/150/170° recorded).
- Pre-registered decision rules (priority A → C → B → D) written to `config.json`
  BEFORE any PnP call; thresholds in `results/config.json`.

## 3. Identity / reproduction checks

- **Check A (GT PnP)**: C0 rotation error is exactly 0.000° on all 20 images (0/20 flips
  at every threshold, reprojection 0, ADD pass 20/20).
  `INTRINSIC_AMBIGUITY_SUSPECTED = False`. C1–C4 still completed as pre-registered.
- **Check B (reproduction)**: PASSED. C1 reproduces EXP016 D0 and C2 reproduces EXP016 D1
  per-image flip flags exactly (4/20 and 14/20, same image IDs); max rotation diff
  4.8e-4°, max reprojection diff 5.0e-4 px (CSV 3-decimal rounding only; inputs are
  bit-identical). Per-image table in `results/reproduction_check.json`.
- **Check C (noise identity)**: PASSED. C3/C4 share one ε realization; elementwise
  max|diff| = 1.4e-14 < 1e-12. Perturbation norms: mean 1.249 px, median 1.161 px,
  p90 2.180 px (theory mean 1.253 px); `results/noise_statistics.json`.

## 4. C0–C4 main results

Per-image records: `results/per_image.csv`; aggregates: `results/summary.csv`.

| cond | kp median err (px, med-of-medians) | rot median (°) | ADD pass | reproj mean (px, med) | ADD mean dist (mm, med) |
|------|------|------|------|------|------|
| C0 GT | 0.000 | 0.000 | 20/20 | 0.000 | 0.08 |
| C1 D0 | 2.896 | 9.4 | 1/20 | 2.698 | 52.1 |
| C2 D1 | 1.920 | 168.6 | 0/20 | 1.842 | 2389.4 |
| C3 D0+1px | 3.224 | 11.1 | 2/20 | 2.956 | 67.3 |
| C4 D1+1px | 2.206 | 177.4 | 1/20 | 1.818 | 2475.7 |

D1's better keypoints (C1→C2 median 2.90→1.92 px) coincide with rot median 9.4°→168.6°,
i.e. better 2D keypoint quality lands in the flipped basin far more often.

## 5. Flip statistics

| cond | >90° | >120° | >150° | >170° |
|------|------|------|------|------|
| C0 GT | 0/20 | 0/20 | 0/20 | 0/20 |
| C1 D0 | 4/20 | 3/20 | 2/20 | 2/20 |
| C2 D1 | 14/20 | 13/20 | 12/20 | 10/20 |
| C3 D0+1px | 6/20 | 6/20 | 5/20 | 4/20 |
| C4 D1+1px | 17/20 | 17/20 | 17/20 | 14/20 |

Flip image sets: C1 {24,25,27,43}; C2 {24,25,26,27,28,33,35,37,38,39,40,41,42,43};
C3 {24,25,27,30,32,33}; C4 = C2's 14 ∪ {30,31,36}. Both D0-derived conditions C1/C3
include 24/25/27 (25/27 are EXP015's shared catastrophic images), and the D1-derived
C4 retains them as well.
Noise never rescues a D1 flip (all 14 persist) and adds 3 more; D0+1px adds 3 new flips
and loses 1. The D0/D1 gap WIDENS under matched noise (10 → 11 images; rates 20%→30% vs
70%→85%).

## 6. Keypoint residual geometry

Residual r_i = predicted_kp_i − GT_kp_i (px); per-image values in
`results/residual_geometry.csv` (medians below).

| arm | mean dx | mean dy | mean ‖r‖ | coherent shift ‖r̄‖ | rel. scatter mean | rel. scatter p90 | anisotropy λmax/λmin |
|-----|------|------|------|------|------|------|------|
| D0 | −1.090 | −0.822 | 3.050 | 1.792 | 2.487 | 3.826 | 4.30 |
| D1 | +0.539 | −0.092 | 2.132 | 1.245 | 1.701 | 3.092 | 3.57 |

### D1 flip vs stable grouping (n=14 vs n=6, descriptive medians)

| group | kp median (px) | coherent shift (px) | rel. scatter mean (px) | anisotropy | reproj mean (px) | rot median (°) |
|-------|------|------|------|------|------|------|
| flip | 1.880 | 1.032 | 1.670 | 2.78 | 1.587 | 178.1 |
| stable | 1.991 | 1.409 | 1.775 | 6.94 | 2.033 | 6.5 |

Separation ratios (flip/stable): rel. scatter 0.94 (no separation), coherent shift 0.73
(no separation under the 1.5× rule), **anisotropy 0.40 (separated, ≤ 1/1.5)**.
D1 flip images are distinguished by a MORE isotropic residual covariance (2.78 vs 6.94)
and a slightly smaller coherent shift, NOT by larger keypoint error.

## 7. Noise sensitivity

With the identical ε realization added to both arms, D0+1px flips rise only 4→6/20 while
D1+1px rises 14→17/20. The D0/D1 difference does NOT disappear under matched perturbation
(|C3−C4| = 11 ≫ 4, pre-registered CASE C boundary), and every C1/D1 flip persists under
noise. Perturbation of the keypoint configuration does alter the selected hypothesis
(C3 gains 2 flips, C4 gains 3), confirming the solver operates near a basin boundary —
but the basin-entry propensity is systematically higher for the D1 keypoint field, not a
generic 1 px-noise instability of the original PnP.

## 8. Case A/B/C/D decision

**DECISION = CASE B** (pre-registered priority A → C → B → D):

- CASE A excluded: C0 (GT) → 0/20 flips, rot exactly 0.000° — no intrinsic ambiguity
  near zero keypoint error.
- CASE C excluded: matched 1 px noise leaves a wide D0/D1 gap (6/20 vs 17/20; gap 11 > 4)
  — the difference does not largely disappear.
- CASE B met: C0 stable (0/20); C1 = 4/20 ≤ 6 and C2 = 14/20 ≥ 10 reproduced exactly;
  keypoint error magnitude does not explain flip (D1 flip/stable kp-median ratio 0.94);
  residual geometry separates D1 flip from stable (anisotropy ratio 0.40 ≤ 1/1.5).

Conclusion: the wrong pose hypothesis is not caused by larger keypoint error; it is
**associated with the geometry/structure of the keypoint residual field** (notably a more
isotropic residual covariance on flip images). This is a controlled diagnosis; causality
is not claimed beyond it.

## 9. Mechanism interpretation

- The original iterative PnP returns the exact GT pose from GT keypoints: a correct
  keypoint configuration never enters the flipped basin, so the ~180° basin exists in the
  solver/geometry but is not intrinsically favored.
- Under D1, flipped images have keypoint errors as small as stable ones (1.88 vs 1.99 px
  median) — magnitude carries no signal about basin selection (consistent with EXP016).
- What differs is the residual field's shape: D1 flip images show markedly more isotropic
  residual covariance (anisotropy 2.78 vs 6.94) and slightly smaller coherent shift;
  D0's residuals are overall larger and more coherent (mean shift −1.09, −0.82 px).
- Small, geometry-specific changes of the keypoint configuration can flip the selected
  hypothesis (C3/C4 gain flips; C4 retains all C2 flips): the original PnP sits near a
  ~180° basin boundary whose entry depends on the keypoint error GEOMETRY, not magnitude.
- Consistent with EXP016, flipped D1 poses remain image-space-consistent (flip-group
  median reprojection 1.59 px < the official 5 px 2D threshold), so 2D reprojection
  cannot disambiguate the two basins.

## 10. Limitations

- Cat-only, 20-image mechanism diagnosis; no significance claims are made from n=20.
- Single noise realization (seed=0, σ=1 px), single run, no sigma sweep — sensitivity is
  sampled at one perturbation scale only, by pre-registration.
- "Association" language only: the geometry metrics are associated with basin selection,
  not proven causal; the underlying degeneracy of the cat's approximate symmetry and the
  iterative solver's initialization dynamics are not dissected here.
- Keypoints are EXP016's RANSAC voting outputs; voting-induced correlation between
  residuals is inherited, not controlled.
- Two catastrophic images (color_00025/27) remain present in all D0-derived conditions;
  group comparisons are descriptive medians, not robustness-audited statistics.
- This experiment is a mechanism diagnosis on the cat-only PVNet baseline and does not
  establish cross-object generalization.

## 11. STOP

EXP017 is complete. Per the pre-registered protocol, no follow-up experiment, solver
change, or method design is proposed here. Waiting for instructions.
