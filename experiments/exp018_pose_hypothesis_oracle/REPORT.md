# EXP018: Pose Hypothesis Oracle Diagnosis

Branch: `exp018_pose_hypothesis_oracle` (created from `exp017_pose_hypothesis_landscape` @ `88783ad`)
Mode: **partial-evidence diagnosis** (objective-level oracle; candidate hypotheses are not exposed by the current implementation, therefore EXP018 can only provide partial evidence).
Cost: CPU-only, no network forward, no GPU; full run 0.9 s + analysis < 1 s.

## 1. Research Question

EXP015/016/017 established: the EXP013 D1 decoder improves OCC keypoint localization (median 3.150 → 2.143 px) yet the ORIGINAL standard PnP produces ~180° pose flips on 14/20 held-out images vs 4/20 for D0; flip is NOT driven by keypoint error magnitude but is associated with keypoint residual geometry (flip images: lower residual anisotropy, smaller coherent shift). EXP018 asks the single remaining causal question:

> **CASE A**: the correct pose hypothesis EXISTS in the solver's candidate space, and the original PnP's hypothesis SELECTION failed (it picked the wrong basin);
> **CASE B**: D1's keypoint geometry has already EXCLUDED the correct pose from the candidate space (hypothesis GENERATION failure);
> **CASE C**: the correct hypothesis exists but the reprojection OBJECTIVE itself prefers the wrong ~180° hypothesis (image-space reprojection ambiguity).

No solver, voting, keypoint, threshold, seed, GT, or architecture modification of any kind.

## 2. Hypothesis

Pre-registered in `config.json` (written BEFORE any PnP call; unmodified since):

- If, among D1 flip images, the correct (GT) pose fits the input keypoints **strictly better** than the solver-returned pose (J_gt < J_pred − eps) on a plurality → **SELECTION_FAILURE → GO-A**.
- If the ~180° pose fits the input keypoints **strictly better** than the correct pose (J_pred < J_gt − eps, wrong rot > 150°) on a plurality → **REPROJECTION_AMBIGUITY → GO-C** (which is simultaneously the strongest available evidence for the CASE B geometric reading, since a perfect objective-based selector would also choose wrong).
- Tie / near-tie plurality → STOP. GO-B's strict condition (correct hypothesis absent from the candidate space) is NOT verifiable without candidate access → documented limitation.
- tie_eps = 1e-3 px; flip thresholds (90/120/150/170°) reused from EXP016/017, not re-invented.

## 3. Experimental Isolation

- **Data**: the exact 20 OCC held-out images of EXP017 (`results/image_ids.json` asserted identical; EXP017 asserted == EXP015 == EXP016 usage). No other data touched.
- **Inputs**: D0/D1 voted keypoints and GT keypoints/poses loaded **bit-identically** from EXP016's persisted `raw_predictions/raw_occ_D{0,1}.npz` (assert `gt_kps`/`gt_pose` equal across arms). No re-voting, no re-fitting, no re-sampling.
- **Solver**: original `Evaluator.evaluate` → `pnp()` → `cv2.solvePnP(SOLVEPNP_ITERATIVE)`, original `linemod` intrinsics, original `VotingType.Farthest` 3D points. UNMODIFIED.
- **Metrics**: original official Evaluator outputs (rotation/translation error, ADD(-S), mean 2D reprojection) plus an oracle fit J computed with the SAME original `Projector.project_K` on the SAME 9 3D points the solver consumed.
- **No** retraining, no decoder change, no voting/RANSAC change, no new solver, no `solvePnPGeneric`, no threshold change, no keypoint-count change, no dataset expansion, no new seed, no full-dataset run.

Analysis-integrity note: the analyzer's initial delta sign was inverted relative to the pre-registered `config.json` (which was written before any PnP call and was never modified). The sign was corrected to match the pre-registration exactly before any interpretation was drawn; all numbers below come from the corrected analyzer (an artifact of the inverted run was discarded, not reported).

## 4. Exact Image IDs

`color_00024.png` … `color_00043.png` (20 OCC images; `image_id` 20–39 in internal indexing), asserted identical to EXP017's `results/image_ids.json`. Full list in `image_ids.json`.

## 5. PnP Implementation Path

Code check (read from source, not assumed):

- Baseline path: `lib/utils/evaluation_utils.py::Evaluator.evaluate → pnp() → cv2.solvePnP(flags=cv2.SOLVEPNP_ITERATIVE)`: **one deterministic solution**. No candidate list, no intermediate iterates, no inlier masks, no random initializations, no pose RANSAC (`solvePnPRansac` appears only in a comment).
- No `solvePnPGeneric` anywhere in the repo (and it is banned by this protocol as a "replacement method").
- The only other solvePnP sites are `uncertainty_pnp / uncertainty_pnp_v2` (extend_utils) — a DIFFERENT solver used by `evaluate_uncertainty`, NOT part of the baseline path; using it would violate "no new solver".
- `ransac_voting_layer_v3` performs RANSAC over vote LINES for KEYPOINTS, not object poses; it exposes no pose candidates.

Oracle fits use the original `Projector.project_K` with `linemod` intrinsics on the exact `VotingType.Farthest` 9 3D points passed to the solver. Sanity: max |project_K(kp3d, GT pose) − dataset gt_kps| = 1.68e-05 px (projector bias symmetric for J_pred and J_gt).

## 6. Candidate Accessibility

`cv2.solvePnP` with `SOLVEPNP_ITERATIVE` exposes only the final pose from Python; instrumenting OpenCV's internal DLT init / Newton iterates would require reimplementing the solver (forbidden) or changing flags/method (forbidden). Hence EXP018 runs in **partial-evidence mode**: the only observable poses are `pose_pred` (solver output) and `pose_gt` (rot 0 by construction, so a correct hypothesis — rot < 10° — exists among observable poses on every image, `D1_correct_hyp_exists = True` with this caveat on every row). The informative quantity is the objective's RANKING:

```
J(pose) = mean_i || project_K(kp3d_Farthest9, pose)_i − kp_input_i ||   (px)
J_pred  = fit of the solver-returned pose (its converged objective value)
J_gt    = fit of the correct (GT) pose
delta   = J_gt − J_pred
delta < −1e-3  → SELECTION_FAILURE   (J_gt < J_pred: correct pose fits strictly
                                      better; solver left it unfound; CASE A)
delta > +1e-3  → REPROJECTION_AMBIGUITY (J_pred < J_gt: wrong ~180° pose fits
                                      strictly better; Type D; CASE C)
|delta| ≤ 1e-3 → UNRESOLVED
NO_CORRECT_HYPOTHESIS: NOT assignable without the candidate space → recorded 0
with explicit caveat.
```

`D1_best_correct_reproj` = J_gt, `D1_best_wrong_reproj` = J_pred (the two observable poses; the candidate-space "best" cannot be computed without candidate access).

## 7. C0 GT Result

GT keypoints through the ORIGINAL pipeline, re-verified against EXP017:

- rotation error 0.000° on all 20 images; **flips 0/20 at every threshold (90/120/150/170°)**; rot/reproj/translation match EXP017 `per_image.csv` within 1e-3/1e-3/1e-2.
- Confirms EXP017 Check A: no intrinsic pose ambiguity for the cat object under the original PnP; the pipeline CAN recover the exact correct pose when keypoints are correct.

## 8. C1 D0 Result

- Flips: **>90° 4/20** (24, 25, 27, 43); >120° 3/20; >150° 2/20; >170° 2/20.
- Reproduction vs EXP017: exact (rot < 1e-3°, reproj < 1e-3 px, trans < 1e-2 mm); fresh poses bit-match EXP016 stored `pred_pose` (< 1e-9, asserted per image).
- Oracle on D0 flips: REPROJECTION_AMBIGUITY 3 (24: J_pred 4.845 vs J_gt 14.083; 25: 4.319 vs 399.162, catastrophic voting; 43: 0.856 vs 2.612), SELECTION_FAILURE 1 (27: J_pred 589.726 vs J_gt 479.640, catastrophic voting; both fits are hundreds of px).
- D0 flip median delta = **+5.497 px** (wrong pose fits better).

## 9. C2 D1 Result

- Flips: **>90° 14/20** (24–28, 33, 35, 37–43); >120° 13/20; >150° 12/20; >170° 10/20. Exact reproduction as above (bit-match < 1e-9).
- Diagnosis counts (14 flips): **REPROJECTION_AMBIGUITY 10** (rot > 150° in 9), **SELECTION_FAILURE 4**, NORMAL 6, UNRESOLVED 0.
- D1 flip median delta = **+0.404 px** (wrong pose fits better); mean −120.5 px is dominated by the single solver-divergence image 26.
- SELECTION_FAILURE breakdown: 26 (solver diverged: J_pred 2001.28 px vs J_gt 11.64 px — the solver failed to converge at all, not a basin choice), 27 (catastrophic voting: both fits 480–590 px), 37 (−0.202 px), 39 (−0.305 px). Only 37/39 are "clean" selection failures, with sub-pixel margins.

## 10. Candidate Hypothesis Analysis

Because the candidate space is not exposed (Section 6), the analysis is over the two observable poses:

| Type | Definition (observable-pose proxy) | D0 | D1 |
|---|---|---:|---:|
| A — NORMAL (correct basin selected) | rot < 90° | 16 | 6 |
| B — SELECTION_FAILURE | flip AND J_gt < J_pred − eps | 1 | 4 |
| C — NO_CORRECT_HYPOTHESIS | correct hypothesis absent from candidate space | **not assignable** (0, caveat) | **not assignable** (0, caveat) |
| D — REPROJECTION_AMBIGUITY | flip AND J_pred < J_gt − eps | 3 (rot>150°: 2) | 10 (rot>150°: 9) |

Key clean-subset statistics (excluding catastrophic 25/27 and solver-divergence 26):

- D1 clean flips: 11 → **RA 9 / SF 2**. In 9/11 clean flips the correct pose does NOT fit the D1 keypoints better — the objective itself prefers the wrong basin.
- Clean RA cluster: median J_pred = **1.113 px** vs median J_gt = **1.890 px**; images 33–43 form a near-consecutive run of 9 flips (7 RA + 2 SF) with wrong-pose rotation 176.5–180.0° and J_pred 0.84–1.74 px — both poses fit to ~1–3 px, and the WRONG one fits better.
- D0 also exhibits RA on 3/4 of its flips (24, 25, 43) — reprojection ambiguity is a property of the objective + keypoint configuration, not unique to D1; but D1 enters the ambiguous regime on 10/20 images vs D0's 3/20.

## 11. Reprojection Ambiguity Analysis

The dominant D1 pattern is exactly the user-specified Type D signature:

- wrong pose: rotation 147–180°, reprojection (oracle) ≈ 0.8–6.4 px (clean cluster 0.84–1.74 px; official Evaluator reprojection likewise ≈ 1.0–1.7 px, e.g. 33: 1.51, 37: 1.06, 39: 0.94, 43: 1.05 — far inside the project's official < 5 px 2D threshold);
- correct pose: rotation 0° by construction, but oracle fit J_gt HIGHER than the wrong pose's on 10/14 flips (clean-cluster margins +0.29 … +9.60 px).

This confirms and localizes EXP016's observation ("flipped poses with reprojection < 5 px, D1 flip-group reprojection median LOWER than stable"): the wrong ~180° hypothesis is not merely acceptable in image space — it is the **better-fitting** solution to D1's keypoints under the original objective. A hypothetical perfect selection layer that minimizes the SAME reprojection objective would therefore also choose wrong on these images: this is the strongest available evidence for the CASE B geometric reading (D1's residual geometry makes the wrong basin the objective's global preference), while the strict GO-B condition remains unverifiable without candidate access.

Residual geometry re-validation (EXP017 `residual_geometry.csv`, read-only; D1 flip n=14 vs stable n=6, medians): kp_median 1.880 vs 1.991 px; coherent shift 1.032 vs 1.409 px; relative scatter 1.670 vs 1.775 px; anisotropy 2.78 vs 6.94. All three pre-registered checks hold (flip anisotropy lower ✓, flip coherent shift smaller ✓, magnitude not explanatory ✓) — EXP017's findings are unchanged under EXP018's reproduction of the keypoint inputs.

## 12. Per-image Diagnosis

Full table: `results/per_image_diagnosis.csv` (plus `results/objective_oracle.csv` for C0/C1/C2 raw values). "reproj" = original official Evaluator mean reprojection (all model points); "J_*" = oracle fit over the solver's 9 Farthest points. `D1_correct_hyp_exists = True` on every row among observable poses (candidate-space caveat applies to all rows).

| image | D0 flip | D1 flip | D0 reproj | D1 reproj | D0 rot° | D1 rot° | D1 hyp exists | D1 best_correct (J_gt) | D1 best_wrong (J_pred) | diagnosis |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| color_00024 | 1 | 1 | 10.4145 | 13.2017 | 180.000 | 165.557 | True | 15.991 | 6.395 | REPROJECTION_AMBIGUITY |
| color_00025 | 1 | 1 | 401.4582 | 400.2365 | 149.329 | 147.464 | True | 398.068 | 3.803 | REPROJECTION_AMBIGUITY |
| color_00026 | 0 | 1 | 7.9436 | 40072.2679 | 30.744 | 161.543 | True | 11.643 | 2001.282 | SELECTION_FAILURE |
| color_00027 | 1 | 1 | 2828.7046 | 2828.7046 | 108.926 | 108.926 | True | 479.640 | 589.726 | SELECTION_FAILURE |
| color_00028 | 0 | 1 | 4.8280 | 6.1694 | 16.169 | 171.591 | True | 7.000 | 2.317 | REPROJECTION_AMBIGUITY |
| color_00029 | 0 | 0 | 3.5085 | 3.9937 | 9.468 | 10.413 | True | 4.351 | 0.600 | NORMAL |
| color_00030 | 0 | 0 | 1.8746 | 2.5197 | 6.213 | 6.967 | True | 2.699 | 0.773 | NORMAL |
| color_00031 | 0 | 0 | 1.6895 | 2.0468 | 5.691 | 6.997 | True | 2.316 | 0.740 | NORMAL |
| color_00032 | 0 | 0 | 1.7307 | 2.0192 | 7.600 | 6.103 | True | 2.191 | 0.675 | NORMAL |
| color_00033 | 0 | 1 | 2.1799 | 1.5092 | 9.361 | 177.982 | True | 1.649 | 0.869 | REPROJECTION_AMBIGUITY |
| color_00034 | 0 | 0 | 2.2536 | 1.3810 | 9.398 | 5.747 | True | 1.660 | 0.594 | NORMAL |
| color_00035 | 0 | 1 | 1.2956 | 2.6002 | 3.599 | 176.532 | True | 2.728 | 0.843 | REPROJECTION_AMBIGUITY |
| color_00036 | 0 | 0 | 1.2906 | 1.6223 | 4.987 | 3.859 | True | 1.813 | 0.751 | NORMAL |
| color_00037 | 0 | 1 | 2.4986 | 1.0555 | 6.572 | 180.000 | True | 1.059 | 1.262 | SELECTION_FAILURE |
| color_00038 | 0 | 1 | 2.7390 | 1.4428 | 7.264 | 178.663 | True | 2.073 | 1.744 | REPROJECTION_AMBIGUITY |
| color_00039 | 0 | 1 | 2.7158 | 0.9442 | 9.510 | 179.168 | True | 1.279 | 1.584 | SELECTION_FAILURE |
| color_00040 | 0 | 1 | 2.6806 | 1.3028 | 9.159 | 178.934 | True | 1.813 | 1.523 | REPROJECTION_AMBIGUITY |
| color_00041 | 0 | 1 | 2.7882 | 1.2125 | 9.748 | 178.263 | True | 1.458 | 1.113 | REPROJECTION_AMBIGUITY |
| color_00042 | 0 | 1 | 3.2345 | 1.6649 | 10.944 | 179.001 | True | 1.890 | 0.960 | REPROJECTION_AMBIGUITY |
| color_00043 | 1 | 1 | 2.6271 | 1.0532 | 176.201 | 179.032 | True | 1.462 | 0.998 | REPROJECTION_AMBIGUITY |

Most informative cases:

1. **color_00033** — D0 rot 9.4° (correct basin) → D1 rot 178.0°; D1 keypoints fit the WRONG pose at 0.869 px vs the correct pose's 1.649 px. A perfect objective-based selector would also flip.
2. **color_00042** — wrong 179.0° pose fits at 0.960 px vs correct 1.890 px (largest clean margin, +0.93 px); D1's official reprojection (1.66 px) is BETTER than D0's (3.23 px) while the pose is wrong — keypoint fit and pose correctness are decoupled by objective ambiguity.
3. **color_00039** — one of only two clean SELECTION_FAILUREs: correct pose fits strictly better (1.279 vs 1.584 px) yet the solver returned the 179.2° pose; margin only −0.305 px.
4. **color_00026** — solver divergence, not a basin choice: J_pred 2001 px (official reproj 40072 px) vs correct 11.64 px; iterative PnP failed outright on D1 inputs where D0 was NORMAL (rot 30.7°).
5. **color_00024** — D0 is ALSO ambiguous here (wrong 180.0° pose fits 4.85 px vs correct 14.08 px), showing the ambiguity predates D1; D1 merely lands far more images (10/20 vs 3/20) in this regime.

## 13. Decision

**GO-C.**

Pre-registered plurality among D1 flips: REPROJECTION_AMBIGUITY(rot>150°) 9 vs SELECTION_FAILURE 4 vs UNRESOLVED 0 → GO-C.

Conclusion: on the majority of D1 flip images the correct pose fits the input keypoints WORSE than the ~180° pose under the original reprojection objective — the objective itself is pose-ambiguous for the D1 keypoint configuration (Type D / image-space reprojection ambiguity). Without candidate-space access this is simultaneously the strongest available evidence for the CASE B geometric reading (D1's residual geometry makes the wrong basin the objective's preference; a perfect objective-based selector would also choose wrong); the strict GO-B condition (correct hypothesis absent from the candidate space) is not verifiable here and remains a documented limitation. GO-A (pure selection failure) is rejected as the majority mechanism: only 2 clean D1 flips show the solver leaving a strictly better-fitting correct hypothesis unfound, with sub-pixel margins.

## 14. What This Experiment Rules Out

1. **Majority hypothesis-selection failure (GO-A)** as the mechanism of D1's flip burst: in 9/11 clean D1 flips the correct pose does not fit better — the solver did not "overlook" a better hypothesis under its own objective; it returned the objective's preferred (wrong) one.
2. **Keypoint error magnitude** as the discriminator: re-validated (Section 11) — flip images have the same keypoint error magnitude as stable images (1.880 vs 1.991 px) but different residual geometry.
3. **Intrinsic pose ambiguity**: C0 GT keypoints → 0/20 flips, rot 0.000°, re-verified through the unmodified pipeline.
4. **PnP-convergence quality as the primary mechanism**: excluding the divergent image 26, D1's flipped solutions are internally converged with excellent image-space consistency (J_pred ≈ 0.8–1.7 px) — they converge *to the wrong basin* rather than failing to converge.

## 15. What It Does NOT Prove

1. It does **not** prove NO_CORRECT_HYPOTHESIS (Type C / strict CASE B): the candidate space was never observable; correct_hypothesis_exists is only vacuously true among the two observable poses.
2. It cannot fully **separate CASE B from CASE C**: with only pose_pred and pose_gt observable, "objective prefers the wrong basin" is evidence for both the geometric-basin reading and the objective-ambiguity reading; disambiguating them requires candidate-level or controlled-residual evidence.
3. The oracle J is the solver's own reprojection proxy (mean distance over the 9 Farthest 3D points), not a likelihood over all correspondences; margins are small (clean-cluster median gap 0.78 px) though consistent across 9 images.
4. All conclusions are association-level, on 20 cat-only OCC held-out images, D0/D1 as frozen by EXP016; no cross-object or cross-dataset claim.

## 16. Next Experiment Recommendation

EXP019 — **residual-geometry dose–response through the unmodified solver** (diagnosis, no new method): start from GT keypoints and inject controlled residual fields that vary anisotropy and coherent-shift magnitude (structured, one shared realization per geometry class, seed = 0), run the ORIGINAL PnP unchanged on each configuration, and map P(flip) and the objective gap J_gt − J_pred over the (anisotropy, coherent shift) plane. This directly tests the mechanism claim this experiment supports — that D1-like isotropic, small-coherent-shift residual geometry is SUFFICIENT to move the objective's preference into the ~180° basin — while keeping solver, thresholds, and data untouched. CPU-only, seconds-scale, same 20 held-out images.

---

This experiment is a mechanism diagnosis on the cat-only PVNet baseline and does not establish cross-object generalization.
