# EXP014: End-to-End Pose Causal Validation

## 1. Research Question

> Can the generalized direction improvement from EXP013 propagate through the
> original PVNet voting and PnP pipeline to improve 6D pose accuracy?

EXP012/013 established that a lightweight cross-channel 3×3 decoder (D1) on
the frozen PVNet shared feature improves pixel-to-keypoint direction
prediction and that the improvement generalizes to held-out images
(HELD-OUT OCC 6.156° → 4.936°, +19.83%, GO-B). EXP014 is a causal
validation experiment only: no new method, no architecture work, no
optimization, no retraining, no parameter tuning.

## 2. Experimental isolation

The ONLY variable is the vertex prediction:

* **D0 Original**: 199.pth original convraw[3] 1×1 vertex head.
* **D1 Context decoder**: 199.pth + EXP013 frozen D1 decoder
  (zero-init Conv3x3(32→32) → LeakyReLU(0.1) → Conv1x1(32→18) residual,
  9842 params).

Everything downstream is the project's original baseline, unmodified:

* Voting: `lib.ransac_voting_gpu_layer.ransac_voting_gpu.ransac_voting_layer_v3`
  via the EvalWrapperBaseline call convention; parameters taken from
  `configs/exp001_reliability_voting_scratch.json` and asserted at runtime
  (round_hyp_num=128, inlier_thresh=0.99, max_num=100). Seed 0 re-applied
  before EVERY voting call so both branches see the identical RNG stream.
  No RANSAC parameters, thresholds, or weighting were changed.
* Pose: the project's original `Evaluator` (`lib/utils/evaluation_utils.py`)
  standard-PnP path (`pnp()`, cv2.SOLVEPNP_ITERATIVE, `linemod` intrinsics,
  Farthest 3D points, original ply model/diameter). Uncertainty PnP was NOT
  used (per protocol the project's formal baseline is standard PnP). ADD is
  the evaluator's standard ADD recorder (non-symmetric; cat), consistent
  with the "ADD(-S)" figures in prior reports.
* Segmentation is untouched by construction (D1 adds channels after the
  original head split); mask equality asserted per image.

D1 parameters: EXP013 did not persist the decoder weights, so the final D1
state was regenerated deterministically with the EXP013 fitting protocol
verbatim (same data, seed/optimizer/epochs; init loss exactly equals the D0
reference, and the training-loss trajectory is identical to EXP013's to all
recorded digits). The regenerated state was verified against EXP013's
recorded HELD-OUT direction numbers BEFORE any pose evaluation (gate below)
and then frozen. No re-fitting decisions, no tuning. The state is saved
locally as `results/d1_decoder_<split>.pth` (not committed, per checkpoint
rule).

## 3. Held-out data

HELD-OUT set only, IDs loaded from
`experiments/exp013_heldout_decoder_generalization/results/image_ids.json`:

* LIN (20): 000052, 000053, 000055, 000056, 000057, 000058, 000059, 000061,
  000063, 000074, 000075, 000077, 000078, 000079, 000080, 000083, 000087,
  000091, 000092, 000095 (`val_real_set[20:40]`)
* OCC (20): color_00024 … color_00043 (`test_real_set` first half `[20:40]`)

Leakage control: FIT images are excluded (IDs asserted against EXP013's
file), and EXP013's recorded `disjoint_assert` flags are re-asserted at
runtime (`set(fit) ∩ set(heldout) = ∅` for both splits). No re-sampling.

## 4. D0 reproduction

**Pipeline identity check** (before any fitting/evaluation, all 40 held-out
images): D0 branch (manual trunk → convraw[3]) vs original `net.forward`
pipeline —

```
max |seg diff|     = 0.00e+00
max |vertex diff|  = 0.00e+00
argmax mask        = identical on every image
max |keypoint diff| (identical voting call) = 0.00e+00
```

PASSED; the D0 branch IS the original baseline numerically.

**Direction sanity check** vs EXP013 recorded HELD-OUT values (gate, tol
0.01°) — PASSED exactly:

| | LIN | OCC |
| --- | ---: | ---: |
| D0 | 2.436° (EXP013: 2.436) | 6.156° (EXP013: 6.156) |
| D1 | 2.541° (EXP013: 2.541) | 4.936° (EXP013: 4.936) |

The D0 regen fitting reference losses also match EXP013 exactly
(LIN 0.002869 / OCC 0.008945; identical 5-epoch trajectories).

## 5. Direction results (Level 1; EXP006/008 pool, 9 kps pooled)

| Split | Decoder | mean° | median° | p90° | >10° | endpoint (unit) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LIN | D0 | 2.436 | 1.485 | 5.025 | 0.0263 | 0.0423 |
| LIN | D1 | 2.541 | 1.629 | 5.094 | 0.0270 | 0.0441 |
| OCC | D0 | 6.156 | 3.551 | 12.873 | 0.1460 | 0.1053 |
| OCC | D1 | 4.936 | 2.873 | 9.975 | 0.0997 | 0.0847 |

OCC direction improves −19.8% (mean), LIN degrades slightly +4.3% — the
known EXP013 result, unchanged (per protocol D1 was not touched because of
the LIN degradation).

## 6. Keypoint results (Level 2; voted keypoints vs GT projected keypoints,
pooled over all 9 keypoints × 20 images)

| Split | Decoder | mean px | median px |
| --- | --- | ---: | ---: |
| LIN | D0 | 1.226 | 1.111 |
| LIN | D1 | 1.310 | 1.235 |
| OCC | D0 | 47.410 | 3.150 |
| OCC | D1 | 47.139 | 2.143 |

Per-keypoint means are in `results/keypoint_summary.csv` (no keypoint-specific
analysis performed, per protocol).

Reading (reported as-is): OCC **median** KP error improves 32% (3.150 →
2.143 px) and KP mean error improves on **12/20** OCC images — but the OCC
mean is dominated by 2 catastrophic voting failures shared by both branches
(color_00025: ~399 px both; color_00027: 479.64 px — bit-identical for both
branches), so the OCC mean is flat (47.410 → 47.139). LIN KP degrades
slightly (mean +0.08 px), mirroring the LIN direction degradation.

## 7. Pose results (Level 3; original Evaluator, standard PnP, 20 images each)

Primary metric — ADD (fraction of images with ADD < 0.1×diameter; success
rate, higher = better):

| Method | LIN ADD | OCC ADD |
| --- | ---: | ---: |
| D0 Original | 0.65 (13/20) | 0.05 (1/20) |
| D1 Context decoder | 0.70 (14/20) | 0.00 (0/20) |

Secondary (same evaluator):

| Method | LIN 2D (<5px) | OCC 2D (<5px) | LIN 5cm5deg | OCC 5cm5deg | LIN ADD mean dist | OCC ADD mean dist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 | 1.00 (1.08 px) | 0.80 (164 px) | 1.00 | 0.10 | 13.27 mm | 468.65 mm |
| D1 | 1.00 (1.14 px) | 0.75 (2167 px) | 1.00 | 0.05 | 11.93 mm | 1632.67 mm |

OCC mean columns are tail-dominated (see §8) and not run-stable in magnitude
(§8 note 3); the threshold fractions were stable across three runs.

## 8. Causal chain (direction → keypoint → pose)

```
L1 Direction   OCC 6.156° → 4.936°   (improved, −19.8%)
L2 Keypoint    OCC mean 47.41 → 47.14 px (flat, tail-dominated)
               OCC median 3.15 → 2.14 px (improved, −32%)
L3 Pose        OCC ADD 1/20 → 0/20 (−100%), 2D 0.80 → 0.75, 5cm5deg 0.10 → 0.05
```

Per the pre-registered §14 analysis rule, this is the
"direction improvement / keypoint(typical) improvement / pose improvement ≈ 0
(or negative)" pattern — a mechanism finding, not a method failure: **the
bottleneck sits in keypoint estimation → PnP, not in direction
representation.** Three concrete pipeline phenomena drive it (descriptive,
from the same run, no modifications made):

1. **Standard-PnP degenerate ~180° flip mode.** With 1–2 px model
   reprojection, `cv2.SOLVEPNP_ITERATIVE` sometimes converges to a solution
   with rot err ≈ 165–180° and translation err ≈ 2.4–3.4 m
   (e.g., color_00043 under D0: KP 2.61 px, reproj 2.63 px, rot 176.2°,
   trans 2548 mm). This is a two-fold ambiguity of the small cat model under
   the project's unmodified standard PnP. D0 hits it on 2/20 OCC images;
   D1 on 10/20 — including color_00033, D0's ONLY ADD-passing image, where
   D1's keypoints are actually *better* (2.58 → 1.65 px) yet the solve
   flips (rot 178.0°, trans 2525 mm). Per-image rot/trans for all 40 images
   are in `results/per_image_pose.csv`.
2. **Shared catastrophic voting failures.** color_00025 / color_00027 fail
   voting for BOTH branches (KP ≈ 399/480 px); they cap what any
   vertex-field improvement can deliver at pose level on this set.
3. **RANSAC near-tie nondeterminism.** Across three identical runs the
   threshold fractions were stable (ADD 0.65/0.70, 0.05/0.00; 2D; 5cm5deg),
   but tail magnitudes on degenerate images are not (OCC D1 2D mean
   555/640/2167 px across runs) — atomic-ordering effects in the RANSAC
   scoring on near-tie degenerate images. Magnitude means on OCC should
   therefore be read as order-of-magnitude indicators only.

Where the PnP behaves (non-flipped, non-collapsed images), D1's KP
improvement does translate into equal-or-better ADD distance on most images
(e.g., color_00029: 141.5 → 130.8 mm; color_00030: 90.4 → 86.3 mm;
color_00032: 16.6 → 15.2 mm) — but this is visible only in the mean-distance
secondary metric, not in the threshold metrics, and LIN moves within noise
(3 single-image ADD flips: 2 gained, 1 lost).

## 9. GO / STOP (pre-registered, thresholds unchanged)

HELD-OUT OCC ADD improvement = (0.00 − 0.05) / 0.05 = **−100%** < 3%

→ **STOP-C**

> The improved direction representation does not translate into meaningful
> pose improvement under the current PVNet geometric pipeline. Do not
> continue modifying the decoder; the next step is to locate the bottleneck
> in the direction → keypoint → pose chain (the evidence above points at
> standard-PnP fragility — the ~180° flip basin — plus shared voting
> collapses, not at the direction field).

(For completeness: GO-C would have required ≥ +10%; BORDERLINE band 3–10%
does not apply. Note the improvement sign convention: ADD/2D/5cm5deg are
success rates, so improvement is the relative increase (D1−D0)/D0.)

## 10. Limitations

* Only 20 LIN + 20 OCC held-out images; ADD fractions move in 5-percentage-point
  steps (one image), so LIN "+7.69%" and OCC "−100%" are both single-image-flip
  scale effects.
* Cat-only baseline; single seed; diagnostic validation only.
* The D1 state is a deterministic regeneration of EXP013's final decoder
  (EXP013 did not persist weights), verified against EXP013's recorded
  results before use; it is not an independent refit.
* No proximity analysis (per protocol), no cross-object or cross-dataset
  claims of any kind.

---
This experiment is a mechanism diagnosis on the cat-only PVNet baseline and does not establish cross-object generalization.
