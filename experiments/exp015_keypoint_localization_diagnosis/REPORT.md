# EXP015: Keypoint Localization Causal Diagnosis

## 1. Research Question

> Did the D1 OCC direction improvement (EXP013: 6.156° → 4.936°, +19.8%;
> verified again here) actually and stably transfer to PVNet keypoint
> localization, or was EXP014's keypoint-median improvement (3.150 → 2.143 px)
> a limited-sample phenomenon? And where in the direction → keypoint → pose
> chain does EXP014's pose failure actually occur?

This is a measurement experiment. Pipeline: image → segmentation → vertex
prediction → **original PVNet RANSAC voting → predicted keypoints, STOP**.
No PnP, no ADD, no pose is computed anywhere in this experiment.

## 2. Experimental isolation

The ONLY variable is the vertex prediction:

* **D0 Original**: 199.pth original convraw[3] 1×1 vertex head.
* **D1 Context decoder**: 199.pth + EXP013 final D1 decoder (zero-init
  Conv3x3(32→32) → LeakyReLU(0.1) → Conv1x1(32→18) residual, 9842 params),
  loaded from EXP014's persisted states. **No re-fitting.**

Everything else identical: input, frozen backbone/shared features,
segmentation, voting implementation (`ransac_voting_layer_v3`), RANSAC
parameters (round_hyp_num=128, inlier_thresh=0.99, max_num=100, asserted at
runtime), seed 0 re-applied before EVERY voting call (identical RNG stream for
both branches), intrinsics, GT keypoints.

Identity checks (all passed, abort thresholds 1e-5):

| Check | Result |
| --- | --- |
| D0 vs original net.forward: seg logits / vertex / mask / voted keypoints | max diff 0.00e+00, masks identical, kp diff 0.00e+00 |
| cached shared feature vs fresh trunk forward | 0.00e+00 |
| **D0 vs D1 segmentation logits** | **bitwise identical (0.00e+00)** |
| **D0 vs D1 argmax mask** | **identical on all 40 images** |

→ D1 is a pure vertex causal intervention.

## 3. Held-out data

Reused EXP013's HELD-OUT set from
`experiments/exp013_heldout_decoder_generalization/results/image_ids.json`
(the exact set EXP014 used; no re-sampling; saved to
`results/image_ids.json` here):

* LIN (20): 000052, 000053, 000055, 000056, 000057, 000058, 000059, 000061,
  000063, 000074, 000075, 000077, 000078, 000079, 000080, 000083, 000087,
  000091, 000092, 000095 (val_real_set[20:40])
* OCC (20): color_00024 … color_00043 (test_real_set[20:40])

Leakage assert: FIT ∩ HELD-OUT = ∅ (re-asserted in code, EXP013 disjoint
flags true). FIT images are never loaded.

D1-state verification gate: loaded states reproduce EXP013's recorded
HELD-OUT direction means exactly (LIN D0 2.4356 / D1 2.5412; OCC D0 6.1564 /
D1 4.9355; tol 0.01°) → the loaded state IS the EXP013 final state.

## 4. Keypoint localization results (pooled: 9 kps × 20 images = 180 / group)

### Q1 — Does D1 lower OCC keypoint error?

| Split | Method | mean px | median px | p75 | p90 | p95 | max | cat imgs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LIN | D0 | 1.226 | 1.111 | 1.746 | 2.153 | 2.632 | 4.02 | 0/20 |
| LIN | D1 | 1.310 | 1.235 | 1.630 | 2.276 | 2.541 | 4.06 | 0/20 |
| OCC | D0 | 47.410 | 3.150 | 5.217 | 59.201 | 450.23 | 508.99 | 2/20 |
| OCC | D1 | 47.139 | 2.143 | 5.219 | 60.935 | 449.90 | 508.99 | 2/20 |

OCC relative change: **median +31.97% improvement** (3.150 → 2.143);
**mean +0.57%** — the mean is dominated by the two shared catastrophic images
(~400–480 px mean error each) and must NOT be used alone (per protocol).

LIN relative change: median −11.2% (D1 slightly worse), consistent with the
known LIN direction degradation (+4.3% in EXP013). Recorded, no action taken.

Keypoint-level catastrophic rates:

| Split | Method | >20 px | >50 px | >100 px |
| --- | --- | ---: | ---: | ---: |
| LIN | D0 / D1 | 0 / 0 | 0 / 0 | 0 / 0 |
| OCC | D0 | 11.11% | 10.00% | 10.00% |
| OCC | D1 | 12.78% | 10.00% | 10.00% |

Image-level catastrophic images (>100 px max error): OCC D0 2/20, D1 2/20
(keypoint-level vs image-level rates are reported separately as required).
No NaN keypoint predictions occurred (nan_count 0 everywhere).

### Q2 — Is the improvement stable in paired per-image analysis?

Per image: mean/median over 9 kps, delta = D1 − D0 (negative = D1 better),
tie if |delta| ≤ 1e-6.

| Split | basis | better | worse | tie | mean paired delta | median paired delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LIN | mean | 5 | 15 | 0 | +0.083 px | +0.162 px |
| LIN | median | 8 | 12 | 0 | — | — |
| OCC | mean | **12** | **7** | 1 | **−0.271 px** | −0.471 px |
| OCC | median | 13 | 6 | 1 | — | — |

OCC: D1 improves the majority of images (12/20 mean-based; 13/20
median-based) with negative paired deltas. The improvement is stable across
images, not a one-image effect. (LIN degrades pairwise, as expected.)

### Q3 — Is the improvement only on non-catastrophic images?

Non-catastrophic image = max keypoint error ≤ 100 px (in that branch).
Common subset = images non-catastrophic in BOTH branches.

| Subset | n | D0 median | D1 median | change |
| --- | ---: | ---: | ---: | ---: |
| OCC common non-cat | 18 | 2.877 | **1.948** | **−32.3%** |
| LIN common non-cat | 20 | 1.111 | 1.235 | +11.2% (worse) |

The two OCC catastrophic images are **the same images in both branches**
(color_00025: D0 399.16 → D1 398.07; color_00027: D0 479.64 → D1 479.64,
**bit-identical voting output**). They are pre-existing voting failures,
untouched by the vertex intervention.

→ **Case A**: D1 improves keypoint localization generally on normal images
(median −32.3% on the 18-image common subset), while the catastrophic voting
failures are unresolved (they are branch-invariant). One caveat: the
sub-catastrophic tail is slightly worse under D1 (OCC >20 px rate 11.1% →
12.8%, p90 59.2 → 60.9 px) while >50/>100 px rates are identical — the gain
is concentrated in the median region, not the tail.

## 5. Q4 — Where did EXP014's pose failure actually occur?

Explanation-only diagnosis; EXP014's `per_image_pose.csv` (standard PnP
outputs from the unmodified baseline evaluator) was reused. **No PnP was run
or modified here.**

~180° flip cases (rotation error > 90°):

| Branch | flips | flip ∩ keypoint-catastrophic |
| --- | ---: | ---: |
| LIN D0 / D1 | 0/20 / 0/20 | 0 |
| OCC D0 | 4/20 {20,21,23,39} | 2 |
| OCC D1 | 14/20 {20,21,22,23,24,29,31,33,34,35,36,37,38,39} | 2 |

Two decisive observations:

1. **Flips occur even where D1 keypoints are better.** E.g. image 35
   (color_00039): keypoint mean error improves by 1.90 px under D1, yet its
   EXP014 pose flipped. Better keypoints do not prevent the standard
   iterative PnP from entering the degenerate ~180° mode.
2. **The 2 shared catastrophic images (color_00025/27) are flipped in BOTH
   branches** — ~400–500 px keypoint errors poison PnP regardless of decoder.

Combined with Q1–Q3 (direction → keypoint transfer is real and stable), the
EXP014 STOP-C is explained by the **keypoint → PnP stage** (PnP fragility +
the pre-existing branch-invariant voting tail), **not** by unstable
direction → keypoint transfer.

## 6. Q5 — Next step

**A. keypoint localization / voting.**

Rationale: (i) the pre-registered GO rule names this direction; (ii) the
remaining actionable defect measured here is the OCC voting heavy tail —
10% of keypoints >50 px in BOTH branches and 2 branch-invariant catastrophic
images — which any downstream PnP will be sensitive to; (iii) the PnP flip
mode is now diagnosed (above), and PnP modification is forbidden by the
project baseline, so the productive lever is upstream vote/keypoint
robustness on the tail, not PnP.

## 7. Decision (pre-registered operationalization)

* OCC pooled median improves: 3.150 → 2.143 ✓
* OCC paired mean-based better > worse: 12 > 7 ✓
* OCC common non-cat subset median improves: 2.877 → 1.948 ✓
  (catastrophic failures do not dominate the conclusion)

→ **GO**: the EXP013/014 direction representation gain causally and stably
improves keypoint localization on held-out images. Next stage (pre-registered):
keypoint localization / voting robustness.

## 8. Limitations

* Only 20 LIN + 20 OCC held-out images; cat-only baseline; single seed.
* Diagnostic measurement only; no training, no architecture, no tuning.
* Keypoint errors measured against GT projected keypoints via the original
  voting pipeline only; no pose metrics were computed in this experiment.
* PnP flip counts are reused from EXP014 (standard PnP, unmodified) and are
  explanatory only.
* No proximity band analysis (out of scope per protocol); no cross-object or
  cross-dataset claims.

---
This experiment is a mechanism diagnosis on the cat-only PVNet baseline and does not establish cross-object generalization.
