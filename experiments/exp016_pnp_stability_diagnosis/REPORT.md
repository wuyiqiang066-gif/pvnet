# EXP016: PnP Stability Diagnosis

## 1. Research Question

> Why does D1 — which EXP015 proved improves OCC keypoint localization
> (median 3.150 → 2.143 px; paired 12 better / 7 worse) — produce MORE PnP
> pose flips in EXP014 (OCC D0 4/20 vs D1 14/20, rotation error > 90°)?

Diagnosis only. The FULL original pipeline is executed and observed
(image → segmentation → vertex → original PVNet RANSAC voting → predicted
keypoints → original standard PnP → pose). **Nothing is modified**: no
training, no D0/D1 change, no voting/RANSAC change, no PnP solver or
threshold change, no intrinsics/GT change, no data expansion.

## 2. Experimental isolation (all asserted before evaluation)

* Image IDs loaded from EXP015 `results/image_ids.json`, asserted ==
  EXP013/EXP014 held-out sets (20 LIN val_real_set[20:40] + 20 OCC
  test_real_set[20:40]); FIT ∩ HELD-OUT = ∅.
* D1 loaded from EXP014 persisted states; direction gate reproduces EXP013
  exactly (LIN 2.4356/2.5412, OCC 6.1564/4.9355 deg) → loaded state IS the
  EXP013 final state. No re-fitting.
* Identity checks: D0 branch vs original `net.forward` pipeline — seg/vertex
  diff 0.00e+00, masks identical, voted-keypoint diff 0.00e+00; **D0 vs D1
  segmentation logits bitwise identical, argmax masks identical on all 40
  images** → pure vertex causal intervention.
* Voting: original `ransac_voting_layer_v3`, round_hyp_num=128,
  inlier_thresh=0.99, max_num=100 (asserted), seed 0 re-applied before every
  call. PnP: original `Evaluator` standard PnP path exactly as EXP014.
* Internal vote-count statistics are not exposed by the original voting code;
  per spec they were NOT obtained by modifying voting.

Definitions (reused, not re-invented): flip = rotation error > 90° (also
>120°/>150°/>170° recorded); voting_catastrophic = max keypoint error
> 100 px (EXP015 definition); "low reprojection" = < 5 px (the project's own
official 2D-projection success threshold — no new threshold invented).

## 3. Analysis A — flip rates and reproduction (Q1)

| Split | Method | >90° | >120° | >150° | >170° |
| --- | --- | ---: | ---: | ---: | ---: |
| LIN | D0 | 0/20 | 0/20 | 0/20 | 0/20 |
| LIN | D1 | 0/20 | 0/20 | 0/20 | 0/20 |
| OCC | D0 | **4/20** | 3/20 | 2/20 | 2/20 |
| OCC | D1 | **14/20** | 13/20 | 12/20 | 10/20 |

**EXP015's stored flip counts (OCC D0 4/20, D1 14/20; LIN 0/20) are exactly
reproduced** by a fresh end-to-end run (deterministic seeded voting +
deterministic PnP). The flip increase is real and stable, not a stored-artifact.

## 4. Analysis B — keypoint improvement vs flip (Q2, the key result)

Per OCC image: delta_kp = D1 − D0 (mean-based primary; median-based
consistent), classified improved / worsened / tie (|Δ| ≤ 1e-6).

|  | D1 PnP stable | D1 PnP flip | total |
| --- | ---: | ---: | ---: |
| D1 KP improved | 3 | **9** | 12 |
| D1 KP worsened | 3 | 4 | 7 |
| tie | 0 | 1 | 1 |

* **P(flip | KP improved) = 9/12 = 75.0%**
* P(flip | KP worsened) = 4/7 = 57.1%

**Yes — among images where D1 improved the keypoints, PnP still flips 75% of
the time.** The median-based table shows the same pattern (improved 9 flip /
4 stable). Keypoint improvement does not prevent, and is weakly positively
associated with, flipping (Spearman(improvement, D1 rot error) = +0.63).

Striking sub-cluster: images 33–39 (color_00037–00043) — D1 improves KP mean
from ~3.0–3.5 px to ~1.1–2.1 px, yet rotation error is 178.3–180.0° with mean
reprojection error 0.94–1.66 px.

## 5. Analysis C — is flip caused by large keypoint error? (Q5 part 1)

Median of per-image KP errors, grouped by flip status:

| Group | n | median of KP median (px) | median of KP mean (px) |
| --- | ---: | ---: | ---: |
| OCC D1 flip | 14 | **1.880** | — |
| OCC D1 stable | 6 | **1.991** | — |
| OCC D0 flip | 4 | 202.046 | — |
| OCC D0 stable | 16 | 2.855 | — |

**Flip images under D1 have keypoint errors just as small as stable images**
(1.880 vs 1.991 px). PnP flip is NOT caused by globally worse keypoint
localization. (Under D0 the flip group's 202 px median is driven by the two
catastrophic voting images — a different failure mode.)

Spearman (OCC, D1): KP mean vs rotation error = −0.47; improvement vs
rotation error = +0.63. If anything, better keypoints ↔ larger rotation error.

## 6. Analysis D/E — reprojection ambiguity (Q4)

Mean 2D reprojection error (px) of the predicted pose, by group:

| Group | mean | median | p90 |
| --- | ---: | ---: | ---: |
| OCC D0 all | 164.4 | 2.698 | 49.5 |
| OCC D0 flip (n=4) | 810.8 | 205.94 | 2100.5 |
| OCC D0 stable (n=16) | 2.83 | 2.590 | 4.17 |
| OCC D1 all | 2167.3 | 1.842 | 643.1 |
| OCC D1 flip (n=14) | 3095.2 | **1.587** | 2100.2 |
| OCC D1 stable (n=6) | 2.26 | 2.033 | 3.26 |

* **Case 1 (low reprojection + large rotation) is the dominant D1 flip
  signature: 9/14 (64.3%) flipped poses have reprojection < 5 px**; among the
  12 non-catastrophic D1 flips it is 9/12 (75%). D0: 1/4 (25%).
* The D1 flip group's median reprojection (1.587 px) is LOWER than the D1
  stable group's (2.033 px) — flipped poses are image-space consistent, often
  MORE consistent than correct ones.
* Non-catastrophic D1 flip rotation errors are 161.5–180.0° (11 of 12 in
  171–180°) — a degenerate ~180° mode, with reprojection ~1–2.6 px for 9 of
  them. One wild outlier (color_00026) flips with 40072 px reprojection
  (a generic PnP blow-up, not ambiguity).
* Spearman(D1 reprojection, rotation error) = −0.52: larger rotation errors
  associate with SMALLER reprojection errors — the ambiguity signature in
  correlation form.

Mechanistic reading: the flipped ~180° hypothesis is (near-)equivalent in
image space for this object/keypoint configuration, and the original
iterative PnP selects it on 12/18 non-catastrophic OCC images under D1.
Which basin is entered depends on the fine GEOMETRY of the keypoint error
pattern, not its magnitude.

## 7. Analysis F — voting catastrophic vs PnP flip separation (Q3)

voting_catastrophic = max KP error > 100 px (EXP015 definition); the two
catastrophic images are again color_00025/27 in both branches.

| Branch | cat imgs | flip among cat | flip among NON-cat | non-cat flip rate |
| --- | ---: | ---: | ---: | ---: |
| OCC D0 | 2/20 | 2/2 | 2/18 (ids 20, 39) | 11.1% |
| OCC D1 | 2/20 | 2/2 | **12/18** (20, 22, 24, 29, 31, 33–39) | **66.7%** |

**Flip is NOT mainly caused by the catastrophic voting images**: 12 of D1's
14 flips survive after removing them. Voting catastrophic failure and PnP
flip are two INDEPENDENT failure modes.

## 8. Non-catastrophic OCC comparison (18 common images)

| Metric | D0 | D1 |
| --- | ---: | ---: |
| KP median (median over images) | 2.855 px | **1.861 px** |
| KP mean (median over images) | 2.936 px | **1.981 px** |
| Rotation error (median) | 9.38° | **174.06°** |
| Flip rate (>90°) | 11.1% | **66.7%** |
| Reprojection (median) | 2.654 px | **1.644 px** |
| ADD pass fraction (diagnosis only) | 0.056 | 0.000 |
| ADD mean dist (median, mm) | 44.7 | 2399.1 |

The paradox in one row: D1 has BETTER keypoints, BETTER reprojection, and
3.6× MORE flips with ~180° rotation error. LIN shows none of this (0 flips
both branches; rot median D0 1.97° / D1 2.48°; ADD pass 0.65 / 0.70,
matching EXP014).

## 9. Answers to the six questions

* **Q1**: Yes. Flip counts reproduce exactly: OCC D0 4/20, D1 14/20 (>90°);
  LIN 0/20 both. At stricter thresholds the gap persists (D1 13/12/10 vs
  D0 3/2/2 at >120°/>150°/>170°).
* **Q2**: Yes. P(flip | KP improved) = 9/12 = 75%. Keypoint improvement does
  not prevent flipping.
* **Q3**: No. Only 2/14 D1 flips are catastrophic-voting images; 12/18
  non-catastrophic images still flip (66.7%). Two independent failure modes.
* **Q4**: Yes. 9/14 (64.3%) of D1 flipped poses have reprojection < 5 px
  (official 2D threshold), with rotation errors of ~178–180°. The D1 flip
  group's median reprojection (1.59 px) is lower than its stable group's
  (2.03 px) — low reprojection / high rotation error ambiguity is the
  dominant signature.
* **Q5**: No magnitude relation. Flip images' KP median (1.88 px) ≈ stable
  images' (1.99 px); improvement correlates POSITIVELY with rotation error
  (Spearman +0.63). The flip depends on the error GEOMETRY of the keypoint
  set interacting with the iterative PnP basin selection, not on error size.
* **Q6**: **B. PnP geometric stability** (per the pre-registered GO-B gate).

## 10. Decision (pre-registered gates)

* reproduction_ok ✓ (exact)
* D1 majority KP improved ✓ (12 > 7)
* D1 non-catastrophic flip rate 66.7% ≥ 25% ✓
* ≥1 D1 flip survives excluding catastrophic images ✓ (12)
* ≥30% of D1 flipped poses have reprojection < 5 px ✓ (64.3%; 75% among
  non-cat flips)

→ **GO-B**: there is a genuine **keypoint accuracy → pose stability mismatch**
in the original standard PnP: image-space-consistent ~180° hypotheses absorb
images whose keypoints D1 improved. Next stage (per gate): research PnP/pose
hypothesis stability WITHOUT changing PVNet representation — and only via a
new low-cost diagnosis first; no method design.

## 11. Limitations

* 20 LIN + 20 OCC held-out images, cat-only baseline, single seed; counts
  and percentages only, no significance testing (per protocol).
* Internal voting statistics (vote counts) not recorded — the original
  voting code does not expose them and modifying it is forbidden.
* "Low reprojection" uses the project's existing 5 px 2D-projection success
  threshold; no new threshold was introduced.
* ADD/2D/5cm5deg are reported for diagnosis only; nothing was tuned on them.
* No fix attempted anywhere; all pipeline components untouched.
* The exact basin-selection mechanism inside SOLVEPNP_ITERATIVE (initial
  guess, correspondence ordering via Farthest 3D points) is only inferred
  from input–output behavior here; a dedicated diagnosis would be the GO-B
  next step.

---
This experiment is a mechanism diagnosis on the cat-only PVNet baseline and does not establish cross-object generalization.
