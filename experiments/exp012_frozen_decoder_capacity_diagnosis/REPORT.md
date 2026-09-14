# EXP012: Frozen Shared-Feature Decoder Capacity Diagnosis

## 1. Research question

Is the occlusion-robust keypoint-direction information still present in the
frozen 32-ch shared feature F of the cat-only PVNet baseline (199.pth), with
the original 1x1 vertex head merely a readout-capacity bottleneck (hypothesis
A) — or has it already been lost at feature-extraction time (hypothesis B)?

## 2. Motivation from EXP006/008/009/010/011

- EXP006: direction prediction is the main degraded mechanism
  (LIN 2.969 deg vs OCC 5.121 deg).
- EXP008: with GT directions, local r=3 reaches 0.362/0.569 deg and global
  aggregation 0.001 deg — a large headroom exists in principle.
- EXP009: frozen trunk + tiny learned local residual: LIN improves 5.5% but
  OCC degrades 25.7% (STOP; domain-wide miscalibration).
- EXP010: oracle validity-aware local selection is 3.2x WORSE than blind
  local aggregation (STOP).
- EXP011: object-level aggregation of the predicted rays improves OCC only
  3.5% (STOP-object-level); with GT rays the same aggregation recovers ~0 deg.

Conclusion so far: the information exists (EXP008), but no post-hoc geometric
selection/aggregation of the current predictions recovers it (EXP009/010/011).
EXP012 tests the remaining hypothesis: the information is still in F, and the
1x1 head is the readout bottleneck.

## 3. Experimental protocol

- Data: the fixed EXP005-011 20 LIN + 20 OCC cat images (IDs asserted in-run
  against EXP005 image_ids.json); seed 0; checkpoint 199.pth; PVNet entirely
  frozen (requires_grad=False, no_grad trunk); total GPU runtime 13.5 s.
- Offline feature cache: one frozen forward per image stores
  F = convraw[0:3](cat(fm, x)) (1,32,H,W) — the identical shared feature used
  by EXP009 — plus the dataset GT vertex targets and fg vertex weights.
- Capacity-probe protocol: D1/D2 are fitted IN-SAMPLE per split (D1-LIN
  fitted on the 20 LIN images and evaluated on the same 20 LIN; D1-OCC fitted
  on the 20 OCC images and evaluated on the same 20 OCC; likewise D2).
  In-sample fitting deliberately removes generalization and domain-shift
  confounds (the EXP009 lesson): the experiment asks ONLY whether more readout
  capacity can extract more direction information from F, not whether the
  decoders generalize.
- EXP007-trap control: D1/D2 are defined as D0(F) plus a zero-initialized
  branch, so at initialization each decoder is EXACTLY D0 (verified: init
  loss = D0 loss = 0.002869 LIN / 0.008945 OCC). In-sample loss can only
  decrease from the baseline; optimization budget cannot manufacture a
  spurious degradation.
- Fitting: 5 epochs over all fg pixels (pixel minibatch 4096, Adam lr=1e-3,
  seed 0 re-seeded per fit). Loss = lib.utils.net_utils.smooth_l1_loss on
  unit-normalized predictions vs the dataset GT vertex targets with fg
  weights (EXP009 direction-loss convention; GT supervision definition
  unchanged). Backbone, original vertex head, segmentation path, RANSAC/PnP,
  data splits and 199.pth untouched.
- D0 gate: D0 must reproduce EXP006 before any fitting; the run aborts
  otherwise. Gate passed exactly: LIN 2.969 / OCC 5.121, pooled counts
  440940 / 250033 (identical to EXP006/011).
- Evaluation (direction representation only): mean/median/p90 deg, bad rate
  >10 deg, endpoint error (unit space) on the EXP006 pool (visible fg pixels
  with |K-p|>=1px, all 9 kps pooled); OCC proximity bands 0-5 / 5-20 / 20+ px
  from the occluded region (EXP005/011 amodal-mask definitions); unit-field
  roughness as a smoothing check. No ADD/ADD-S/PnP/RANSAC-pose; no
  keypoint-specific analysis.

## 4. D0/D1/D2 definitions

- D0: original frozen 1x1 vertex head convraw[3] (32->20, vertex channels),
  no training. Strict baseline.
- D1: D0(F) + zero-init branch [Conv3x3(32->32) -> LeakyReLU(0.1) ->
  Conv1x1(32->18)]; 9,842 parameters.
- D2: D0(F) + zero-init branch [Conv3x3 -> LeakyReLU(0.1) -> Conv3x3 ->
  LeakyReLU(0.1) -> Conv1x1(32->18)]; 19,090 parameters.
- LeakyReLU(0.1) is the network's own activation; without a nonlinearity the
  deeper branch would collapse to a linear map. No BN / SE / attention /
  deformable conv / multi-scale / transformer components.

**D1/D2 are diagnostic decoders, not proposed methods.**

## 5. Exact results

| Decoder | LIN mean° | OCC mean° | LIN median° | OCC median° | LIN p90° | OCC p90° | OCC >10° |
|---|---:|---:|---:|---:|---:|---:|---:|
| D0 Original 1×1 | 2.969 | 5.121 | 1.699 | 2.822 | 6.046 | 10.901 | 11.57% |
| D1 3×3→1×1 | 2.842 | 4.040 | 1.615 | 2.372 | 5.807 | 8.461 | 7.58% |
| D2 3×3→3×3→1×1 | 2.838 | 4.051 | 1.600 | 2.253 | 5.819 | 8.230 | 7.32% |

Supplementary metrics (same pool):

- Endpoint error (unit space): LIN 0.0514 / 0.0492 / 0.0491
  (D0/D1/D2); OCC 0.0881 / 0.0697 / 0.0698 — OCC endpoint improves 20.9%.
- LIN bad rate >10°: 4.27% / 3.95% / 4.00% (D0/D1/D2).
- Relative mean improvement: OCC D1 +21.1%, D2 +20.9%; LIN D1 +4.3%,
  D2 +4.4%.
- In-sample fit quality (EXP009-normalized full-image loss): LIN
  0.002869 → 0.002728 (D1) / 0.002739 (D2); OCC 0.008945 → 0.006191 (D1) /
  0.006800 (D2). The OCC loss was still decreasing at the 5-epoch cap.

## 6. OCC proximity results

| OCC distance | D0 | D1 | D2 |
|---|---:|---:|---:|
| 0–5 px | 7.653 | 6.680 | 6.779 |
| 5–20 px | 4.921 | 4.657 | 4.412 |
| 20+ px | 3.522 | 3.667 | 3.350 |

The improvement is largest in the 0–5 px band nearest the occluded region
(D1 -12.7%, D2 -11.4%), smaller at 5–20 px (-5.4% / -10.3%), and small/mixed
at 20+ px (D1 +4.1%, D2 -4.9%).

## 7. GO/STOP decision

**GO-A** (pre-registered: GO iff min(OCC D1, OCC D2) <= 4.097 deg, i.e.
>=20% relative improvement). Best: D1, OCC mean 4.040 deg = 21.1% relative
improvement. D2 independently crosses the threshold (4.051 deg, 20.9%), so
the decision does not hinge on one architecture. The margin is thin
(4.040 vs 4.097) and is reported as such.

Interpretation per the pre-registration: the frozen shared feature does
contain exploitable occlusion-robust direction information; the original 1x1
vertex head / readout capacity is (at least partly) the bottleneck. Only now
is it admissible to discuss a new direction-representation method.

## 8. Mechanistic interpretation

- Hypothesis A is supported in-sample: a single 3x3 context layer (9,842
  params) fitted on cached frozen features recovers >=20% of the OCC
  direction error, while LIN improves only ~4% — the extracted headroom is
  OCC-specific, consistent with occlusion being the mechanism identified in
  EXP005/006.
- Not a smoothing effect: unit-field roughness is essentially unchanged
  (OCC 0.0343 -> 0.0347 / 0.0343 for D1/D2); the improvement is a genuine
  change in the decoded directions, not field smoothing.
- The bad-direction rate (>10 deg) drops from 11.6% to 7.3-7.6% (-34%
  relative): the gain is concentrated on the worst pixels, not only the
  typical ones.
- Improvement is concentrated near the occlusion boundary (0-5 px band),
  exactly where EXP005/006 localized the degradation.
- D2 ≈ D1: adding a second 3x3 layer yields no additional mean improvement
  (and slightly better median/p90 but slightly worse mean). One 3x3 context
  layer already saturates what this fitting protocol extracts; depth alone is
  not the lever.
- Reconciliation with EXP009: its depthwise (spatial-only, no cross-channel)
  914-param residual, trained on LIN only, degraded OCC cross-domain. Here a
  full cross-channel 3x3 branch, fitted in-sample per split, improves OCC.
  The contrast points to cross-channel feature mixing and domain shift — not
  absence of information — as EXP009's failure mode.
- Reconciliation with EXP010/011: hand-crafted geometric selection and
  aggregation of the current 1x1 predictions fail, yet a learned 3x3 readout
  of F succeeds in-sample. The information is present in F but is accessible
  only through a learned readout, not through post-hoc geometric processing
  of the 1x1 outputs.

## 9. Limitations

- In-sample capacity probe: D1/D2 are fitted and evaluated on the same fixed
  20 images per split. The results establish information presence in F (an
  upper-bound probe) and make no generalization claim of any kind.
- 5-epoch cap: the in-sample OCC loss was still decreasing at the cap, so
  21.1% is plausibly a lower bound on what is extractable in-sample;
  conversely the thin margin (4.040 vs 4.097) makes the GO sensitive to
  fitting details. Single seed, per protocol.
- The decoders are residual on top of the frozen 1x1 head; a full-replacement
  decoder trained to its optimum could behave differently.
- Cat-only, 20+20 images, direction error pooled over 9 keypoints; no
  keypoint-specific analysis by design; no ADD/ADD-S/PnP evaluation — this is
  a representation diagnosis, not a pose-performance result.

This experiment is a mechanism diagnosis on the cat-only PVNet baseline and
does not establish cross-object generalization.
