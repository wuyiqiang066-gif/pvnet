# EXP013: Held-out Frozen Decoder Generalization Validation

## 1. Research Question

Does the EXP012 frozen-feature cross-channel spatial decoder (D1) generalize
beyond the images used for fitting?

EXP012 was an **in-sample capacity probe**: D1 was fitted and evaluated on the
same fixed 20+20 images. Before building any direction-representation method
on top of it, the +21.1% OCC improvement must be shown to transfer to images
that never entered the decoder fitting. No new architecture, no optimization,
no additional decoders — only D0 vs D1, FIT vs HELD-OUT.

## 2. Relation to EXP012

* EXP012 (in-sample, commit `a53b56e`): D0 OCC 5.121° → D1 OCC 4.040°,
  improvement 21.1%; D1 = D0 + zero-init [Conv3x3(32->32) → LeakyReLU(0.1) →
  Conv1x1(32->18)], 9842 params; judged GO-A.
* EXP013 reuses EXP012's final D1 implementation and fitting protocol
  **verbatim** (same zero-init residual, same loss, Adam 1e-3, 5 epochs,
  pixel minibatch 4096, seed 0 re-seeded per fit, final decoder state).
* This experiment only validates held-out generalization of that result.
  D2 was not tested. No architecture search was performed.

## 3. Data split

FIT set = EXP012's exact images (asserted in-code against
`experiments/exp005_visibility_diagnosis/results/image_ids.json`).

HELD-OUT set = the next 20 images of the same deterministic dataset listings
(LIN `val_real_set[20:40]`; OCC first half of `test_real_set[20:40]`).

| Set | LIN (20) | OCC (20) |
| --- | --- | --- |
| FIT | 000001, 000002, 000004, 000005, 000007, 000011, 000016, 000021, 000025, 000027, 000030, 000032, 000035, 000037, 000040, 000042, 000044, 000048, 000050, 000051 | color_00000–color_00017, color_00022, color_00023 |
| HELD-OUT | 000052, 000053, 000055, 000056, 000057, 000058, 000059, 000061, 000063, 000074, 000075, 000077, 000078, 000079, 000080, 000083, 000087, 000091, 000092, 000095 | color_00024–color_00043 |

Disjointness asserted in code and saved:
`set(fit_ids) ∩ set(heldout_ids) = ∅` for both LIN and OCC
(`results/image_ids.json`, `disjoint_assert: {linemod: true, occ: true}`).

## 4. Protocol

* 199.pth loaded frozen (`requires_grad=False`, no_grad trunk forward);
  backbone / shared feature F = convraw[0:3] / segmentation / GT / preprocessing
  untouched.
* seed 0 everywhere; D1 fitted **only** on the FIT caches (one frozen forward
  per FIT image, offline pixel-minibatch fitting, EXP012 verbatim; init loss
  exactly equals the D0 reference: LIN 0.002869 / OCC 0.008945 — EXP007-trap
  control intact).
* Fitting loop per split: D1-LIN on 20 LIN FIT images, D1-OCC on 20 OCC FIT
  images (EXP012 convention); 5 epochs over all fg pixels, pixel minibatch
  4096, Adam lr=1e-3.
* The FINAL decoder state is used (no checkpoint selection, no early
  stopping, no loss-based model selection). Held-out images never enter
  optimizer, gradients, or any parameter update; after fitting, D1 is frozen
  and the held-out set is only forwarded under no_grad.
* D0 gate BEFORE any fitting: D0 on FIT must reproduce EXP006/012 —
  LIN 2.969 / OCC 5.121, counts 440940/250033. **Passed exactly.**
  D0 on HELD-OUT (recorded, not gated — different images): LIN 2.436°,
  OCC 6.156°, counts 422693/208882.
* Metrics: direction error (deg) on the EXP006/008 pool (visible fg pixels
  with |K-p|≥1px, all 9 kps pooled): mean / median / p90, bad rate >10°,
  endpoint error (unit space); OCC proximity bands 0–5 / 5–20 / 20+ px.

## 5. Results

Core table:

| Split | Decoder | LIN mean° | OCC mean° | LIN median° | OCC median° | LIN p90° | OCC p90° | OCC >10° |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FIT | D0 | 2.969 | 5.121 | 1.699 | 2.822 | 6.046 | 10.901 | 0.1157 |
| FIT | D1 | 2.842 | 4.040 | 1.615 | 2.372 | 5.807 | 8.461 | 0.0758 |
| HELD-OUT | D0 | 2.436 | 6.156 | 1.485 | 3.551 | 5.025 | 12.873 | 0.1460 |
| HELD-OUT | D1 | 2.541 | 4.936 | 1.629 | 2.873 | 5.094 | 9.975 | 0.0997 |

Bad rate >10° / endpoint (unit space):

| Split | LIN D0 | LIN D1 | OCC D0 | OCC D1 |
| --- | ---: | ---: | ---: | ---: |
| FIT | 4.27% / 0.0514 | 3.95% / 0.0492 | 11.57% / 0.0881 | 7.58% / 0.0697 |
| HELD-OUT | 2.63% / 0.0423 | 2.70% / 0.0441 | 14.60% / 0.1053 | 9.97% / 0.0847 |

Pool counts (visible fg, |K-p|≥1px): FIT 440940 (LIN) / 250033 (OCC);
HELD-OUT 422693 (LIN) / 208882 (OCC).

Fitting loss (train only, FIT set): D1-LIN 0.002869 → 0.002766;
D1-OCC 0.008945 → 0.006643 — identical trajectory to EXP012. Per the
pre-registration, this is reported as context only; the success criterion is
held-out direction error, not training loss.

## 6. Generalization improvement

```
FIT OCC improvement      = (5.121 - 4.040) / 5.121 = +21.11%   (EXP012 ref: +21.11%)
HELD-OUT OCC improvement = (6.156 - 4.936) / 6.156 = +19.83%
generalization gap       = +21.11% - +19.83%       = +1.28 pp
```

The FIT result reproduces EXP012 exactly (D1 LIN 2.8416 vs EXP012 2.8416;
D1 OCC 4.0399 vs 4.0399), confirming the fitting protocol is deterministic
and unchanged. The held-out improvement loses only 1.28 percentage points of
the in-sample improvement.

Secondary observations, reported as-is:

* LIN held-out degrades slightly: 2.436° → 2.541° (+4.3% error; median +9.7%,
  p90 +1.4%). The same sign pattern as EXP009 (OCC gain traded against a
  small LIN cost), but much smaller in magnitude, and LIN was not part of the
  pre-registered GO/STOP criterion.
* Held-out D0 baselines differ from FIT D0 (LIN 2.436 vs 2.969; OCC 6.156 vs
  5.121) — expected, since these are different images; the held-out OCC set
  is harder (more/bigger occluded regions), which is also why absolute errors
  are not comparable across sets, only within-set D0→D1 deltas.

## 7. OCC proximity (0–5 / 5–20 / 20+ px from occluded region)

| OCC distance | FIT D0 | FIT D1 | HELD-OUT D0 | HELD-OUT D1 |
| --- | ---: | ---: | ---: | ---: |
| 0–5 px | 7.653 | 6.680 (−12.7%) | 11.428 | 11.750 (+2.8%) |
| 5–20 px | 4.921 | 4.657 (−5.4%) | 5.903 | 5.885 (−0.3%) |
| 20+ px | 3.522 | 3.667 (+4.1%) | 5.745 | 4.654 (−19.0%) |

Band coverage: FIT 9 / 9 / 7 images; HELD-OUT 12 / 12 / 7 images
(entries 12429 / 40324 / 30306 FIT; 13818 / 34637 / 48737 HELD-OUT).

Reported as-is: on HELD-OUT the improvement is **not** concentrated at the
0–5 px occlusion-boundary band (flat to slightly worse there) but comes from
the 20+ px band (−19.0%), roughly the reverse of the FIT pattern. Under the
strict instruction not to force interpretations, we note only that (a) the
overall held-out improvement is large and consistent with GO-B, and (b) the
per-band pattern differs between FIT and HELD-OUT, so the occlusion-boundary
mechanism attribution from EXP005/006 should be treated as unconfirmed for
this decoder's generalization behavior rather than reinforced by it.

## 8. GO / STOP (pre-registered, thresholds unchanged)

* HELD-OUT OCC improvement = **+19.83%** ≥ 15% → **GO-B**

> EXP012's direction improvement not only exists on the fitting images but
> also generalizes to images that did not participate in decoder fitting.
> Only now is discussing a real direction-representation method admissible.

(For completeness: STOP-B would have required < 5%; BORDERLINE band 5–15%
does not apply.)

## 9. Interpretation

Only what the data supports:

1. The EXP012 in-sample capacity improvement transfers essentially intact to
   unseen images (+19.83% vs +21.11%, gap +1.28 pp) under the identical
   fitting protocol and budget (5 epochs, 9842 params, <9 s GPU).
2. The transfer is OCC-specific in sign structure: OCC improves strongly on
   held-out data while LIN degrades slightly (+4.3%). No LIN-side collapse.
3. The per-band pattern does not confirm the occlusion-boundary mechanism
   story for the generalized decoder (improvement at 20+ px, not 0–5 px);
   this observation is descriptive, from the same run, with no new
   experiment performed.
4. No architecture conclusions beyond D1 are licensed by this experiment; D2
   was not tested and no method was designed or optimized here.

This is still a cat-only diagnosis and does not establish cross-object
generalization.

---
This experiment is a mechanism diagnosis on the cat-only PVNet baseline and does not establish cross-object generalization.
