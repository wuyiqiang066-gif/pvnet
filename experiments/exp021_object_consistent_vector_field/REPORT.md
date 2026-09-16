# EXP021: Object-Consistent Vector Field (OCVF) — One-Shot Minimal Hypothesis Test

## 1. Research Question

Does explicitly modeling a **shared object-level latent state** that jointly
constrains all K keypoint direction fields (cross-keypoint structural
coupling) recover OCC direction error where EXP009/010/011 post-hoc
aggregations and EXP012's keypoint-independent decoder could not, under the
EXP012 in-sample fitting protocol?

## 2. Hypothesis

PVNet regresses the K keypoint direction fields approximately independently
per pixel. The K fields actually share one object-level geometric state.
Making that state explicit and SHARED across the K readouts reduces
correlated direction errors under occlusion.

The mechanism under test is **cross-keypoint structural coupling** — not
reliability weighting, not vote repair, not a larger receptive field, not
global averaging, not attention.

## 3. Why EXP001–020 motivate this hypothesis (numbers re-read from the repository)

- EXP005: under OCC, equal-count counterfactual shows the deficit is not
  just fewer visible pixels (LIN 1.61 px vs OCC 2.90 px endpoint at matched
  count, 1.8x).
- EXP006: the degraded axis is the **direction field itself** — mean
  direction error LIN 2.969° vs OCC 5.121° (unit space, EXP006 pool);
  magnitude scaling is not the issue; supervision is unit vectors.
- EXP008: with oracle context, global aggregation reaches 0.001° and local
  r=3 reaches 0.569° on OCC — the information is present in the frozen
  feature; the failure is representational.
- EXP009: frozen trunk + tiny learned local residual: LIN +5.5% but OCC
  **−25.7%** (STOP).
- EXP010: oracle validity-aware local selection is 3.2x **worse** than blind
  local aggregation (STOP).
- EXP011: object-level aggregation of the *predicted* rays improves OCC only
  3.5% (STOP); with GT rays the same aggregation recovers ~0°.
- EXP012: frozen shared feature + stronger spatial decoder (D1, kp-
  independent 3x3→1x1 residual, 9842 params) reaches in-sample OCC +21.1%
  (5.121°→4.040°) — a decoder-level capacity effect, not a new
  representation.
- EXP013: D1 holds on held-out images (+19.83%) but remains a readout fix.
- EXP014: direction gains do not necessarily transfer to pose on cat (PnP
  ambiguity), which is why this experiment judges representation metrics
  (direction, consistency), not ADD(-S).
- EXP019/020: keypoint residual magnitude is the dominant driver of PnP
  ambiguity; reducing direction error at the source remains the right target.

Gap: every intervention so far was either per-pixel/per-keypoint independent
(EXP009/012/013) or post-hoc geometric aggregation of already-predicted rays
(EXP010/011). None made the object-level state an explicit, shared,
learned constraint that all K fields must satisfy simultaneously.

## 4. Exact mathematical formulation

Let F ∈ R^(32×H×W) be the frozen shared feature, seg the predicted
segmentation, v0 the original frozen vertex field (unit directions), K = 9.

1. Foreground-masked pooling: `z0 = mean_{p ∈ fg} F(:,p)`;
   `z = W_z z0 + b_z` (Linear 32→64). z is derived from the **current
   object's pixel evidence only**.
2. Shared configuration hypothesis: `ĉ = (W_c z + b_c) reshaped to (K,2)`,
   with W_c zero-init and b_c = image center (320,240) — ĉ starts at the
   image center and must learn the configuration from z.
3. Key-level context (shared MLP across keys):
   `c_k = MLP_fuse([q_k ; z])`, q_k learnable 64-d queries (9 of them).
4. Direction residual (conv weights SHARED across the K keys — the same
   reader sees [F ; c_k] for every k):
   `Δv_k(p) = W_out LeakyReLU(W_in [F(p) ; c_k] + b_in) + b_out`,
   W_out zero-init ⇒ Δv ≡ 0 at init (EXP007-trap control: init = D0
   exactly).
5. Refined field (task-spec form):
   `v'_k(p) = normalize(v0_k(p) + Δv_k(p))`.

**Cross-keypoint coupling (the scientifically distinct part):** ĉ is a
single (K,2) configuration predicted from the single shared z; every field
k receives its context c_k from the same z through the same fuse MLP; and
the residual reader weights are shared across k. The K fields cannot
drift independently — they are all written by one object-state-dependent
readout.

**Losses** (weights 1.0, fixed before the run):

- `L_dir` = original PVNet smooth_l1 direction supervision (EXP012 verbatim:
  `smooth_l1_loss(normalize(v0+ΔV), GT vertex, fg weights, normalize=True)`).
  GT used only as training TARGET.
- `L_conf = (1/D) Σ_k ‖ĉ_k − gt_kp_k‖`, D = √(H²+W²). Trains ĉ toward the
  GT keypoint configuration (TARGET, never input).
- `L_struct = mean over fg pixels p and k with ‖gt_kp_k − p‖ ≥ 1px AND
  ‖ĉ_k − p‖ ≥ 1px of [1 − cos(v'_k(p), normalize(ĉ_k − p))]`.
  This is the structural coupling term: all 9 fields are pulled toward ONE
  configuration hypothesis predicted from the shared latent. Exactly one
  structural loss, as required.

## 5. Architecture

22,612 parameters per split instance (OCVF-LIN and OCVF-OCC fitted
independently, EXP012 D1-LIN/D1-OCC convention). No Transformer, no
attention, no graph network. Trunk (Resnet18_8s), segmentation head, and the
original 1x1 vertex head are all FROZEN; only the module above trains.
Voting and PnP are the original untouched implementations
(`ransac_voting_layer_v3`, `cv2.solvePnP(SOLVEPNP_ITERATIVE)` legacy path
not invoked — pose was not evaluated, see §10).

## 6. No-GT-input verification (G4)

Runtime-verified in-run, before any training:

- Forward signature introspection: the module receives only `feat` and
  `seg`.
- Corrupted-GT determinism test: replacing every GT tensor with garbage
  changes **nothing** in the forward outputs (dvert diff 0.00e+00, chat diff
  0.00e+00).

G4 PASSED. GT keypoints enter only as loss TARGETS (allowed by the task
spec), never as inputs. No GT visibility, no GT pose, no GT directions.

## 7. Experimental protocol

- Branch `exp021_object_consistent_vector_field` off `exp020_*` @1be1f0f;
  no existing experiment touched.
- Data: cat only, the exact EXP005/006 deterministic pools — 20 LIN
  (`000001…000051` set) + 20 OCC (`color_00000–17,22,23` set); IDs asserted
  in-run against `exp005_visibility_diagnosis/results/image_ids.json` and
  saved to `results/`-adjacent `image_ids.json`. Seed 0.
- Conditions: **A** = frozen original vertex head (no training, task §8),
  **B** = A + OCVF refinement, fitted with the EXP012 budget: 5 epochs,
  Adam 1e-3, seed 0 per fit, in-sample per split. Identical images, voting,
  PnP, metrics for A and B.
- Training is image-level (batch=1, 20 steps/epoch) because the module needs
  whole-image masked pooling for z (deviation D2, pre-registered).
- Gates run BEFORE training (all PASSED):
  - G1: A reproduces EXP006 — LIN 2.969° / OCC 5.121°, pooled counts
    440940 / 250033, exact.
  - G2: A init L_dir equals EXP012's recorded values — 0.002869 (LIN) /
    0.008945 (OCC), exact to 6 decimals.
  - G3: zero-init identity — Δv ≡ 0 exactly; B's voting at init is
    bit-identical to A's (max kp diff 0.00e+00 px).
  - G4: no-GT-input (§6).

**Pre-registered deviation record:**

- D1: A is evaluated frozen per task §8 ("不要重新训练 baseline"); "identical
  budget" means B gets the EXP012 5-epoch budget and A gets none.
- D2: image-level batching (above).
- D3: two module instances, one per split (EXP012 convention).
- Run-1 abort: the original G3 compared voting on raw v0 vs unit-normalized
  v0 with a 1e-3 px tolerance and failed at 3.63e-02 px. Diagnosis: voting
  kernels are scale-invariant in exact arithmetic (line-normal form), but
  float32 rounding amplified through RANSAC discrete selection gives
  ~0.036 px — a **gate-design flaw**, hit before any training. Fix: feed B's
  RAW field (v0+ΔV) to voting (bit-identical at init; scale invariance
  verified in-run); normalization retained for direction metrics/losses.
  Run 2 restarted from scratch; no loss/architecture/epoch change (task §16
  allows one implementation-bug fix).
- G5 reference error (post-hoc, recorded honestly): the pre-registered
  config cited EXP015's LIN 1.111 / OCC 3.150 px as A's voting reference,
  but EXP015 measured those on the **HELD-OUT** set, never on the FIT pool;
  and G5 was not implemented as an in-run assertion. A's FIT-pool voting
  medians (LIN 1.2987 / OCC 2.3900 px) are the first recorded FIT-pool
  values; they are plausible (the FIT LIN pool is harder: direction 2.969°
  vs held-out 2.436°). This has **no effect on the decision** (see §14).

## 8. Baseline

A is the verified 199.pth baseline, frozen. Gate-exact reproduction of
EXP006/EXP012 references (G1/G2 above). Baseline FIT-pool summary:

| Split | dir mean° | dir median° | dir p90° | >10° rate | M6 median px | voting pooled median px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LIN | 2.9686 | 1.6994 | 6.0459 | 0.0427 | 0.4442 | 1.2987 |
| OCC | 5.1211 | 2.8224 | 10.9012 | 0.1157 | 0.4320 | 2.3900 |

## 9. OCVF

One minimal implementation, as specified in §4/§5. The module genuinely
shares a single object latent z across all 9 keys (shared fuse MLP, shared
residual reader, single configuration hypothesis ĉ). It is not 9 independent
context heads.

## 10. Training behavior

5 epochs × 2 splits, image-level, 48.8 s total. Pre-registered hard cap
respected; no retries beyond the single run-1 gate fix.

| Split | epoch 1 (L_dir/L_conf/L_struct) | epoch 5 (L_dir/L_conf/L_struct) |
| --- | --- | --- |
| LIN | 0.0846 / 0.1841 / 0.9040 | 0.3843 / 0.1998 / 0.2787 |
| OCC | 0.1163 / 0.2782 / 0.7042 | 0.3077 / 0.2797 / 0.2758 |

Observed dynamics (descriptive, not an excuse):

- L_struct falls (0.90→0.28) but L_dir **rises sharply** (0.085→0.39 within
  one epoch and stays there) — the optimizer buys structural consistency at
  the price of direction accuracy.
- L_conf is nearly flat (0.184→0.200 LIN; 0.278→0.280 OCC): the
  zero-weight+center-bias configuration head did **not** learn an accurate
  ĉ within 100 steps/epoch (configuration errors of order 10² px).
- Scale analysis at init: L_dir ≈ 0.003 vs L_struct ≈ 0.9 (~300x). Under
  Adam's per-parameter normalization the large-raw-magnitude term dominates
  the effective step; the shared configuration pull therefore overrode the
  direction supervision. Loss weights were fixed pre-registration and the
  task forbids re-tuning them, so this is recorded as the observed failure
  mode, not repaired.

## 11. Direction results (M1–M5)

| Split | Cond | mean° | median° | p90° | >10° rate |
| --- | --- | ---: | ---: | ---: | ---: |
| LIN | A | 2.9686 | 1.6994 | 6.0459 | 0.0427 |
| LIN | B | **73.9558** | 64.8255 | 152.8605 | 0.9040 |
| OCC | A | 5.1211 | 2.8224 | 10.9012 | 0.1157 |
| OCC | B | **67.1093** | 60.1780 | 125.9895 | 0.9461 |

- C1 (OCC mean improvement ≥10%): **FAIL** — OCC degraded −1210.4%
  (5.12°→67.11°).
- C2 (LIN degradation ≤5%): **FAIL** — LIN degraded +2391.2%
  (2.97°→73.96°).
- ~90–95% of foreground pixels in B exceed 10° error: the fields point in
  essentially arbitrary directions, consistent with being dragged toward an
  inaccurate shared configuration.

## 12. Structural consistency results (M6)

Metric (GT-free, pre-registered): for each (image, k), take the field's own
rays on visible-fg pixels, compute the closed-form least-squares ray
intersection point, and measure the perpendicular residual of each ray to
that point; per-image median over 9 kps; per-split median over 20 images.
Degenerate cases (n<50 or cond>1e12): 0 across all conditions.

| Split | A median px | B median px |
| --- | ---: | ---: |
| LIN | 0.4442 | 5.6418 |
| OCC | 0.4320 | 5.3541 |

- C3 (OCVF consistency clearly better than baseline): **FAIL** — OCC
  consistency degraded −1139.5%; LIN +1170.2%.

**Descriptive finding (baseline):** A's fields are already highly
ray-coherent (median 0.43–0.44 px, tight p90 ≤ 0.66 px, zero degenerate
keypoints, both splits). The baseline's OCC direction error is therefore
**coherent mispointing**, not incoherent per-pixel noise. The objective
"make the 9 fields mutually coherent" had essentially no defect to repair in
the baseline — the fields already agree with each other; they just disagree
with the ground truth. This is a direct, measured refutation of the
premise that cross-keypoint incoherence is a component of the OCC direction
degradation in this regime.

## 13. Keypoint results (M7)

Voted keypoint error, original `ransac_voting_layer_v3`, EXP015 invocation
convention verbatim (seed reset, ROUND_HYP=128, INLIER_THRESH=0.99,
MAX_NUM=100), B fed the RAW refined field (pre-registered, §7):

| Split | A pooled median px | B pooled median px |
| --- | ---: | ---: |
| LIN | 1.2987 | 83.4084 |
| OCC | 2.3900 | 74.5926 |

- C4 (OCC keypoint median ≤ 1.05x A): **FAIL** — 31.2x worse.
- OCC keypoint median is clearly and massively worsened; C4's "no obvious
  worsening" is violated by an order of magnitude.

ADD(-S) / pose was deliberately not used as a judging metric (task §10,
EXP014 lesson); with keypoint errors of 60–80 px it would be uninformative
anyway.

## 14. Decision: GO / MIXED / STOP

Pre-registered criteria (all four required for GO):

| Criterion | Required | Observed | Verdict |
| --- | --- | --- | --- |
| C1 OCC dir mean | ≥ +10% | −1210.4% | FAIL |
| C2 LIN dir mean | ≤ +5% degradation | +2391.2% | FAIL |
| C3 M6 consistency | OCC ≥ +5% better, LIN ≤ +5% worse | −1139.5% / +1170.2% | FAIL |
| C4 OCC kp median | ≤ 1.05x A | 31.2x A | FAIL |

**DECISION: STOP — clean fail, 0/4 criteria.** This is not MIXED; no
criterion was satisfied, not even partially.

## 15. What was actually demonstrated

1. Under the pre-registered protocol (EXP012 budget, one structural loss,
   no tuning), the first minimal OCVF implementation produces catastrophic
   degradation of direction error, structural consistency, and keypoint
   localization simultaneously — on both LIN and OCC.
2. The failure mode is identifiable: the shared-configuration coupling term
   (L_struct, ~300x larger in raw magnitude than L_dir at init) dominated
   optimization, while the configuration head ĉ did not learn an accurate
   configuration (L_conf flat), so all 9 fields were jointly dragged toward
   an inaccurate shared target.
3. The GT-free M6 metric shows the **baseline** direction fields are already
   highly ray-coherent (0.43–0.44 px median). In this regime (single
   object, cat, 20+20 images), OCC direction degradation is coherent
   mispointing rather than cross-keypoint incoherence.

## 16. What was not demonstrated

1. That cross-keypoint structural coupling improves OCC direction error —
   the opposite was observed under this protocol.
2. That the coupling mechanism fails *in principle* for all architectures,
   scales, or loss weightings: this was one 22,612-parameter implementation
   at one fixed loss weighting over 5 epochs. Per the task's own rules, no
   second configuration was permitted, so the negative result is specific to
   this implementation and budget.
3. Any pose-level effect (pose was intentionally not a judging metric).

## 17. Is this scientifically distinct from EXP012?

**Yes.** EXP012's D1 is a keypoint-independent per-pixel spatial residual
(3x3→1x1 on F alone; no object latent, no shared configuration, no
cross-key interaction of any kind) and it *improved* OCC by +21.1%. OCVF
adds a single shared object latent z, shared key-level reader weights, and a
shared configuration constraint — structurally a different mechanism — and
it *catastrophically degraded* every metric. The implementations are
mechanistically distinct and behave in opposite directions; this experiment
is not a repackaging of EXP012.

## 18. Is another experiment justified?

**No.** The pre-registered decision rule is terminal: STOP after this one-
shot test. Per the task spec (§18), after STOP no second seed, no loss
change, no architecture change, no attention/graph variants, no added
objects, no added data, no additional epochs, and no EXP022 are authorized.
The descriptive finding in §12 (baseline fields already coherent) weakens
the original motivation for the coupling hypothesis itself: there was no
incoherence for coupling to fix.

---

**This experiment does not provide sufficient evidence to justify further
investigation of object-consistent vector fields.**
