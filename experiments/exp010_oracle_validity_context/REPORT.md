# EXP010 — Oracle Validity-Aware Context Aggregation

Branch: `exp010_oracle_validity_context` (based on `exp009_frozen_local_context_refinement`)
Cat-only PVNet baseline (`199.pth`), offline diagnosis, no training, no baseline modification.

## 1. Research Question

EXP008: blind local aggregation of **GT** neighbor directions at r=3 reaches
0.569° mean on OCC (vs 5.121° baseline prediction). EXP009: a blind **learned**
local aggregation on frozen features degrades OCC by +25.7% (6.437°).

**If the aggregation knows which neighbors are valid/reliable (oracle validity
from GT), can it avoid error propagation and retain the EXP008 oracle benefit?**

## 2. Protocol

- Data: 20 LIN + 20 OCC cat images, IDs asserted against EXP005
  image_ids.json; seed=0; 199.pth; one forward pass per image; **no training**;
  runtime 2.8 s GPU.
- Pool: EXP006/008 definition — visible fg pixels, |K_k−p| ≥ 1px, 9 kps pooled.
  Cross-checks pass: baseline 2.969/5.121 and blind r3 0.362/0.569 reproduce
  EXP006/008 exactly (Δ<0.0004°); blind fallback count 0.
- **Validity (oracle, GT-based, pre-fixed threshold)**: for neighbor q and
  keypoint k, `e_q = angle(v_q_pred, d_q)`; `valid(q,k) = 1 if e_q ≤ 5°`.
  No learned component.
- **Aggregation at target (p,k)**, radius=3 Euclidean disk, q≠p, eligible =
  visible fg & |K_k−q| ≥ 1px (same neighbor set as EXP008):
  - **Blind**: `d̂ = normalize(mean of GT directions over ALL eligible neighbors)`
    — identical definition to EXP008 local r=3.
  - **Oracle-valid**: same but restricted to `valid(q,k)=1` neighbors.
  - If no valid neighbor: entry is **UNAVAILABLE** — excluded from
    oracle-valid statistics; never replaced by the baseline prediction.
- Aggregated content is GT directions in both arms (oracle-style, as EXP008);
  validity only changes the neighbor **selection**, isolating the effect of
  validity awareness while holding content fixed.
- Proximity bands (OCC only): dist_occ (EXP005/006 definition), [0,5)/[5,20)/[20,∞) px;
  only images with non-empty occluded region (9/20) contribute.
- Pre-registered decision: **GO iff** OCC oracle-valid ≤ 0.9 × OCC blind
  (clear improvement) **AND** OCC oracle-valid ≤ 0.85° (near EXP008 local r=3
  oracle 0.569°); else STOP.

## 3. Main Results

Direction error (deg), `results/context_aggregation_results.csv`:

| Method | LIN mean | OCC mean |
|---|---:|---:|
| Baseline prediction | 2.969 | 5.121 |
| Blind local r=3 | 0.362 | **0.569** |
| Oracle-valid local r=3 | 1.099 | **1.798** |

Oracle-valid is **3.2× worse than blind on OCC** (and 3.0× on LIN); on OCC its
median is 0.098° and p90 4.019° — the damage concentrates in a heavy tail.

## 4. Validity Statistics

`results/validity_statistics.csv` (per (pixel, keypoint) pool entries):

| split | mean valid ratio | median valid ratio | unavailable % |
|---|---:|---:|---:|
| LINEMOD | 0.862 | 1.000 | 6.92 |
| OCC | 0.689 | 1.000 | 20.09 |

Median ratio 1.0 on both splits: most pixels have **all** neighbors valid;
the low-validity mass is concentrated in a spatially clustered tail.

## 5. Occlusion Proximity (OCC)

`results/occ_proximity.csv`:

| dist to occluded region | n_entries | blind mean | oracle-valid mean | oracle-valid unavailable % |
|---|---:|---:|---:|---:|
| 0–5 px | 12,429 | 1.211 | 2.306 | 26.53 |
| 5–20 px | 40,324 | 0.477 | 1.576 | 16.48 |
| 20+ px | 30,306 | 0.755 | 1.639 | 11.50 |

Oracle-valid is worse than blind in **every** band, and unavailability is
highest exactly near the occlusion boundary (26.5%).

## 6. Correlation (mechanism metric)

OCC: **Spearman(baseline error at (p,k), r=3 valid-neighbor ratio at (p,k)) =
−0.8535** (p ≈ 0, n = 250,033).

Hard pixels sit in low-validity contexts: prediction errors are spatially
clustered, so validity selection thins the neighbor set **exactly where
aggregation is most needed**.

## 7. GO / STOP Decision

**STOP** (pre-registered conditions both fail: OCC oracle-valid 1.798° >
0.9×0.569°=0.512° and > 0.85°). Per protocol, no threshold sweeps, no extra
radii, no validity-head design followed.

Interpretation (mechanism level):

1. Blind GT aggregation at r=3 is already at the geometric floor (0.569°): its
   residual error is dominated by geometric smoothing bias (field curvature +
   one-sided truncation near occlusion), **not** by neighbor quality.
2. Knowing which neighbors are valid therefore has nothing to fix — and
   actively hurts: selection removes spatially clustered neighbors
   (Spearman −0.85), leaving smaller, more one-sided sets with larger
   geometric bias (OCC 1.798°; worse in every proximity band; 26.5%
   unavailable near the boundary).
3. Consequently, EXP009's failure cannot be repaired by validity-aware local
   re-selection: even **oracle** selection degrades the near-optimal blind
   aggregate. The binding constraint at r=3 is the spatial support, not
   neighbor curation.
4. Combined with EXP008 (global oracle 0.001° vs local r=3 0.569°), the
   remaining headroom lives at the **object level** (recovering the keypoint /
   direction-field structure from context), not in better local neighbor
   picking. A context mechanism whose only degree of freedom is neighbor
   weighting has no theoretical room at r=3 on this baseline.

## 8. Limitations

1. Aggregated content is GT directions (oracle-style, per protocol); a
   practical method would aggregate predicted content. This experiment
   intentionally isolates the *selection* effect; "selection alone does not
   help" does not exclude content-correcting mechanisms.
2. Fixed 5° validity threshold and r=3 only (both pre-registered; not swept).
3. Oracle-valid means are computed on the available subset (20.1% of OCC pool
   entries excluded as unavailable) — conditions already favor oracle-valid;
   it still loses.
4. Per-(pixel, keypoint) validity and pooling; keypoint identity not analyzed
   per pre-registration.
5. Cat-only baseline, single checkpoint (199.pth), 20+20 images, seed=0.

> This experiment is a mechanism diagnosis on the cat-only PVNet baseline and does not establish cross-object generalization.
