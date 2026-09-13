# EXP003 REPORT — Inference-only Consensus-based Vote Reliability

**Decision: STOP** (per the pre-registered decision rule, Case C)

---

## 1. Motivation

Previous experiments showed: (a) baseline PVNet vertex geometry is good;
(b) oracle reliability filtering has large headroom (LINEMOD +16.6, OCC +19.4
ADD at keep-30%); (c) learned reliability heads failed to produce useful
rankings (frozen trunk: learned ~= random; joint training: ranking signal
exists but destroys the vertex field). EXP003 therefore tests the cheapest
possible reliability signal that requires **no training at all**: geometric
consistency between each vote line and the pass-1 coarse keypoint consensus.

## 2. Method

```
image -> PVNet 199.pth -> seg + vertex field
      -> pass 1: original ransac_voting_layer_v2  -> coarse keypoints C_k
      -> per vote: d_ik = |(C_k - p_i) x v_ik|    (perpendicular point-to-line
                                                    distance, normalized v_ik)
      -> reliability:  A4/A6 (hard): r = 1[d < tau], tau = 4/6 px
                       B4   (soft): r = max(exp(-d^2/(2*4^2)), 1e-3)
      -> pass 2: same hypotheses / inlier threshold / stopping rule as baseline,
         weighted scoring score(h,k) = sum_i r_ik * inlier(i,h,k),
         weighted LS refinement (sqrt-w folded into normals; w==1 -> bit-identical
         baseline formula)
```

Safety fallbacks (all recorded): reliable votes < 32, NaN/Inf coarse keypoint,
no foreground, singular/NaN refinement -> fall back to the pass-1 result.

## 3. Experimental Protocol

- checkpoint: `199.pth` (baseline, frozen), no training anywhere
- LINEMOD val = 501 images; OCC val = first half of Occlusion test = 593 images
- same evaluator / PnP / RANSAC base configuration (round_hyp_num 128,
  inlier_thresh 0.99, confidence 0.99, max_num 100, refine 1 iter) as baseline
- pass-1 equivalence verified per image: `max|pass1 - baseline| = 0.00e+00`
  on all 501 + 593 images (bitwise identical, well under the 1e-5 gate);
  U1 control (weighted pipeline with w==1) also bit-identical to baseline
- seeds 0/1/2 for all methods on both splits
- baseline reference (same ckpt/split/protocol): LINEMOD ~80.24, OCC ~17.54

## 4. Main Results — ADD(-S) %, mean ± std over seeds 0/1/2

| Method | LINEMOD val (501) | d vs base | OCC val (593) | d vs base |
|---|---|---|---|---|
| baseline | 80.31 ± 0.11 | — | 18.10 ± 1.11 | — |
| A4 (tau=4) | 80.11 ± 0.40 | −0.20 | 18.27 ± 0.51 | +0.17 |
| A6 (tau=6) | 80.24 ± 0.00 | −0.07 | 17.99 ± 1.04 | −0.11 |
| B4 (sigma=4) | 80.24 ± 0.35 | −0.07 | 18.27 ± 0.76 | +0.17 |

All deltas are within ±0.20 point, far below the seed-to-seed std
(±0.1–1.1). 2D projection and 5cm5deg move by <0.2 point as well
(see `results/metrics_*_full.json`, per-image rows in `results/exp003_*_full.csv`).

Fallback rates: LINEMOD 0; OCC ≈ 1.14 fallback events / image
(630 `no_fg` = 70 images with no visible cat pixels; 45 `few_reliable_votes`),
i.e. ~12.6% of keypoint round-2s fall back to the baseline result — mostly on
heavily occluded images with tiny visible masks.

## 5. Mechanism Analysis

Correlation between consensus distance d_ik and true vote error e_ik
(100 images per split, seed 0; `results/mechanism_*.json`):

| split | Pearson(d,e) | Spearman(d,e) | d mean / median (px) | keep ratio A4 |
|---|---|---|---|---|
| LINEMOD | +0.317 | +0.246 | 0.49 / 0.37 | 99.99% |
| OCC | +0.405 | +0.151 | 1.14 / 0.46 | 99.93% |

Reading:

1. The correlation is **positive but weak** — consensus distance does carry
   some information about vote quality (better than random), but far from
   enough to rank the bad half of votes that oracle filtering exploits.
2. The distribution is **almost degenerate**: pass-1 coarse keypoints are the
   RANSAC consensus, so by construction ~99.9% of vote lines already pass
   within a few pixels of C_k (A4 filters only 0.07–0.5% of votes). There is
   essentially nothing to filter.
3. The few votes that are far from consensus are already outvoted by
   RANSAC's inlier-count scoring (tn = 100 sampled pixels), so removing them
   changes neither the selected hypothesis nor the refinement.

## 6. Failure Analysis

The hypothesis fails at a structural level: the coarse keypoint is itself the
consensus of the votes, so "distance to consensus" measures agreement with the
result, not vote quality. Bad votes produced at occlusion boundaries are often
geometrically consistent with the *wrong* consensus (they helped elect it in
pass 1), which is exactly the failure mode the diagnosis experiment identified
(occl_boundary pixels have the largest true error yet dominate inlier counts).
A self-referential signal cannot separate those votes; this is consistent with
EXP002 (learned reliability ~= random) and with the top-k oracle gap that only
ground-truth ranking closes.

Note on artifacts: metrics JSON / per-image CSVs of the final runs hold seeds
1–2 (each run overwrites same-tag files); seed-0 metrics are preserved in this
report (LINEMOD: base 80.44, A4 79.64, A6 80.24, B4 80.04; OCC: base 18.04,
A4 18.38, A6 17.88, B4 18.21). Mechanism npz/png kept on disk only.

## 7. Decision

**STOP.**

Per the pre-registered rule: |delta| <= 0.5 ADD point on both splits (Case C),
despite pass-1 being bit-exact and all fallback machinery working. The
coarse-consensus signal is real (positive correlation) but too weak and too
low-variance to improve voting. Per the experiment charter, no threshold
sweeping is performed and this direction is closed. Remaining headroom on
vote reliability (oracle top-k) requires information that the vote geometry
alone does not contain — e.g. appearance/occlusion-aware signals — which is a
different (training) direction and out of scope for EXP003.
