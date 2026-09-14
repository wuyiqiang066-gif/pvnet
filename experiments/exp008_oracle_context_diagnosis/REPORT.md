# EXP008 — Oracle Context Diagnosis: Is the Correct Vertex Direction Recoverable from Visible-Object Context?

Branch: `exp008_oracle_context_diagnosis` (based on `exp006_vertex_representation_diagnosis`)
Inference-only, cat-only PVNet baseline (`199.pth`), no training, no baseline modification.

## 1. Research Question

For a foreground pixel p in an OCC image whose predicted vertex direction is poor:
**does the object-level spatial/context information provided by OTHER VISIBLE
foreground pixels contain enough information that p's correct direction
`normalize(K - p)` could in theory be recovered?**

This separates two hypotheses:

- **A. Information-limited** — occlusion removes the contextual information
  needed to infer the correct direction; context-based representation has
  little theoretical room.
- **B. Representation-limited** — the visible region already contains the
  needed object-level information; the current per-pixel vertex representation
  simply does not exploit it; context-aware vertex representation has clear
  research room.

## 2. Protocol

- Data: 20 LINEMOD cat + 20 OCC cat images, **identical to EXP005/EXP006**
  (asserted at runtime against `exp005_visibility_diagnosis/results/image_ids.json`).
- Checkpoint: existing `data/model/cat_linemod_train/199.pth`, single forward
  pass, **no training**. Total runtime **4.7 s GPU** (seed = 0).
- Pixel pool (identical to EXP006 "all" region): visible foreground pixels with
  `|K_k - p| >= 1 px`, all 9 keypoints pooled. Baseline reproduced EXP006
  exactly (mean/count match to <0.001 deg / exact count), see `run.log`.
- **Baseline**: angle between PVNet predicted unit direction and GT direction.
- **Local oracle (r = 3/7/15/30)**: mean of the *GT* unit directions of eligible
  visible pixels within Euclidean radius r (p excluded). Eligible = visible and
  `|K_k - q| >= 1 px`. If no eligible neighbor exists, fall back to the nearest
  eligible visible pixel's GT direction (deterministic; fallback rate reported).
  Deliberately simple per spec: the goal is the recoverability-vs-context-scale
  trend, not the best estimator.
- **Global oracle**: object-level readout. The keypoint location `K_hat` is
  estimated by least-squares intersection of the GT direction rays of all
  *other* eligible visible pixels (p's own ray excluded), then
  `d_hat = normalize(K_hat - p)`. The trivial oracle `normalize(K_exact - p)`
  is intentionally NOT used (error 0 by construction, non-diagnostic): under
  occlusion K itself must be inferred from the visible context, which is
  exactly what this oracle measures.
- **Proximity bands** (OCC only): `dist_occ` = distance to the occluded region
  (amodal ∧ ¬visible), EXP005/006 definition; bands [0,5) / [5,20) / [20,∞) px.
  Only images with a non-empty occluded region contribute (**9/20** OCC images).
- Recoverable gap = baseline error − oracle error.

Neighbor GT directions are the idealized stand-in for a perfect readout of
visible object geometry (see Limitations).

## 3. Baseline vs Oracle Context Results

Direction error (deg), `results/oracle_context_results.csv`:

| method | LIN mean | LIN median | LIN p90 | OCC mean | OCC median | OCC p90 |
|---|---|---|---|---|---|---|
| baseline | 2.969 | 1.699 | 6.046 | 5.121 | 2.822 | 10.901 |
| local r=3 | 0.362 | 0.001 | 0.925 | **0.569** | 0.001 | 1.393 |
| local r=7 | 1.562 | 0.349 | 3.605 | 2.330 | 0.738 | 5.277 |
| local r=15 | 5.781 | 2.979 | 12.456 | 7.926 | 4.671 | 17.184 |
| local r=30 | 16.196 | 9.648 | 34.893 | 18.266 | 12.215 | 39.412 |
| global | 0.001 | 0.001 | 0.001 | **0.001** | 0.001 | 0.001 |

Fallback rate 0.00000 for all local oracles (every pixel had an eligible
neighbor within r=3; the nearest-visible fallback path was never exercised).

Key readings:

- **Global oracle ≈ 0.001° on both LIN and OCC**: given exact GT directions of
  the visible pixels, the keypoint location is recoverable essentially exactly
  by ray intersection — including when K itself is occluded. No pixel set in
  the sample produced a degenerate/ill-posed case (no solve failures, no
  fallbacks).
- **Local r=3 oracle: 0.569° on OCC** — 9× better than the network's 5.121°
  prediction using only the 1–3 px neighborhood of the GT direction field.
  The GT direction field is locally smooth and highly redundant between
  adjacent pixels.
- **Local r=15/30 get worse than baseline** (OCC r=30: 18.27°). This is a
  property of the naive mean estimator, not of information: at mid/large
  radii the direction field has curvature and a singularity at K, so
  averaging directions across a wide disk mixes incompatible values. Small
  radii (r≤7) or an object-level readout are the viable context scales.

## 4. Recoverable Gap

`results/recoverable_gap.csv`, gap_mean / gap_median (deg):

| method | LIN gap (mean) | OCC gap (mean) | OCC gap (median) |
|---|---|---|---|
| local r=3 | +2.607 | **+4.552** | +2.821 |
| local r=7 | +1.407 | +2.791 | +2.085 |
| local r=15 | −2.812 | −2.805 | −1.848 |
| local r=30 | −13.227 | −13.145 | −9.393 |
| global | +2.968 | **+5.120** | +2.822 |

On OCC the global oracle recovers the **entire** baseline error (gap = 5.120°,
i.e. the full mean error), and even the trivial 3-px local readout of the GT
field recovers 89% of it (4.55° of 5.12°).

## 5. Occlusion Proximity Analysis (OCC)

`results/occlusion_proximity.csv` (9/20 images have a non-empty occluded region):

| dist to occluded region | n_entries | baseline mean | baseline median | global mean | gap mean | gap median |
|---|---|---|---|---|---|---|
| 0–5 px (near) | 12,429 | 7.653 | 3.368 | 0.001 | **7.652** | 3.368 |
| 5–20 px (middle) | 40,324 | 4.921 | 2.509 | 0.001 | 4.920 | 2.508 |
| 20+ px (far) | 30,306 | 3.522 | 1.885 | 0.001 | 3.521 | 1.885 |

The recoverable gap is **largest exactly where the baseline is worst**: pixels
adjacent to the occlusion boundary lose 7.65° of error under the global
oracle. The information barrier, if any, is not at the occlusion boundary.

## 6. GO / STOP Decision

**GO — Representation-limited.** All three pre-registered conditions hold:

1. OCC baseline direction error is clearly elevated (5.121° vs 2.969° LIN).
2. Oracle context clearly reduces the error: global 0.001° (gap 5.12°), and
   even the deliberately naive local r=3 readout reaches 0.569° (gap 4.55°).
3. The near-occlusion region has the largest recoverable gap (7.65°).

Conclusion: the correct direction information is **not removed by occlusion**.
It is present in the visible region — redundantly at the 1–3 px scale of the
direction field, and exactly at the object level (keypoint location) — while
PVNet's per-pixel vertex prediction (5.12°) does not exploit it. The failure
is representational, not informational.

## 7. What This Means for the Next Research Direction

- Context-aware / object-aware vertex representation has genuine theoretical
  headroom on this baseline: the oracle headroom equals the full baseline
  error, concentrated at occlusion boundaries.
- **Context scale matters structurally.** The direction field
  `normalize(K - p)` is smooth at small scales but has a singularity at K;
  naive wide-radius aggregation (r=15/30) is anti-productive. A context
  mechanism should either (a) use narrow local context, or (b) recover the
  object-level quantity (keypoint location / direction-field structure) from
  context and re-derive per-pixel directions — mirroring what the global
  oracle does.
- Relation to EXP007: EXP007's learned local-context residual showed no
  positive signal under an extremely small training budget. This experiment
  shows that negative result cannot be explained by absent information — the
  information is demonstrably there (even trivially local). The gap between
  oracle and learned aggregator is an optimization/capacity/budget problem,
  not an information problem.
- Practical bridge that remains to be built: a real method must obtain context
  from images or predicted outputs rather than GT. PVNet's own visible-region
  direction error (≈3–5°) and EXP004's finding that no cheap vote-quality
  signal exists together imply that *how* context is aggregated (robust,
  geometric, object-level) is the actual research object.

## 8. Limitations

1. The oracles use **GT directions of neighboring pixels** — a perfect readout
   of visible geometry that no practical method has. Results are an upper
   bound on context value, not an achievable number; a learned method must
   estimate context and will face error propagation (visible-region prediction
   error ≈3–5°).
2. The global oracle ≈0.001° is partly **by construction**: exact GT rays
   intersect exactly at K. Its diagnostic content is limited to ruling out
   gross information barriers (degenerate slivers, ill-posed geometry) — which
   it does for this sample; it does not mean 0° is achievable by a real method.
3. The local estimator is deliberately naive (mean of unit directions); the
   r=15/30 degradation reflects estimator bias (field curvature + singularity
   near K), not absent information.
4. Proximity analysis covers only the 9/20 OCC images with non-empty occluded
   regions; band definition follows EXP005/006 `dist_occ`.
5. Cat-only baseline, single checkpoint (199.pth), 20+20 images, seed=0.

> This experiment is a mechanism diagnosis on the cat-only PVNet baseline. It does not establish cross-object generalization.
