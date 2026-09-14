# EXP004 REPORT — Cheap Vote-Quality Signals in Baseline PVNet Outputs

**Decision: STOP** (pre-registered rule: every signal's |Spearman| < 0.10)

## 1. Goal

Find a cheap, non-self-referential signal, computable from the already-trained
baseline PVNet output *before* voting, that correlates with the true vote
error. Context: EXP002 (learned head, frozen trunk) and EXP003 (consensus
distance) both failed to rank vote quality; this experiment sweeps the
remaining zero-cost candidates.

## 2. Protocol

- checkpoint `199.pth`, baseline inference, no training, no model/code changes
- deterministic sample: first 20 images of LINEMOD val (501) and first 20 of
  OCC val (593); image indices 0–19 per split, recorded in `rows.npz`
- 446,841 LINEMOD vote rows (fg pixel x keypoint), 256,761 OCC rows
  (deterministic row-major stride, ≤3000 fg px/image)
- target: e_ik = point-to-line distance(GT keypoint, normalized vote line)
- self-referential signals (distance to RANSAC keypoint etc.) excluded by design
- Signal D (local vote-LINE consistency) skipped per directive (cost)
- runtime: < 2 min GPU total

## 3. Signal Results (Pearson / Spearman with true vote error)

| Signal | LIN Pearson | LIN Spearman | OCC Pearson | OCC Spearman | LIN Top50 err | OCC Top50 err |
|---|---|---|---|---|---|---|
| seg_confidence (A) | −0.053 | −0.086 | −0.100 | −0.076 | 1.142 | **1.602** |
| boundary_distance (B) | −0.066 | −0.065 | −0.048 | −0.032 | 1.166 | 1.651 |
| local_vertex_consistency (C) | −0.091 | −0.084 | +0.059 | −0.004 | **1.117** | 2.444 |
| vertex_magnitude (E) | −0.004 | −0.030 | −0.062 | −0.023 | 1.248 | 2.471 |
| seg × boundary (F) | −0.044 | −0.048 | −0.040 | −0.038 | 1.181 | 1.726 |

(negative = higher signal ↔ lower error, the expected direction)

## 4. Ranking Results (keep 50%, mean true error in px)

| Signal | LIN: selected / random / oracle | OCC: selected / random / oracle |
|---|---|---|
| seg_confidence | 1.142 / 1.224 / 0.438 | **1.602** / 2.209 / 0.611 |
| boundary_distance | 1.166 / 1.224 / 0.438 | 1.651 / 2.209 / 0.611 |
| local_vertex_consistency | 1.117 / 1.224 / 0.438 | 2.444 / 2.209 / 0.611 |
| vertex_magnitude | 1.248 / 1.224 / 0.438 | 2.471 / 2.209 / 0.611 |
| seg × boundary | 1.181 / 1.224 / 0.438 | 1.726 / 2.209 / 0.611 |

Full band analysis (top/bottom 10/30/50% by true error vs signal level) in
`results/error_bands.csv`; raw arrays in `results/rows.npz`.

## 5. Decision: STOP

Per the pre-registered rule:

1. **No signal reaches the Spearman bar**: best |Spearman| is 0.086
   (seg_confidence, LINEMOD) and 0.076 (seg_confidence, OCC) — all < 0.10.
2. LINEMOD Top50 gains are marginal (best −0.107 px, −9% vs random; oracle is
   −0.786 px, −64%). OCC Top50 gains look larger in relative terms
   (seg_confidence −0.61 px, −27% vs random) but stem from correlations far
   below the bar, and the oracle (0.611 px) remains 2.6× better than the best
   candidate — the headroom is not being captured.
3. Notably, the two signals that "work" slightly on OCC (segmentation
   confidence, boundary distance) are exactly the ones that fail to help
   RANSAC when used as vote weights would be expected to — and the
   local-consistency signal even flips sign on OCC (higher consistency ↔
   higher error), consistent with the diagnosis finding that occlusion-boundary
   votes are geometrically coherent with the wrong consensus.

Not performed (budget/scope): the optional occlusion-boundary vs interior
breakdown using amodal masks — the STOP decision is already determined by the
Spearman criterion, and importing the diagnosis region annotation was not a
few-minute change.

### Verdict

**STOP — insufficient evidence.** No cheap pre-vote signal in the baseline
PVNet output carries enough vote-quality information to exploit. Together
with EXP002 (learned head ≈ random) and EXP003 (consensus distance ±0.2 ADD),
this closes the "cheap reliability" avenue: the oracle headroom (top-30% true
ranking: LIN 98.4 / OCC 38.3 ADD) requires information that is neither
geometric (EXP003), learnable-from-frozen-features (EXP002), nor
appearance-cheap (EXP004).
