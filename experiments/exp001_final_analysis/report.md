# EXP001 Final Analysis — Reliability-aware Pixel Voting (joint training, scratch)

Status: EXP001-S completed all 200 epochs (e0–e199, no crashes, no manual
intervention). This report performs the final hypothesis validation only.
Protocol: LINEMOD val 501 / OCC val 593 (corrected split), same PnP/evaluator,
3-seed ABC for e199; per-image CSVs in `results/`.

## DATA INTEGRITY

Correct OCC split: first half of Occlusion test = 593 images.
Correct baseline (199.pth, same protocol): LINEMOD ADD(-S) ~80.24,
OCC ~17.54. The historical 46.25 / 1219-image OCC reference is INVALID and
superseded (see `experiments/exp001_reliability_voting/DATA_INTEGRITY_NOTE.md`).

## 1. Same-epoch comparison, ADD(-S) %

| epoch | baseline (uniform) | Exp001-S uniform | Exp001-S weighted |
|---|---|---|---|
| **LINEMOD val** | | | |
| e60  | 74.1 | — | 56.5 |
| e70  | 76.1 | 50.5 | 52.1 |
| e100 | 76.9 | 46.9 | 51.9 |
| e149 | 79.9 ± 0.6 | 48.2 ± 1.2 | 50.6 ± 0.7 |
| e199 | 81.0 ± 0.5 | 43.3 ± 0.2 | 48.8 ± 0.5 |
| **OCC val** | | | |
| e60  | 17.5 | — | 12.5 |
| e70  | 15.2 | 14.2 | 15.4 |
| e100 | 17.5 | 15.0 | 16.7 |
| e149 | 17.6 ± 1.1 | 12.7 ± 0.5 | 13.9 ± 0.2 |
| e199 | 18.2 ± 0.5 | 16.9 ± 1.0 | 16.7 ± 0.6 |

Final gap (e199): LINEMOD −32.2 pt (weighted) / −37.7 pt (uniform);
OCC −1.5 / −1.3 pt.

## 2. Vertex field quality (true vote-line error, 200 val images)

| epoch/model | LIN mean | LIN median | LIN p90 | LIN bad>10px | OCC mean | OCC bad>10px |
|---|---|---|---|---|---|---|
| baseline e149 | 1.24 | 0.98 | 2.61 | 0.01% | 2.58 | 2.29% |
| baseline e199 | 1.21 | 0.97 | 2.56 | 0.01% | 2.54 | 2.11% |
| Exp001-S e149 | 2.18 | 1.58 | 4.75 | 1.06% | 2.86 | 2.63% |
| Exp001-S e199 | 2.34 | 1.74 | 5.03 | 1.18% | 2.92 | 2.70% |

Trajectory (LIN mean, earlier 50-img estimates): 1.59 (e60) → 1.60 (e70) →
1.52 (e100) → 2.18 (e149) → 2.34 (e199). The field degrades early, never
recovers, and the ADD gap even widens late (48.2→43.3 uniform e149→e199)
despite LR decay.

## 3. Weighted − uniform voting trajectory (same checkpoint)

| epoch | LINEMOD | OCC |
|---|---|---|
| e70 | +1.6 | +1.2 |
| e100 | +5.0 | +1.7 |
| e149 | +2.5 | +1.3 |
| e199 | **+5.6** | −0.2 |

LINEMOD: consistently positive, always beyond seed noise (±0.5).
OCC: within noise, inconclusive.

## 4. Reliability effectiveness at e199 (deterministic LS oracle, keep% → ADD)

| keep | LIN true | LIN learned | LIN random | OCC true | OCC learned | OCC random |
|---|---|---|---|---|---|---|
| 100% | 42.5 | 42.5 | 42.3 | 17.9 | 17.9 | 17.9 |
| 70% | 74.5 | 52.5 | 41.5 | 24.1 | 17.0 | 17.7 |
| 50% | 91.8 | 51.5 | 41.5 | 30.7 | 17.0 | 17.4 |
| 30% | 98.4 | 53.3 | 42.3 | 38.3 | 16.2 | 17.4 |

Reliability stats: LIN mean 0.64, std 0.19, Pearson(r,e) −0.03, Spearman −0.03;
OCC mean 0.68, std 0.17, Pearson −0.36, Spearman −0.19.

Key observation: on LINEMOD the learned ranking clearly separates from random
(+10 to +11 at keep 30–50%) although the global correlation is ~0 — the
signal lives in the top of the ranking. On OCC learned filtering does not
beat random (even slightly hurts). Note also: in the RANSAC path (ABC table)
weighted voting gains +5.6 on LINEMOD while the deterministic full-weight LS
path gains only +0.2 — the weights mainly help RANSAC select the correct
consensus mode, not the refinement itself.

## 5. Answers to the three questions

**Q1: Did Exp001-S learn a reliability ranking?**
Partial YES, split-dependent. LINEMOD: yes — learned ranking beats random by
+10 pt and captures a real slice of the oracle headroom (42.5→53.3 vs true
98.4 at keep-30%). OCC: no — learned ≈ random (slightly worse), despite the
strongest global correlation (Pearson −0.36); its headroom (17.9→38.3) is
uncaptured.

**Q2: Is reliability-weighted voting better than uniform voting?**
LINEMOD: YES, consistently (+1.6 to +5.6 across e70–e199, all beyond noise).
OCC: inconclusive (−0.2 to +1.7, within noise). The mechanism is hypothesis
selection: weighted scoring steers RANSAC to the correct consensus mode.

**Q3: Does the performance drop come from representation interference?**
YES — structurally and irreversibly. The vertex field degrades from 1.21 px
to 2.34 px mean error (bad votes 0.01%→1.18% on LINEMOD), the gap never
closes over 200 epochs, and late training even widens it. The reliability
weights recover only +5.6 pt of the −37.7 pt interference cost on LINEMOD.

## 6. Hypothesis validation statement (no improvement claims)

> Joint reliability training corrupts the underlying vertex representation
> (strong evidence, e199 final). The jointly-trained reliability head does
> learn a useful vote-quality ranking where the representation is damaged
> (LINEMOD), and reliability-weighted voting consistently outperforms uniform
> voting there, but the gain is an order of magnitude smaller than the
> interference cost. Under a frozen trunk the head learns no ranking
> (EXP002). Within the current design space, reliability-aware pixel voting
> does not improve PVNet: the ranking signal and the representation quality
> trade off against each other.

Context from sibling experiments (same protocol): EXP002 frozen-trunk
diagnostic — learned ≈ random (Case 1); EXP003 inference-only consensus
weighting — ±0.2 pt (Case C, STOP). Three independent attempts to exploit
vote reliability without fixing the representation conflict all fail to beat
baseline; the oracle headroom (top-30% true ranking: LIN 98.4, OCC 38.3)
remains open and requires signals the current design does not provide.
