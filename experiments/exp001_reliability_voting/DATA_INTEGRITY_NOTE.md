# DATA INTEGRITY NOTE — Invalid OCC baseline reference (2026-09-11)

Status: **CORRECTED PROTOCOL IN EFFECT.** All future evaluations use the
corrected protocol below. This note permanently documents the incident.

## 1. The invalid reference

`experiments/exp001_reliability_voting/results/baseline_val_reference.json`
originally reported, for baseline `199.pth` on `OCC val (first half)`:

```
ADD(-S) = 46.25%   (2D projection = 79.34%)    <-- INVALID / SUPERSEDED
```

## 2. Root cause

`/tmp/eval_baseline_val_reference.py` created **one shared `Evaluator` object
outside `run()`** and reused it for both the LINEMOD val run and the OCC val
run. PVNet's `Evaluator` accumulates per-frame errors; `average_precision()`
does not reset them. The "OCC" metrics therefore pooled 501 LINEMOD val frames
(ADD ≈ 80%) with the OCC frames, inflating the result.

All trajectory/diagnostic scripts written afterwards
(`eval_baseline_traj1.py`, `eval_exp001s_traj.py`, `eval_stage1_frozen.py`,
`eval_trunk_plus_rel7.py`) create a **fresh `Evaluator` inside each run** and
are unaffected — cross-checked against the independent full-test number from
EXP000 (OCC full test plain PnP ADD(-S) = 17.52%, 2438 images), which makes any
"first half = 46%" impossible (the second half would be negative).

## 3. Correct validation protocol (mandatory from now on)

* Splits: LINEMOD `val_real_set` (501 images); OCC `test_real_set[:len//2]`
  (593 images).
* One fresh `Evaluator` per (run x split); never shared across runs.
* batch size 1, sequential sampler, plain PnP, same `Evaluator` code as baseline.
* Metrics recorded: ADD(-S), 2D projection, 5cm5deg, image count, split name.
  (For category `cat` ADD and ADD-S coincide; both labels refer to the same number.)

## 4. Corrected baseline reference (199.pth, equal-weight voting)

| split | n | ADD(-S) | 2D projection | 5cm5deg |
|---|---|---|---|---|
| LINEMOD val | 501 | 80.24% | 99.80% | 98.60% |
| OCC val (first half) | 593 | 17.54% | 62.06% | 11.64% |

Stored in `results/baseline_val_reference.json` (overwritten with corrected
values by `eval_baseline_ref_fixed.py`; the invalid numbers exist only in this
note and in conversation history).

## 5. Historical conclusions retracted

Retracted (based on 46.25):

* "Exp001-S OCC val 已追平/超过 baseline" (claimed at e70: 15.35 vs 46.25).
  Corrected comparison at e100: Exp001-S weighted 16.69 vs baseline 17.54 —
  still 0.85pt below.
* Any OCC-vs-baseline statement derived from the old reference file.

Still valid (independent of 46.25):

* All `baseline_traj_eval.json` numbers (per-run Evaluators).
* All `epoch_results.csv` rows of EXP001-S (per-run Evaluators).
* Same-checkpoint weighted-vs-uniform gaps: e70 LIN +1.6pt; e100 LIN +5.0pt,
  OCC +1.7pt.
* EXP001-S LINEMOD interference gap (vertex field 1.52 px vs baseline 1.13 px
  at e100).
* `stage1_rel_warmup/7.pth` behavioral equivalence to the 199.pth trunk
  (LINEMOD 81.24 / OCC 18.21, both ≈ baseline + noise).

## 6. Going forward

* Every evaluation script must construct its `Evaluator` inside the per-split
  run function. A grep for shared evaluators is part of result review.
* Final EXP001-S / exp002 reports must restate metrics with the corrected
  protocol only.
