# EXP001-S Smoke Test Report (5 epochs, PASSED)

Run: `experiments/exp001_reliability_voting_scratch_smoke/` (separate dir),
config `configs/exp001_reliability_voting_scratch_smoke.json`, completed 5/5 epochs.

| Check | Result |
|---|---|
| 1. loss decreases | PASS — seg 0.0053 / ver 0.0205 at e4 (from scratch), decreasing per epoch |
| 2. reliability loss normal | PASS — L_rel ~0.59, no divergence |
| 3. reliability no collapse | PASS — pseudo-label mean 0.149→0.324 monotonic rise, std 0.29→0.37 |
| 4. segmentation normal | PASS — train precision/recall 0.95/0.94 at e4 |
| 5. vertex normal | PASS — val ver loss 0.013 |
| 6. weighted voting normal | PASS — evaluator consumed weighted corner output |
| 7. evaluator normal | PASS — val ADD 0.008→0.154 over 5 epochs (scratch trajectory) |
| 8. checkpoints | PASS — model/cat_exp001s_smoke/{0..4}.pth |
| 9. no NaN | PASS |
| 10. no inf | PASS |

Reliability pseudo-label percentiles (epoch 0 → 4):

| epoch | mean | std | p50 | p90 | p99 |
|---|---|---|---|---|---|
| 0 | 0.149 | 0.286 | 0.000 | 0.654 | 1.000 |
| 2 | 0.272 | 0.354 | 0.050 | 0.910 | 1.000 |
| 4 | 0.324 | 0.368 | 0.129 | 0.940 | 1.000 |

Interpretation: early in scratch training the vertex field is poor, so pseudo
labels concentrate near 0 with a wide spread; the distribution broadens and
shifts up as the vertex field improves. No saturation to 0 or 1 (§18 criteria).

Pre-flight (in `results/equivalence_test.md` of this experiment dir):
- r=1 weighted voting vs baseline: max abs diff **0.000e+00** (< 1e-6 required)
- L_rel gradient routing: rel_head > 0, convraw (seg+vertex head) = 0, backbone > 0
  (multi-task auxiliary path, by design)

Verdict: **PASS → formal 200-epoch run launched**
(`experiments/exp001_reliability_voting_scratch/`, ~15.5 min/epoch, ~54 h).
