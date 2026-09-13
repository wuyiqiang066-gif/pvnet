# EXP003 — Inference-only Consensus-based Vote Reliability

Inference-only experiment. No training, no network changes, no PnP/evaluator changes.

## Hypothesis

A vote whose line is inconsistent with the coarse (pass-1) keypoint consensus is
more likely unreliable. Pass-1 = original PVNet RANSAC voting (bit-identical to
baseline, verified per image, tolerance 1e-5). Pass-2 = re-score the SAME
hypotheses with per-pixel reliability weights and refine with weighted LS.

## Methods

- `baseline`: original `ransac_voting_layer_v2` (control, per-image)
- `A4` / `A6`: hard filtering, r = 1[d < tau], tau = 4 / 6 px
- `B4`: soft weighting, r = max(exp(-d^2 / (2*4^2)), 1e-3)
- `U1` (smoke only): weighted pipeline with w == 1, must equal baseline (<1e-5)

d = perpendicular point-to-line distance from the coarse keypoint to the
(normalized) vote line. Fallbacks (recorded): reliable votes < 32 -> coarse
result; NaN coarse -> coarse; singular / NaN refinement -> coarse.

## Protocol

- checkpoint `199.pth` (baseline, frozen)
- LINEMOD val = 501 images; OCC val = first half of Occlusion test = 593 images
- same evaluator / PnP as all prior experiments on this branch
- baseline reference (same ckpt/split/protocol): LINEMOD ADD(-S) ~80.24,
  OCC ADD(-S) ~17.54 (the historical 46.25 / 1219-image protocol is INVALID)
- seeds 0/1/2 for final numbers (mean +- std); seed 0 for development

## Files

- `config.json` — all knobs (RANSAC params identical to baseline protocol)
- `consensus_weighting.py` — problem replication, reliability, weighted RANSAC
- `run_exp003.py` — driver (pass-1 verification + methods + CSV/JSON outputs)
- `evaluate_exp003.py` — metric comparison tables
- `analyze_exp003.py` — mechanism check (d vs e correlations, distributions)

## Decision rule (fixed in advance)

improvement >= 2 ADD pts: CONTINUE; 0.5-2: weak, consider local consensus;
+-0.5: no effect, stop threshold tuning; clear drop: STOP direction.
