# EXP017 — Pose Hypothesis Landscape Diagnosis

Pure diagnosis experiment on branch `exp017_pose_hypothesis_landscape`
(parent: `exp016_pnp_stability_diagnosis` @ e4348c1).

## Question

EXP016 showed that D1's improved OCC keypoints push the ORIGINAL standard PnP
into ~180° flip basins far more often than D0 (OCC >90° flips 14/20 vs 4/20),
while keypoint error magnitude does not separate flip from stable images.
EXP017 asks whether this is:

- **CASE A** — intrinsic ambiguity of the cat keypoint/PnP geometry itself
  (GT keypoints already flip), or
- **CASE B** — stable GT regime where D1's specific keypoint residual
  geometry is associated with pushing the original PnP into pre-existing
  wrong hypothesis basins, or
- **CASE C** — perturbation-sensitive numerical instability of the original
  PnP (matched 1 px noise erases the D0/D1 gap), or
- **CASE D** — inconclusive.

## Fixed protocol (nothing modified)

- Data: EXP016 `results/raw_predictions/raw_occ_D{0,1}.npz` reused
  bit-identically (voted keypoints, GT projected keypoints, GT poses) for the
  same 20 OCC held-out images as EXP015/016 (asserted against
  `image_ids.json` / `per_image.csv`). No re-sampling.
- PnP: original `lib.utils.evaluation_utils.Evaluator.evaluate` → `pnp`
  (cv2.SOLVEPNP_ITERATIVE, zero dist coeffs, `linemod` intrinsics,
  `VotingType.Farthest` 3D points, cat ply model). No flags/init/iteration/
  RANSAC/refinement/multi-hypothesis/flip-correction/GT-based selection.
- Conditions (only manipulated variable = 2D keypoint vector):
  - **C0** GT projected keypoints (control)
  - **C1** EXP016 D0 voted keypoints (must reproduce 4/20 flips)
  - **C2** EXP016 D1 voted keypoints (must reproduce 14/20 flips)
  - **C3** C1 + ε, ε ~ N(0, 1.0 px²) iid per coordinate, seed=0, ONE run
  - **C4** C2 + **same** ε realization as C3 (noise identity asserted
    elementwise < 1e-12)
- Flip definition (EXP016, not re-invented): rotation error
  `acos((trace(R_pred R_gtᵀ)−1)/2)`; flip = rot > 90/120/150/170°.
- No sigma sweep, no extra seeds, no extra datasets, no solver design.

## Usage

```bash
python experiments/exp017_pose_hypothesis_landscape/run_exp017.py
python experiments/exp017_pose_hypothesis_landscape/analyze_exp017.py
```

No GPU / network forward / voting is needed (inputs are stored EXP016
predictions; original PnP is CPU).

## Files

- `run_exp017.py` — conditions C0–C4 → original PnP; Check A (GT PnP,
  INTRINSIC_AMBIGUITY_SUSPECTED flag), Check B (exact C1/C2 reproduction vs
  EXP016, else STOP), Check C (noise identity); writes `results/`
- `analyze_exp017.py` — summary.csv, residual_geometry.csv (residual mean /
  spread / covariance / anisotropy / coherent shift / de-meaned scatter),
  D1 flip-vs-stable grouping, pre-registered decision → `decision.json`
- `REPORT.md` — fixed 11-section report
- `results/` — `image_ids.json`, `per_image.csv`, `summary.csv`,
  `residual_geometry.csv`, `noise_statistics.json`, `config.json`
  (pre-registered rules written BEFORE any PnP call),
  `reproduction_check.json`, `decision.json`

## Limitations

Cat-only, 20-image mechanism diagnosis. No significance claims. Decision
language is associational ("keypoint residual geometry is associated with
hypothesis selection"), not causal beyond the controlled comparison.
