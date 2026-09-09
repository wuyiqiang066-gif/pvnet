# Experiment: Reliability-aware Pixel Voting (exp001)

## Experiment

Reliability-aware Pixel Voting — PVNet 6D pose estimation, LINEMOD cat.

## Motivation

PVNet treats all pixel votes equally (uniform inlier counting in RANSAC voting).

Diagnosis experiment (`experiments/diagnosis_vote_analysis/`) shows:

- Occlusion boundary pixels generate unreliable votes: `occl_boundary` has the
  largest mean vote error (6.48 px) yet contributes 20.8% of inliers.
- Under occlusion (OCCLUSION LINEMOD), 19.2% of images have keypoint error > 20 px.
- The built-in uncertainty signal fails under occlusion
  (rank-corr with true kp error: LINEMOD 0.40 vs OCC -0.07).

## Hypothesis

Learning per-pixel per-keypoint vote reliability can improve voting robustness.
Oracle experiment (using GT vote error to weight/filter pixels):
LINEMOD ADD 79.6 -> 99.4 (top 30% pixels), OCC ADD 17.9 -> 36.4 — there is
learnable headroom.

## Baseline

PVNet `baseline_v1` (tag `baseline_v1`, branch `pvnet_baseline_verified`,
checkpoint `data/model/cat_linemod_train/199.pth`).
Verified numbers (LINEMOD cat test 1002): ADD(-S) 80.64, 2D Projection 99.80;
OCC test: ADD(-S) 16.85.

## Modification

1. Reliability prediction head (`rel_head`, output `B x 9 x H x W`,
   one logit per pixel per keypoint) on the shared raw-resolution feature.
2. Weighted voting aggregation:
   `score(h,k) = sum_i r_i^k * inlier(i,h,k)` (same RANSAC framework, same CUDA
   kernels; only scoring + LS refinement become weighted).
3. No change to: dataset, PnP, evaluation, baseline branch code.

Training: two-stage.
Stage 1 freezes backbone + vertex head (incl. BN stats), trains only `rel_head`
with pseudo labels `r* = exp(-e^2/sigma^2)` clamped to [tau1, tau2]
(e = distance from GT keypoint to the pixel's *predicted* vote line).
Stage 2 jointly fine-tunes everything.

## Switch

`configs/exp001_reliability_voting.json`: `"use_reliability_vote": false`
recovers the exact baseline path from the same entry.
