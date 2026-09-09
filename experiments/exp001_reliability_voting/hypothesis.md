# Hypothesis — exp001

## H1 (main)

Per-pixel, per-keypoint vote reliability is *predictable* from image evidence
(learned by a small head on the shared feature map), and using it to weight
RANSAC vote aggregation improves keypoint localization under occlusion.

## Operationalization

- Pseudo label: `r_i^k = exp(-e_i^2 / sigma^2)`, clamped to 1 for `e <= tau1`
  and 0 for `e >= tau2`, where `e_i^k` is the perpendicular distance from the
  GT keypoint k to the vote line of pixel i under the *predicted* (detached)
  vertex field. Not GT vertex — GT lines pass exactly through keypoints
  (e = 0 everywhere), so only predicted lines carry information.
- Inference: weighted inlier scoring `score(h,k) = sum_i r_i^k * inlier(i,h,k)`
  inside the unchanged RANSAC framework.

## Success criteria

- LINEMOD cat ADD(-S) >= baseline (80.64) within noise, AND
- OCC cat ADD(-S) > baseline (16.85) by a margin larger than seed noise
  (target: +2 pts or more), OR
- fewer voting-catastrophe images (kp error > 20 px) than baseline on OCC.

## Falsification

If weighted voting (with a converged reliability head) does not change, or
degrades, both LINEMOD and OCC metrics relative to the equal-weight baseline,
the hypothesis is falsified for this reliability definition.

## Known risks

- Cold start: r* depends on evolving vertex predictions (mitigated by stage 1
  warm-up on a frozen, converged baseline).
- Reliability is instance-agnostic; occluder pixels inside the mask may remain
  overweighted (motivates instance-aware voting, exp002+).
