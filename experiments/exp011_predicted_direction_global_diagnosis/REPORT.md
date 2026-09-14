# EXP011: Predicted-Direction Global Keypoint Diagnosis

## 1. Research Question

EXP008 showed that object-level aggregation of GT visible-pixel rays recovers
keypoints essentially exactly (GT-global oracle direction error 0.001 deg);
EXP010 showed that local validity-aware selection cannot beat blind local
aggregation. EXP011 answers the single remaining question on this line:

> If we do NOT use GT directions, but use PVNet's own predicted
> visible-pixel directions, can object-level geometric aggregation recover
> the keypoint?

This decides whether the next stage should target object-level readout or the
direction prediction itself.

## 2. Protocol

- Data: 20 LINEMOD + 20 OCC cat images, exactly the EXP005/006/008/010 IDs
  (asserted in-run); seed 0; checkpoint 199.pth; no training; GPU 3.1 s;
  no full-dataset evaluation.
- Ray set (identical for both estimators): every GT visible foreground pixel
  p with |K_k - p| >= 1 px; ray = p + t * v_p where v_p is PVNet's PREDICTED
  unit direction. No GT directions and no GT keypoints are used for
  estimation.
- Estimator A, Global-LS: least-squares intersection of all predicted rays
  (A = sum (I - u u^T), b = sum (I - u u^T) q, fixed ridge 1e-9).
- Estimator B, Global-Robust: the existing PVNet RANSAC voting mechanism
  (`ransac_voting_layer_v3`) called verbatim on the same GT visible mask with
  the baseline wrapper parameters (round_hyp_num=128, inlier_thresh=0.99,
  max_num=100), torch.manual_seed(0) before each call; mechanism unchanged.
- Regenerated target-pixel direction: d_hat = normalize(K_hat - p); metric =
  angle(d_hat, GT direction) on the EXP006/008 pool, 9 keypoints pooled.
- In-run controls: (i) baseline predicted-direction error reproduces
  EXP006/008 exactly (LIN 2.969 / OCC 5.121); (ii) the same LS estimator on
  GT rays reproduces the EXP008 global-oracle machinery (keypoint error
  ~1e-8 px), confirming the aggregation path is exact.
- Pre-registered decision rule: GO-object-level iff
  min(OCC mean of Global-LS, OCC mean of Global-Robust) <= 0.8 x 5.121 =
  4.097 deg (>=20% relative improvement), using the better estimator (fixed
  in advance to avoid cherry-picking); otherwise STOP-object-level.

## 3. Main Results

| Method                       | LIN mean | OCC mean |
| ---------------------------- | -------: | -------: |
| Baseline predicted direction |    2.969 |    5.121 |
| Global-LS predicted-ray      |    2.729 |    4.960 |
| Global-Robust predicted-ray  |    2.743 |    4.944 |
| EXP008 GT-global oracle      |    0.001 |    0.001 |

Best (Global-Robust) OCC relative improvement over baseline:
(5.121 - 4.944) / 5.121 = 3.5%.

## 4. Keypoint Localization

| Estimator              | LIN mean px | LIN median px | OCC mean px | OCC median px |
| ---------------------- | ----------: | ------------: | ----------: | ------------: |
| Global-LS              |       1.610 |         1.240 |       2.905 |         2.291 |
| Global-Robust          |       1.625 |         1.270 |       2.906 |         2.337 |
| LS on GT rays (control) |         ~0 |            ~0 |          ~0 |            ~0 |

## 5. Occlusion Proximity (OCC)

| dist to occluded region | baseline mean deg | Global-LS mean deg | Global-Robust mean deg |
| ----------------------- | ----------------: | -----------------: | ---------------------: |
| 0-5 px  | 7.653 | 7.507 | 7.573 |
| 5-20 px | 4.921 | 4.762 | 4.709 |
| 20+ px  | 3.522 | 3.134 | 3.045 |

## 6. GO / STOP

**STOP-object-level.** Best OCC mean = 4.944 deg (Global-Robust) against the
pre-registered GO threshold 4.097 deg; relative improvement is 3.5%, far
below the required 20%. Both estimators agree within 0.02 deg, so the
outcome does not depend on the choice of robust aggregator.

## 7. Interpretation

- The aggregation machinery is not the bottleneck: the in-run control (LS on
  GT rays) recovers keypoints to ~1e-8 px on the identical ray sets. When
  only the ray content is changed from GT to predicted directions, keypoint
  localization degrades to ~2.9 px (OCC) and the regenerated direction error
  only improves from 5.121 to ~4.94 deg.
- The EXP008 headroom (5.12 -> 0.001 deg) is therefore almost entirely locked
  inside the predicted direction field itself. Object-level aggregation of
  the existing predicted directions extracts only marginal extra global
  geometry beyond what the raw per-pixel directions already carry.
- Consistent with EXP004 (no cheap vote-quality signal), EXP009 (frozen local
  refinement fails on OCC), and EXP010 (validity-aware local selection
  fails), the actionable direction is the direction representation itself
  (EXP006 GO-A), not another aggregation or readout mechanism.

## 8. Limitations

- 20+20 cat-only images; direction error pooled over 9 keypoints; no
  keypoint-specific analysis by design.
- Estimator B reuses the baseline RANSAC wrapper with fixed parameters; no
  parameter sweep was performed, so it is a faithful reuse of the existing
  mechanism, not a tuned robust optimum.
- The LS estimator uses a fixed tiny ridge (1e-9); near-degenerate tangent
  configurations are handled but not specially optimized.
- Only direction-regeneration error and keypoint localization are reported;
  downstream pose metrics are out of scope.

This experiment is a mechanism diagnosis on the cat-only PVNet baseline and
does not establish cross-object generalization.
