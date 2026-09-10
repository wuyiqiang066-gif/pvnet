# EXP001-S Equivalence & Gradient Isolation Test

## 1) r=1 numerical equivalence (RNG-aligned, same hypotheses)
- max_abs_difference: **0.000e+00** (require < 1e-6)
- PASS

## 2) L_rel gradient routing (backward of L_rel only)
- rel_head |grad| sum: **1.2548e+01** (must be > 0)
- convraw (seg+vertex head) |grad| from L_rel: **0.0000e+00** (must be 0)
- resnet18_8s.conv1 |grad| from L_rel: 2.8498e-01 (multi-task aux path, expected > 0)
- other backbone/upsample params receiving L_rel grad: 71 tensors
- PASS

## 3) forward shapes
- seg (2, 2, 256, 256) | vertex (2, 18, 256, 256) | rel_logits (2, 9, 256, 256)
- NaN/inf: none

## Verdict
- ALL PASS