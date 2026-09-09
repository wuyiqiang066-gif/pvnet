# Design — exp001 Reliability-aware Pixel Voting

## Invariants (hard constraints)

- Baseline branch code: zero modification (verified by `git diff HEAD` empty
  on all tracked files).
- Dataset, PnP, evaluation: untouched.
- No new CUDA kernels: weighted voting reuses `generate_hypothesis` /
  `voting_for_hypothesis` from the compiled extension; only scoring (einsum)
  and the final weighted least-squares refinement are new (pure PyTorch).

## Model

`Resnet18_8sReliability(Resnet18_8s)` — additive subclass
(`lib/networks/model_repository_exp001.py`):

```
fm = shared raw-resolution feature (s2dim=32)   # identical to baseline
seg_pred, vertex_pred = convraw(cat[fm, x])     # identical to baseline
rel_logits = rel_head(fm)                       # NEW: [B, 9, H, W]
```

Baseline checkpoints load with `strict=False` (only `rel_head` is new).

## Loss

```
L_total = L_seg + w_ver * L_vertex + lambda * L_rel
L_rel   = BCE_with_logits(rel_logits, r*) over fg pixels x 9 kps
r*_i^k  = clamp_exp(e_i^k; sigma, tau1, tau2),  e = |perp dist to predicted vote line|
```

The vertex field used for r* is **detached** — L_rel sends no gradient into the
baseline branch through the labels.

## Two-stage training

| | Stage 1 (warm-up) | Stage 2 (joint) |
|---|---|---|
| init | baseline `199.pth` (`strict=False`) | stage 1 result |
| trainable | `rel_head` only | all parameters |
| frozen | backbone + vertex head (params + BN running stats) | — |
| epochs | `stage1_epochs` (8) | rest (`epoch_num - 8`) |
| ckpt dir | `model/stage1_rel_warmup/` | `model/stage2_joint/` |

## Weighted voting

```
score(h,k) = sum_i w_i^k * inlier(i,h,k),  w = sigmoid(rel_logits),  w < min_weight -> 0
confidence loop / hypothesis sampling: unchanged (uniform sampling)
refinement: weighted LS (sqrt(w) folded into normals; identical formulas)
fallback: if all pixels of keypoint k are filtered out -> uniform weights (baseline)
parity:   uniform weights reproduce ransac_voting_layer_v3 exactly (smoke-tested, diff=0.0)
```

## Switch

`use_reliability_vote=false` -> baseline model class, baseline losses, baseline
equal-weight voting, from the same entry (`tools/train_linemod_exp001.py`).

## Outputs

```
experiments/exp001_reliability_voting/
  config.json                          resolved config snapshot
  record/<model_name>.log              text log + tensorboard
  model/stage1_rel_warmup/<ep>.pth     stage 1 checkpoints
  model/stage2_joint/<ep>.pth          stage 2 checkpoints
  results/reliability_label_stats.jsonl per-epoch label stats
  results/label_hist/epoch_XXX.npz     per-epoch foreground label histogram
  results/result_<split>_e<ep>.txt     ADD / 2D proj / 5cm5deg
  results/result_summary.txt           appended history
  results/figures/                     visualization (Part 7)
```
