# TRAINING STABILITY NOTE — Exp001 (warm-start) is STOPPED

Status: **STOPPED at epoch 99** (process killed on request; `kill` + `kill -9`
verified, GPU released). All checkpoints, logs, records, results, configs and
reports in this directory are preserved and must not be deleted.

## What this experiment was

Exp001 was initialized from the converged PVNet checkpoint `199.pth`
(`data/model/cat_linemod_train/199.pth`):

* epoch 0–7: backbone + vertex head frozen (incl. BN running stats), only the
  new reliability head trained (`model/stage1_rel_warmup/`),
* epoch 8+: full joint fine-tuning with the ORIGINAL PVNet learning rate
  (1e-3) (`model/stage2_joint/`).

## Why it was stopped

Joint fine-tuning with the original PVNet learning rate caused substantial
degradation relative to the baseline at matched epochs (same evaluator, same
val splits, plain PnP; ADD(-S) %):

| epoch | baseline (from-scratch run) | Exp001 warm-start |
|-------|------------------------------|-------------------|
| 80    | 77.45 (LINEMOD val)          | 51.70             |
| 85    | 80.44                        | 57.68             |
| 90    | 78.04                        | 50.70             |
| 80    | 16.19 (OCC val first half)   | 10.96             |
| 85    | 16.86                        | 13.66             |
| 90    | 17.37                        | 13.15             |

The pseudo-label mean also drifted from 0.83 (stage 1) to 0.55–0.62 (stage 2),
consistent with the backbone being perturbed by the aggressive LR.

## Interpretation

This result must NOT be interpreted as "reliability voting is ineffective".
It demonstrates that

> warm-start (199.pth) + original PVNet learning rate + joint fine-tuning is
> an unstable training protocol.

Therefore this experiment is treated as a **training-stability / feasibility
study** rather than the primary fair comparison.

The main experiment is `experiments/exp001_reliability_voting_scratch/`
(EXP001-S): scratch training with matched (seeded, archived) initialization
and the matched optimization protocol, on branch
`exp001_reliability_voting_scratch`.

## Artifact index

* `logs/train_exp001.log` — full training log (last epoch: 99)
* `model/stage1_rel_warmup/`, `model/stage2_joint/` — checkpoints
* `results/reliability_label_stats.jsonl` — per-epoch pseudo-label statistics
* `results/baseline_traj_eval.json` — baseline checkpoints 80/85/90 on the same protocol
* `results/baseline_val_reference.json` — baseline 199.pth on the same protocol
* `result_*_e*.txt`, `result_summary.txt` — Exp001 val metrics (epochs 80–95)
* `report.md` — original experiment report (pre-stop)
