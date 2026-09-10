# EXP001-S: Reliability-aware Pixel Voting — From Scratch

Identifier: **EXP001-S** (`EXP001` = reliability-aware voting, `S` = scratch training).

This experiment is trained from scratch and does not load `199.pth`.

## Research Question

Does learned pixel-wise reliability improve PVNet's voting-based 6D pose
estimation when trained under the same protocol as the baseline?

One experiment, one hypothesis, one controlled variable. The ONLY added
variables vs baseline are: reliability head + reliability learning +
reliability-aware voting. Nothing else.

## Hypothesis

PVNet assumes that pixel-wise votes contribute equally to keypoint
localization, but vote reliability varies significantly across pixels,
especially near object boundaries and occlusion boundaries. Learning vote
reliability and using reliability-aware aggregation should improve robustness
under occlusion.

## Baseline

* branch `pvnet_baseline_verified`, commit `1911b9a`, tag `baseline_v1`
* original scratch training run checkpoints: `data/model/cat_linemod_train/{0..199}.pth`
* baseline trajectory ADD(-S) references (same evaluator/splits, plain PnP):
  `results/baseline_traj_eval.json`, `results/baseline_val_reference.json`

## Method

* Model: `Resnet18_8sReliability` — Resnet18_8s + `rel_head` (3x3 + 1x1 conv),
  output `[B, 9, H, W]` reliability logits (one channel per keypoint).
  Seg/vertex branches and backbone untouched.
* Reliability target (pseudo label): `r*(i,k) = exp(-e(i,k)^2 / sigma^2)`, where
  `e(i,k)` is the perpendicular distance from the GT keypoint to pixel i's
  PREDICTED vote line (detached; no gradient flows to the vertex head from L_rel).
  Clamp: `e <= 0.5 px -> 1`, `e >= 30 px -> 0`. `sigma = 3.0 px`.
* Loss: `L = L_seg + L_vertex + 1.0 * L_rel`, `L_rel = BCE(logits, r*)` over
  foreground pixels x 9 keypoints only.
* Voting: same RANSAC framework (same CUDA kernels), weighted hypothesis
  scoring `score(h,k) = sum_i r_i^k * inlier(i,h,k)` + weighted LS refinement
  (`min_weight = 0.001`).

## Initialization

* Exp001-S: **from scratch** — `torch.manual_seed(0)` BEFORE network
  construction; the resulting state is snapshotted to
  `initialization/shared_initialization.pth` and reloaded (bit-identical) by
  paired experiments (Ablation A).
* Baseline: the baseline training entry (`tools/train_linemod.py`) does NOT set
  a seed in its training path (`torch.manual_seed(0)` exists only in the
  `--test_model` branch; verified in this repo and in the old broken repo).
  Therefore baseline used the PyTorch default random initialization with an
  unknown draw. Bit-identical paired initialization with the existing baseline
  checkpoints is impossible; we fix this by (a) fixing the seed for Exp001-S,
  (b) archiving the exact initial state, (c) relying on same-epoch trajectory
  comparison with the baseline's own checkpoints. Initialization is
  PyTorch-default (Kaiming-uniform convs) for both, differing only in the RNG draw.
* Reliability head: PyTorch default conv init under the same seed
  (deterministic, recorded in the snapshot).

## Random Seed

`init_seed = 0` (network init + RandomSampler + dataloader workers).

## Optimizer / LR / Batch / Epochs

Identical to `configs/linemod_train.json` (see `config_diff.txt`):
Adam, lr 1e-3, decay x0.5 every 20 epochs, batch 32, 200 epochs,
`vote_round_hyp_num = 128` (= baseline hardcoded value), `inlier_thresh = 0.99`,
`max_num = 100` (= baseline hardcoded value).

## Dataset Split

Identical to baseline: LINEMOD cat, render_set + real train + fuse set for
training; `val_real_set` (501) and OCC test first half (1219) used only for
loss monitoring during training (no metric eval during training, matching
baseline `eval_epoch:false`); final metrics via the same test protocol.

## Loss

`L = L_seg + L_vertex + lambda * L_rel` with `lambda = 1.0`.
L_seg = CrossEntropy (baseline), L_vertex = smooth L1 (baseline).

## Reliability Target

See Method. Implemented in `tools/train_linemod_exp001.py::pseudo_reliability`
using the DETACHED predicted vertex field.

## Voting Method

`lib/ransac_voting_gpu_layer/ransac_voting_weighted.py`:
`ransac_voting_layer_v3_weighted` — identical hypothesis generation and
inlier computation as baseline `ransac_voting_layer_v3`; only hypothesis
scoring and the LS refinement are weighted by reliability.
Numerical equivalence at `r = 1` verified (see `results/equivalence_test.md`).

## Evaluation Protocol

Baseline `Evaluator` (PnP, ADD / ADD-S / 2D projection / 5cm5deg), plain PnP
(`ransac_voting_layer_v3` corner output -> `evaluator.evaluate`), batch size 1,
sequential sampler — identical to the baseline test path. No evaluator changes.

## Ablations (planned)

* **A** (reliability head + uniform voting): `configs/exp001_ablation_a.json`,
  shares `shared_initialization.pth`. Not run yet.
* **B** (oracle reliability filtering): existing diagnosis experiment (upper bound).
* **C** (learned reliability + weighted voting): this experiment.

## Training Protocol Note

No warm-up, no `199.pth`, no frozen backbone, no two-stage schedule. Joint
training from epoch 0.

## Results

See `results/` (`main_results.csv`, `epoch_results.csv`,
`reliability_statistics.csv`, `ablation_results.csv`) after training completes.
