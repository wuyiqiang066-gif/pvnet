# exp000 — baseline (frozen reference)

- Branch `pvnet_baseline_verified`, tag `baseline_v1`
  (commit `ffb1f93` "Fix PVNet environment compatibility: CUDA13 Torch2.10",
  HEAD `bfd3e15` "Add baseline experiment record").
- Entry: `tools/train_linemod.py` + `configs/linemod_train.json` (untouched).
- Checkpoint: `data/model/cat_linemod_train/199.pth` (user-trained, verified:
  keypoint error < 1 px on demo; `data/model/199_onedrive.pth` is the
  incompatible OneDrive re-download and is NOT used).

## Verified reference numbers (LINEMOD cat)

| split | metric | value |
|---|---|---|
| LINEMOD test (1002) | ADD(-S) | 80.64 |
| LINEMOD test (1002) | 2D Projection | 99.80 |
| OCC LINEMOD test | ADD(-S) | 16.85 |

Logs: `test_cat_unc.log`, `test_cat_plain.log`; per-frame errors:
`data/record/cat_linemod_train.log`.

exp001 must compare against these numbers (or rerun the baseline entry for
matched-protocol numbers).
