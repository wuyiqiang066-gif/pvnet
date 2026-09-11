# EXP002: Frozen-Trunk Reliability Diagnostic

**身份：这不是最终方法。** 唯一目标：

> Determine whether a reliability head can learn discriminative vote quality
> while keeping the original PVNet geometric representation fixed.

## 定位（实验树）

```
EXP000  Baseline PVNet (scratch)
  ├── EXP001-S  scratch + joint reliability + weighted voting   (运行中, branch exp001_reliability_voting_scratch)
  ├── EXP002    Frozen-Trunk Reliability Diagnostic (本实验)
  └── Oracle    ideal reliability (diagnosis 实验)
```

三个因素拆解：**representation（trunk 是否被扰动）× reliability（head 学到什么）× voting（加权聚合）**。
EXP002 固定第一个因素（trunk = 199.pth，参数级验证 diff=0，见
`results/trunk_freeze_verification.json`），只训 rel_head，回答 Q2。

## 协议

* trunk: `data/model/cat_linemod_train/199.pth` 全部冻结（参数 requires_grad=False，
  **BN running stats 锁定**——BN 层强制 eval 模式，前向不更新统计量）
* 只训 rel_head（8 tensors），Adam lr 1e-3（与 baseline 协议一致），每 20 epoch ×0.5
* 50 epochs，batch 32，augmentation 与 baseline 完全一致（`config_diff.txt`）
* 伪标签：同 EXP001-S（predicted vertex 的投票线距离，detach；trunk 冻结故
  vertex 场固定，标签非平稳性只来自增广）
* 数据：LINEMOD cat，与 baseline 相同 split

## A/B/C 评估（每个检查点 e1/5/10/20/30/50）

* **A** = 199.pth + uniform voting（参考值，`results/baseline_reference.json`，
  corrected protocol：LINEMOD val 501 ADD 80.24；OCC val 593 ADD 17.54）
* **B** = frozen trunk + rel_head + **uniform** voting
* **C** = frozen trunk + rel_head + **weighted** voting
* B 与 C 使用完全相同的 seg/vertex 输出（同一 ckpt）→ **C − B = learned
  reliability → voting gain**（最纯粹的度量）
* control: **random reliability**（同形状 randn logits）与 uniform 对照

## Reliability 质量（§8）

对每个评估 ckpt，在 LINEMOD val / OCC val 各取 100 张图，计算
`sigmoid(rel_logits)` 与 true vote-line error（`pseudo_reliability` 的 e_all，
基于冻结 trunk 自身的预测）在前景像素 × 9 kp 上的 **Pearson 与 Spearman 相关**。
预期符号：负（reliability ↑ ↔ error ↓）。OCC 为重点观察对象。

## 判断标准（§9）

* Case 1: C ≈ B 且 |ρ| ≈ 0 → head 没学出来，不得声称 learned reliability 有效
* Case 2: C > B 且 reliability ↑ / vote error ↓ → frozen 表征包含足够信息（强正结果）
* Case 3: C > B 但相关性弱 → 检查非线性排序 / calibration / 加权效应 / RANSAC 交互，
  不简单否定

## 结果

见 `results/epoch_results.csv`（A/B/C + random control）与
`results/reliability_correlation.csv`。
