# EXP002 Final Report — Frozen-Trunk Reliability Diagnostic

**身份**：Frozen-Trunk Reliability Diagnostic（非最终方法）。
**目标**：在完全冻结的 PVNet 表征（199.pth trunk，参数级验证 diff=0）上，确定
reliability head 能否学到有判别力的 vote 质量排序。
**日期**：2026-09-12 ｜ **分支**：exp001_reliability_voting_scratch ｜ **协议**：
corrected protocol（见 exp001 DATA_INTEGRITY_NOTE.md），PnP/Evaluator/dataset 零修改，
模型代码零修改（仅训练入口的 config 门控 `freeze_trunk`，默认关闭）。

---

## 1. Setup

* trunk = 199.pth 全部冻结（152 张量 max diff = 0，含 BN running stats，强制 eval 模式）
* 只训 rel_head（5 参数 + 3 buffer），Adam lr 1e-3（同 baseline），50 epochs（0–49）
* 伪标签 = 预测 vertex 投票线到 GT kp 的距离 r*=exp(−e²/9)（与 EXP001-S 相同）
* 评估：LINEMOD val 501 / OCC val 593；ABC 三模式 × 3 RANSAC seed 取 mean±std；
  top-k oracle 用确定性 LS 直线交点（无 RANSAC，无采样噪声），PnP/Evaluator 原样

## 2. A/B/C/R voting 评估（ADD(-S) %, mean ± std over 3 seeds）

| epoch | B uniform (LIN) | C weighted (LIN) | C−B (LIN) | R random (LIN) | B (OCC) | C (OCC) | C−B (OCC) | R (OCC) |
|---|---|---|---|---|---|---|---|---|
| e1  | 80.97±0.52 | 81.24±0.28 | +0.27 | 80.84±0.99 | 18.21±0.50 | 18.16±0.62 | −0.05 | 17.31±1.05 |
| e5  | 80.97±0.52 | 81.17±0.25 | +0.20 | 80.84±0.99 | 18.21±0.50 | 18.16±0.52 | −0.05 | 17.31±1.05 |
| e10 | 80.97±0.52 | 81.44±0.16 | +0.47 | 80.84±0.99 | 18.21±0.50 | 18.16±0.52 | −0.05 | 17.31±1.05 |
| e20 | 80.97±0.52 | 81.30±0.25 | +0.33 | 80.84±0.99 | 18.21±0.50 | 18.04±0.50 | −0.17 | 17.31±1.05 |
| e30 | 80.97±0.52 | 81.20±0.35 | +0.23 | 80.84±0.99 | 18.21±0.50 | 17.99±0.52 | −0.22 | 17.31±1.05 |
| e49 | 80.97±0.52 | 81.37±0.25 | +0.40 | 80.84±0.99 | 18.21±0.50 | 17.99±0.52 | −0.22 | 17.31±1.05 |

* B 跨所有 epoch 逐位一致（冻结 + 固定 seed）✓ 协议确定性成立
* C 在 LINEMOD 稳定高于 B (+0.2~0.5pt) 且 seed 方差更小（0.16~0.35 vs B 0.52）；
  OCC 上 C−B ≈ 0 且后期转负
* R（随机权重）在两个 split 都明显最差（OCC −0.9pt）→ 加权通道承载了真实信号，
  但见 §4：不是来自有效排序

## 3. Reliability prediction vs true vote error（100 张图/split，前景像素×9 kp）

| epoch | LIN Pearson | LIN Spearman | OCC Pearson | OCC Spearman |
|---|---|---|---|---|
| e1  | −0.163 | −0.158 | −0.321 | −0.287 |
| e5  | −0.178 | −0.165 | −0.322 | −0.282 |
| e10 | −0.171 | −0.163 | −0.286 | −0.272 |
| e20 | −0.176 | −0.166 | −0.252 | −0.254 |
| e30 | −0.182 | −0.169 | −0.249 | −0.237 |
| e49 | −0.183 | −0.168 | −0.229 | −0.219 |

符号正确（r↑ ↔ e↓）但幅值弱；**OCC 相关性随训练单调退化**（−0.32 → −0.23），
LINEMOD 平稳——训练数据只有 LINEMOD，head 对 OCC 的排序能力随对 LINEMOD 的
特化而衰减（OOD 退化）。

## 4. Top-k reliability filtering oracle（确定性 LS，keep% → ADD(-S) %）

| keep | LIN true | LIN learned(e49) | LIN random | LIN bottom | OCC true | OCC learned(e49) | OCC random | OCC bottom |
|---|---|---|---|---|---|---|---|---|
| 100% | 82.0 | 82.0 | 82.0 | 82.0 | 17.7 | 17.7 | 17.7 | 17.7 |
| 70%  | 94.6 | 80.6 | 81.4 | 77.5/78.0 | 22.3 | 17.5 | 17.5 | 17.0 |
| 50%  | 97.2 | 80.8 | 81.6 | 68.7/74.1 | 27.8 | 18.4 | 17.7 | 15.9 |
| 30%  | 98.6 | 80.6 | 81.2 | 57.3/63.9 | 37.1 | 18.2 | 17.5 | 14.3 |

（bottom 两列 LIN：e1 / e49）

**关键读数**：
* true ranking 的 headroom 巨大：LIN +16.6pt、OCC +19.4pt（keep 30%）——
  vote-error 信号真实存在且 e 标签本身是有信息的（bottom-k 恶化 −18~−25pt 亦证明）
* **learned ranking ≈ random ranking**（所有 keep、两个 split、两个 epoch）
  → head 没有把信号转化为可用排序
* learned 的微小超出（OCC keep50/30 +0.7~1.1pt vs random）方向为正但幅度在
  确定性协议的判读边界上，不构成有效排序证据

## 5. Reliability 直方图（前景像素，e1 → e49）

| | mean | p10 | p50 | p90 | frac>0.9 |
|---|---|---|---|---|---|
| LINEMOD e1 | 0.841 | 0.75 | 0.85 | 0.93 | 19.4% |
| LINEMOD e49 | 0.859 | 0.75 | 0.87 | 0.95 | 33.2% |
| OCC e1 | 0.828 | 0.73 | 0.85 | 0.91 | 16.7% |
| OCC e49 | 0.857 | 0.75 | 0.87 | 0.95 | 34.8% |

图：`results/reliability_histogram.png`

**动态范围不足**：分布压缩在 [0.75, 0.95]（p90−p10 = 0.20），且随训练进一步
向 1 饱和（frac>0.9: 19% → 33%）。加权≈常数权重 → weighted 退化回 uniform，
这直接解释 C−B 微小。**LIN 与 OCC 分布几乎重合**——尽管 OCC 的 vote error
更大（diagnosis: mean 5.4px vs 1.1px），若 head 真在按内容预测 e，OCC 分布应
整体左移；它没有 → head 输出被几何先验（像素到 kp 距离）主导，而非 per-image
vote 质量。

---

## 6. 三个问题的回答

### Q1: reliability head 是否学到了有效 pixel importance？
**否（按排序/filtering 标准）。** 相关性弱且 OCC 随训练退化；top-k oracle 中
learned ≈ random，而 true ranking +16.6/+19.4pt 的空间几乎完全未被捕获。
信号存在（bottom-k 恶化证明标签有信息），失败发生在"学习/表达"环节。

### Q2: weighted voting 收益是否来自真实排序？
**否。** C−B 的 LIN +0.2~0.5pt 不能归因于有效排序（learned 排序≈random）。
最可能的来源是近均匀权重的轻微正则化效应（加权 LS 的平滑 + min_weight 截断），
而非 vote-quality 排序。R（随机权重）最差说明该通道对"噪声权重"敏感、
对"学到的近均匀权重"几乎不敏感——进一步支持权重缺乏动态范围。

### Dynamic range 是否足够？
**否。** [0.75, 0.95] 的压缩分布 + 训练中持续饱和（BCE 软目标均值 0.82 的
回归效应）。没有动态范围就没有聚合层面的杠杆。

### 按 §9 判定：**Case 1（head 未学出）**
且 Case 3 的"非线性排序/calibration"豁免理由被 top-k oracle 直接排除
（oracle 是单调重排，learned 仍≈random；直方图排除了 calibration-only 解释）。

## 7. 对后续实验的含义（仅记录，不在本阶段执行）

1. **标签重设计**：r*=exp(−e²/9) 的几何先验主导了可学成分。候选：
   按 kp 归一化 e（消去距离先验）、二值尾部标签（e>τ）、ranking/pairwise loss
   （只要求坏像素排在好像素后），或对 e 减去 spatial prior 后再变换。
2. **head 容量/上下文**：8 参数的 3×3 head 可能不足以从外观预测 vote 失败
   （遮挡边界需要更大感受野与更高层特征）。
3. **OOD**：OCC 相关性退化说明 LINEMOD-only 训练的 head 不迁移；
   frozen 配置若要服务 OCC 需要混合域或域增广（在冻结约束内可行）。
4. EXP001-S（联合训练）的 weighted−uniform 增益（e100: LIN +5.0）与本项目
   的对比变成一个独立发现：联合训练的 head 反而有更强排序（待 e200 后检验其
   top-k oracle 是否接近 true）。

## 8. 产物清单

* `results/epoch_results.csv` — ABC×3seed 全量（1/5/10/20/30/49）
* `results/reliability_correlation.csv` — Pearson/Spearman 演化
* `results/topk_oracle.csv` — learned/random/bottom/true × 100/70/50/30 × e1/e49
* `results/reliability_hist_e{1,49}_{linemod_val,occ_val}.npz`、
  `results/reliability_histogram.png`
* `results/trunk_freeze_verification.json`、`results/baseline_reference.json`
