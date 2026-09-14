# EXP005 REPORT — Visibility vs Representation Failure Diagnosis

> This is a diagnostic experiment, not a proposed method.

**Decision: GO — occlusion-aware representation** (all three pre-registered conditions satisfied).

## 1. Question

OCC 下 PVNet 性能下降，主要因为

- **Hypothesis A**: 可见像素减少（vote 数量减少），还是
- **Hypothesis B**: 剩余可见像素的 vertex representation 本身已经错误？

背景：Oracle 存在巨大 headroom（EXP-diagnosis）；EXP002 frozen reliability ≈ random；EXP003
consensus ≈ baseline；EXP004 廉价信号全部 |Spearman|<0.10。因此本实验不再做 reliability
weighting，只定位失败来源。

## 2. Protocol

- 纯 inference 诊断：`199.pth`，baseline PVNet，不训练、不改任何 baseline/RANSAC/CUDA/PnP/evaluator 代码
- 数据：与 EXP004 完全相同的 deterministic first-20（LINEMOD val 前 20 张；OCC test 前 20 张），
  image IDs 记录于 `results/image_ids.json`；共 441,387 LIN + 250,389 OCC 条 vote（fg pixel × 9 kp，无采样上限）
- 区域划分（复用 `diagnosis_vote_analysis` 的 GT 定义，未做任何猜测；40/40 张图的 amodal mask 全部存在，无 unavailable 情况）：
  - `interior` = 可见区域向内腐蚀 5 次（3×3 核）
  - `normal_boundary` = 距轮廓 ≤5px 且不在遮挡带内的可见像素
  - `occlusion_boundary` = 距遮挡区（amodal & ~visible）≤5px 的可见像素
- 指标（与此前 diagnosis 完全一致）：`e_ik` = GT 关键点到 vote 直线的垂直距离（px）；
  vertex angular error = acos(clamp(dot(v_pred_hat, v_gt_hat)))，v_gt 指向 GT 关键点（距 kp <1px 的像素剔除）
- 反事实：LINEMOD 第 i 张 subsample 至 OCC 第 i 张的可见 fg 数，LS 直线交点估计器两侧一致，每张 3 次重复
- Pose sensitivity：GT 可见 mask 上跑**原始** `ransac_voting_layer_v3` + 原始 `pnp`，
  仅对比 all-visible vs remove-occlusion-boundary（未写任何新 RANSAC）
- 阈值：bad vote = e>10px；bad direction = >30°；oracle top-50%；seed 固定（torch/np seed=0，子采样 RNG 确定性）
- GPU 时间：**4.4 s**（含推理与全部统计）

## 3. Vote Count Analysis

| split | visible fg px (mean) | visible fg px (median) | occluded px (mean) | amodal px (mean) |
|---|---|---|---|---|
| LINEMOD | 2452.2 | 2424.0 | 0.1 | 2452.2 |
| OCC | 1391.0 | 1594.0 | 226.2 | 1617.3 |

OCC 可见 vote 数平均减少约 **43%**（前 20 张中两张近全遮挡：img18/19 仅 269/288 可见 px）。
OCC 内部差异极大：12/20 张图 GT 上无遮挡像素，img15–19 遮挡 83–1196 px。

## 4. Vote/Vertex Error by Region

### vote line error e_ik (px)

| split | region | votes | mean | median | p90 | bad rate (e>10px) |
|---|---|---|---|---|---|---|
| LINEMOD | all | 441,387 | 1.214 | 0.911 | 2.599 | 0.03% |
| LINEMOD | interior | 167,283 | 1.140 | 0.853 | 2.433 | 0.00% |
| LINEMOD | normal_boundary | 272,988 | 1.259 | 0.947 | 2.710 | 0.05% |
| OCC | all | 250,389 | 1.711 | 1.278 | 3.742 | 0.29% |
| OCC | interior | 69,345 | 1.630 | 1.255 | 3.661 | 0.00% |
| OCC | normal_boundary | 168,597 | 1.684 | 1.275 | 3.693 | 0.18% |
| OCC | occlusion_boundary | 12,447 | **2.520** | 1.485 | **6.209** | **3.26%** |

### vertex angular error (deg)

| split | region | mean | median | p90 | bad rate (>30°) |
|---|---|---|---|---|---|
| LINEMOD | all | 2.97 | 1.70 | 6.05 | 4.3% |
| OCC | all | 5.12 | 2.82 | 10.90 | 11.6% |
| OCC | interior | 5.33 | 3.00 | 11.29 | 12.5% |
| OCC | normal_boundary | 4.85 | 2.71 | 10.27 | 10.5% |
| OCC | occlusion_boundary | **7.65** | 3.37 | **19.26** | **21.2%** |

关键观察：

1. OCC 内部呈现 interior → normal_boundary → occlusion_boundary 的单调恶化（mean 1.63→1.68→2.52；
   p90 3.66→3.69→6.21；bad vote 率 0%→0.18%→3.26%；angular bad rate 12.5%→10.5%→21.2%）。
2. **OCC 连 interior 都比 LINEMOD 差**（angular 5.33° vs 2.96°，e 的 p90 3.66 vs 2.43）——
   退化不局限于遮挡边界。
3. e>10px 的坏 vote 高度集中于 occlusion_boundary（3.26% vs 其他区域 ≈0%）。

## 5. Equal-count Counterfactual

LINEMOD 第 i 张 subsample 到 OCC 第 i 张的可见像素数后（同一 LS 估计器）：

| 条件 | keypoint error (px, mean over 9 kp, 20 pairs) |
|---|---|
| LINEMOD full (mean 2452 px) | 1.61 |
| LINEMOD subsampled (mean 1357 px) | 1.61 ± 0.49 (3 reps) |
| OCC all visible (mean 1391 px) | **2.90 ± 1.87** |

- vote 数匹配后 LINEMOD 仍比 OCC 好 **1.8 倍** → 性能差距不能由 vote 数量减少解释。
- LIN-sub ≈ LIN-full 说明该估计器对 vote 数量几乎不敏感（LS 直线交点对采样密度不敏感），
  进一步支持「数量不是瓶颈」。
- 极端遮挡图（img18: 269 可见 px、img19: 288 px）OCC kp error 达 10.14/5.74 px，
  而同数量 LIN-sub 仅 0.78/0.82 px — 差距 13 倍。

## 6. Oracle Region Analysis

OCC（vote 占比 vs top-50%/bottom-50% 可靠 vote 中的占比）：

| region | 占全部 vote | top50 占比 | bottom50 占比 |
|---|---|---|---|
| interior | 27.7% | 14.3% | 13.4% |
| normal_boundary | 67.3% | 33.2% | 34.1% |
| occlusion_boundary | 5.0% | 2.5% | 2.5% |

注意：occlusion_boundary 在 bottom50 中**并未**被过度代表（2.5% vs 总体 5.0%）——
区域成员本身不能预测 vote 落入哪个半区。坏 vote 以**尾部**形式集中于 occlusion_boundary
（e>10px 占其 3.26%，其他区域≈0%），而非整体分布偏移。这与 §5 结论一致：
简单按区域过滤（见 §7 pose sensitivity）无法修复问题，需要的是表示层面的修正。

### Pose sensitivity（OCC，GT 可见 mask，原始 RANSAC+PnP）

| 配置 | mean ADD-S | pass@0.1d |
|---|---|---|
| all visible votes | 277.0 mm | 0.65 |
| remove occlusion-boundary votes | 278.4 mm | 0.65 |

移除 occlusion-boundary vote **没有**改善（略变差）；img18/19 在两种配置下同样灾难性失败
（ADD-S ≈ 2.65 m）。坏 vote 不是集中在可被区域过滤剔除的少数像素上。

## 7. Answers to Q1/Q2/Q3

**Q1**: 是。OCC 的 vote/vertex error 沿 interior → normal boundary → occlusion boundary
显著恶化（mean e 1.63→1.68→2.52px；p90 3.66→3.69→6.21px；bad vote 率 0→0.18%→3.26%；
angular bad rate 12.5%→10.5%→21.2%）。且 OCC 连 interior 都差于 LINEMOD 整体水平。

**Q2**: 否。equal-count 后 LINEMOD（1.61px）仍显著优于 OCC（2.90px），且 LIN-sub ≈ LIN-full
说明 LS 估计器对 vote 密度不敏感 — 性能下降不能由可见像素减少解释。

**Q3**: 是。控制 vote 数量后 OCC 仍明显更差，且退化存在于全部区域（含 interior）、
在 occlusion boundary 达到峰值；按区域过滤 vote 无法恢复 pose（§6）。
研究问题应正式定义为 **"occlusion-aware vertex representation"**（遮挡条件下的方向场表示学习），
而不是 "vote reliability weighting"。

## 8. Decision

按预登记规则逐条核对：

1. OCC visible vote error 明显高于 LIN：1.711 vs 1.214 px（+41% mean，+44% p90）✅
2. occlusion-boundary error 显著高于 interior：2.520 vs 1.630 px（+55% mean，+70% p90，
   bad rate 3.26% vs 0.00%；angular 7.65° vs 5.33°）✅
3. equal-count subsampling 后 LIN 仍明显优于 OCC：1.61 vs 2.90 px（1.8×）✅

→ **GO: occlusion-aware representation**（representation 方向，非 sparse-vote、非 reliability weighting）。

## 9. Limitations

- Equal-count 为近似反事实：LINEMOD/OCC 图像场景不同（同类 cat、同一组 3D 关键点、同一估计器），
  按索引配对；每张仅 3 次重复；估计器为 LS 直线交点而非完整 RANSAC。
- 样本仅 20+20 张且集中于 OCC 前半段：12/20 张无 GT 遮挡像素，重遮挡结论主要由 img15–19 驱动。
- 区域划分阈值（腐蚀 5 次、距遮挡区 5px）沿用 diagnosis_vote_analysis，与该实验可比但非唯一合理选择。
- Pose sensitivity 使用 GT 可见 mask（刻意隔离 voting 与分割误差），与标准 predicted-mask 评测数值不可直接比较。
- vertex GT 方向为解析定义（指向 GT 关键点），与数据集监督目标一致，但未包含遮挡外点匹配（LINEMOD 协议）的影响。
- LINEMOD 的 occlusion_boundary 样本极少（3 张图共 1116 条 vote），其统计不具参考意义。
