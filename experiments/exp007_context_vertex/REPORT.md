# EXP007 REPORT — Context-Enhanced Vertex Representation

> EXP007 is the first method-validation experiment following the diagnostic studies EXP001–EXP006.
>
> The experiment is intentionally limited to 20 LINEMOD images and 20 OCC images, one seed, and 20 epochs. It is a screening experiment, not a final benchmark.

**Decision: STOP**（预登记阈值全部未达标；mechanism not supported）

## 1. Hypothesis

如果给 PVNet vertex prediction 提供局部 spatial context（而不是只依赖当前 feature location），
能否改善 occlusion-induced vertex direction error，而不破坏原有 geometric representation？

依据（EXP005–006）：OCC 退化不是 vote 数量问题、不是 reliability 问题、不是 magnitude 问题，
主要是 vertex **direction** 退化，集中于 kp2/kp3/kp7，并随接近 occlusion boundary 连续恶化。

## 2. Method

在 verified baseline（`pvnet_baseline_verified` @ bfd3e15）上新增一个文件
`context_model.py`（未修改任何 baseline 文件）：

```text
shared feature fm (32ch, raw 分辨率)
    context = PW(DW(fm))          # DW 3×3 depthwise + PW 1×1 pointwise, C→C
    f_context = fm + context      # residual fusion
seg_pred = convraw(cat[fm, img])[:, :2]          # 与 baseline 逐位一致
ver_pred = vertex_head(cat[f_context, img])[:, 2:]  # vertex_head = convraw 权重拷贝
```

- 由于 baseline 的 convraw 是 seg/vertex 共享头，为保证 segmentation 路径完全不变，
  vertex 使用共享头的**权重拷贝**作为独立 vertex head（初始时 f_context ≈ fm → 函数等价）
- 输出形状不变：seg [B,2,H,W]，vertex [B,18,H,W]；GT 仍为单位方向向量；两个 loss 均未改动
- 无 attention / pooling / dilation / 多尺度 / 新 branch
- 参数量（`checkpoints/exp007/init_info.json`）：
  baseline 12,957,748；**context module 1,344**；vertex head 拷贝 10,804；**新增共 12,148（+0.094%）**
- 初始化：baseline 参数从 `199.pth` 逐位加载（load 后 assert 仅 context 3 个张量缺失）；
  context module 按 repo 新层惯例 `normal(0, 0.01)`、bias=0（`Resnet18_8s._normal_initialization` 同款）；
  验证：将 PW 置零后 context 模型与 baseline 输出 **torch.equal 完全一致**

## 3. Experimental Protocol

- **branch 隔离**：从 `pvnet_baseline_verified`（bfd3e15）新建 `exp007_context_vertex`，
  不基于 EXP001–006 的任何模型改动（`baseline_commit.txt` 记录）
- **两组对照，完全同预算**：
  - Control A：原始 PVNet，199.pth 初始化，20 epochs，seed=0
  - EXP007：context 模型，同一 199.pth 初始化，20 epochs，seed=0
  - 同 20 张 cat real-train 图像（train_real_set 前 20，deterministic）、同数据顺序
    （loader 前 re-seed，逐 epoch loss 可对齐到 ~1e-4）、同增广（baseline aug_cfg）、
    同 optimizer/lr（Adam 1e-3，decay 0.5/20ep）、同 batch size（32, ImageSizeBatchSampler）、
    同 loss 权重（vertex_loss_ratio=1.0）、num_workers=0（确定性）
- **评测**：20 LINEMOD val + 20 OCC（与 EXP005/006 完全相同的 deterministic IDs，运行时逐张 assert）；
  pose = predicted-mask + 原始 `ransac_voting_layer_v3` + 原始 `pnp`；
  vertex 诊断 = EXP006 同款指标（direction/endpoint 监督空间、region、per-kp、proximity bins）
- 初始化来源记录：199.pth（`data/model/cat_linemod_train/199.pth`），两组一致
- GPU 时间：训练 2×~27s + 评测 3×~4s ≈ **1.2 分钟**

## 4. Pose Results

| model | LIN ADD-S pass@0.1d | LIN ADD-S mean | OCC ADD-S pass@0.1d | OCC ADD-S mean | LIN proj pass@5px | OCC proj pass@5px |
|---|---|---|---|---|---|---|
| 199.pth (ref) | **1.000** | **7.4 mm** | **0.650** | **149.7 mm** | 1.000 | 0.800 |
| Control A | 0.750 | 312.7 mm | 0.350 | 158.3 mm | 0.950 | 0.350 |
| EXP007 | 0.600 | 410.9 mm | 0.400 | 157.9 mm | 0.950 | 0.350 |

两个 fine-tune 臂都显著差于 199.pth —— 规定的协议（lr=1e-3 在收敛权重上 fine-tune 20 张图 20 epochs）
本身就造成大幅退化（与 EXP001-S 的教训一致：lr=1e-3 在已收敛权重上继续训练会破坏性能）。
这正是设置同预算对照的原因：EXP007 vs Control A 才是 context 模块的净效应。
EXP007 相对 Control A：OCC pass 0.400 vs 0.350（20 张图上 ±1 张即 ±0.05，属噪声），
LIN pass 反而 0.600 vs 0.750。**无可辨认的 pose 收益**。

## 5. Direction Error Results

| model | LIN all (deg) | OCC all (deg) | OCC interior | OCC normal bd | OCC occl bd | OCC bad>10° |
|---|---|---|---|---|---|---|
| 199.pth (ref) | 2.969 | 5.121 | 5.330 | 4.848 | 7.653 | 11.57% |
| Control A | 3.679 | 7.911 | 7.901 | 7.610 | 12.046 | 23.86% |
| EXP007 | 3.728 | **7.905** | 7.824 | 7.628 | **12.108** | 23.45% |

EXP007 vs Control A（预登记第一门槛）：OCC direction error 变化 **+0.08%**（要求 ≥10% 改善）→ **FAIL**。
OCC occlusion_boundary：12.108 vs 12.046（+0.5%，更差）→ FAIL。
LINEMOD：3.728 vs 3.679（+1.3%，对照内属噪声水平；但对 199.pth 为 +25.6% 退化）。

## 6. Keypoint Analysis

OCC direction error（deg）：

| kp | 199.pth | Control A | EXP007 | EXP007 vs Control A |
|---|---|---|---|---|
| kp2 | 11.28 | 16.86 | 16.74 | −0.7% |
| kp3 | 7.76 | 15.61 | 15.70 | +0.6% |
| kp7 | 6.51 | 7.07 | 7.58 | **+7.2%** |
| other6 pooled | 3.42 | 4.86 | 4.85 | −0.3% |

kp2/kp3/kp7 pooled：Control A 13.18° vs EXP007 13.34°（−1.2%，更差）→ 预登记 ≥15% 改善 **FAIL**。
context 模块对 EXP006 锁定的最差 keypoints 无任何针对性改善。

## 7. Occlusion Proximity Analysis

OCC direction error 随遮挡距离（px）：

| bin | 199.pth | Control A | EXP007 |
|---|---|---|---|
| 20–inf | 3.52° | 6.71° | 6.82° |
| 10–20 | 4.40° | 7.77° | 7.88° |
| 5–10 | 5.89° | 9.49° | 9.61° |
| 2–5 | 7.33° | 11.38° | 11.46° |
| 0–2 | 8.20° | 13.17° | 13.19° |

 EXP007 未减弱 error 随 occlusion proximity 增长的趋势（曲线与 Control A 几乎重合，
 各 bin 差异 <1.3%，方向不一致）。

## 8. Mechanism Check

- direction error 无改善（0.08%，噪声级）→ **hypothesis not supported**
- pose 无一致改善（OCC +1 张、LIN −3 张，均为噪声）→ 也不存在"representation improvement
  without pose gain"的情形
- 训练日志佐证：两臂逐 epoch loss 几乎重合（epoch 18: seg 0.00460 vs 0.00458，
  ver 0.00327 vs 0.00327）—— context 残差在 20 个 optimizer step 内基本未被有效训练
  （模块从 ~0 初始化，1 batch/epoch），其影响淹没在 fine-tuning 退化噪声中

## 9. Success Criteria

预登记阈值逐条核对（对比同预算 Control A）：

1. OCC mean direction error 改善 ≥10%：实际 **+0.08%** → FAIL
2. kp2/kp3/kp7 pooled 改善 ≥15%：实际 **−1.2%** → FAIL
3. LIN direction error 退化 ≤5%（vs Control A）：+1.3% → 名义 PASS（但 vs 199.pth 为 +25.6%，
   来自协议本身而非 context 模块）
4. endpoint error 同方向改善：OCC all 0.1507→0.1533（+1.7%，更差）→ FAIL

**0/4 核心阈值达标 → STOP**（不调参、不堆模块）。

## 10. Decision

**STOP。**

理由：direction error 未改善（噪声级差异），核心阈值全部未达标。
同时记录两条结构性事实供后续判断：

1. 规定的筛选协议（20 图、20 epochs、lr=1e-3、~20 个 optimizer step）本身使两臂都大幅
   退化于 199.pth；在该协议下任何 ≤12k 参数的新模块都无法产生可检测信号。
   本实验无法区分"context module 无效"与"训练预算不足以训练出 context"。
2. 若未来重试该方向，需要先解决预算问题（更多 steps / 更低 lr / 冻结 trunk 只训 context），
   这超出本实验预登记范围，故按规则停止并报告。

## 11. Limitations

- 训练预算极小（20 epochs × 1 batch ≈ 20 steps）：context 残差初始化接近恒等映射，
  20 步内学习量可能不足以产生任何可检测的行为差异 —— 阴性结果的解释力受限（见 §10）
- 单 seed、20+20 张图：pose 与诊断指标的噪声量级为 ±1 张（pass ±0.05）/ ±0.1°（direction）
- fine-tune 协议使两臂相对 199.pth 退化，"LIN 不退化"只能在对照内衡量
- context module 仅作用于 32ch 共享特征图的 3×3 邻域；若 context 需要更大感受野或更高层
  语义，本设计无法回答
- vertex head 为 convraw 权重拷贝（未重新独立训练），两个头的 BN 统计量在训练中略有分化
