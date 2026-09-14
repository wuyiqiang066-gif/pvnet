# EXP006 REPORT — Vertex Representation Failure Analysis under Occlusion

> This is a diagnostic experiment, not a proposed method.

**Mechanism: DIRECTION（keypoint 集中于 kp2/kp3/kp7）→ Decision: GO-A — direction-aware / context-aware vertex representation**

## 1. Question

OCC 下 PVNet vertex representation 到底是"方向预测"退化，还是"vector magnitude"退化，
还是两者同时退化？退化是否集中在少数 keypoints / occlusion boundary？

## 2. Protocol

- 纯 inference 诊断：`199.pth`，baseline PVNet，无训练/无代码修改（RANSAC/CUDA/PnP/evaluator 均未触碰）
- 数据：**严格复用 EXP005** 的 20 LINEMOD + 20 OCC 图像（运行时逐张断言与
  `exp005_visibility_diagnosis/results/image_ids.json` 一致）；441k LIN + 250k OCC 条 (pixel × kp) 样本
- 区域划分与 invalid handling 复用 EXP005：interior / normal_boundary / occlusion_boundary（GT amodal）；
  范数 clamp 1e-9（无 NaN）；距 GT 关键点 <1px 的像素剔除（方向无定义）
- 关键架构事实（先核实 `lib/datasets/linemod_dataset.py: compute_vertex_hcoords`）：
  **PVNet 监督的是单位方向向量**（`vertex = vertex / norm`），因此 `pred_mag ≈ 1` 是结构性的，
  监督空间中 GT magnitude ≡ 1。所有指标同时在两个空间报告：
  - **raw 空间**（spec 公式，v_gt = gt_kp − pixel）：magnitude/endpoint 误差被几何位移量主导
  - **unit 空间**（v_gt = unit(gt_kp − pixel)，即监督目标）：隔离"学到的表示"本身，**机制结论以此为准**
- 指标：direction error（deg，bad > 10°）；magnitude error（raw: |‖v_pred‖−‖v_gt‖| 及归一化；
  unit: |‖v_pred‖−1|）；endpoint error（raw: ‖v_pred−v_gt‖；unit: ‖v_pred−v_gt_unit‖，bad > 10px）
- 阈值/seed：见 `results/config.json`；GPU 时间 **3.3 s**

## 3. Direction Error

| split | region | mean° | median° | p90° | bad rate (>10°) |
|---|---|---|---|---|---|
| LINEMOD | all | 2.969 | 1.699 | 6.046 | 4.27% |
| OCC | all | **5.121** | **2.822** | **10.901** | **11.57%** |
| OCC | interior | 5.330 | 3.004 | 11.292 | 12.47% |
| OCC | normal_boundary | 4.848 | 2.714 | 10.267 | 10.50% |
| OCC | occlusion_boundary | **7.653** | 3.368 | **19.260** | **21.15%** |

OCC 全体方向误差 +72%（p90 +80%，bad rate 2.7×）；occlusion_boundary 最差且尾部恶化最重（p90 19.3°）。
注意 OCC 连 interior 都比 LINEMOD 差（5.33° vs 2.96°）。

## 4. Magnitude Error

| split | region | pred_mag mean | unit dev \|m−1\| (mean / median / p90) | raw abs err mean (px) | raw norm err mean |
|---|---|---|---|---|---|
| LINEMOD | all | 0.9981 | 0.0188 / 0.0122 / 0.0335 | 33.30 | 0.9554 |
| OCC | all | 0.9962 | 0.0203 / 0.0123 / 0.0362 | 27.44 | 0.9469 |
| OCC | interior | 0.9948 | 0.0206 | 24.76 | 0.9447 |
| OCC | occlusion_boundary | 0.9847 | **0.0273** / 0.0143 / 0.0472 | 26.15 | 0.9461 |

- **magnitude 轴在结构上是退化的**：pred_mag ≈ 1.0（ mean 0.996–0.998），偏差仅 ~2%。
- LIN vs OCC unit-space 偏差 0.0188 vs 0.0203（相对 +8%）——量级可忽略，且
  occlusion proximity 下 raw magnitude 无趋势（见 §7）。
- raw 空间的 magnitude "误差"（LIN 33.3 / OCC 27.4 px）恒等于 ⟨像素到关键点距离⟩ − 1
  （LIN 34.3 / OCC 28.4 px，见 endpoint_error.csv），且 OCC 反而更低——纯属几何，
  不是表示退化。normalized raw error ≈ 0.95（两 split 相同），同样无判别力。

## 5. Endpoint Error

| split | region | unit mean (px) | unit median | unit p90 | unit bad (>10px) | raw mean (px) | raw bad (>10px) |
|---|---|---|---|---|---|---|---|
| LINEMOD | all | 0.058 | 0.036 | 0.111 | 0% | 33.31 | 91.5% |
| OCC | all | **0.092** | **0.054** | **0.195** | 0% | 27.45 | 88.7% |
| OCC | interior | 0.096 | 0.057 | 0.202 | 0% | 24.77 | 87.9% |
| OCC | normal_boundary | 0.088 | 0.052 | 0.184 | 0% | 28.64 | 89.1% |
| OCC | occlusion_boundary | **0.133** | 0.063 | **0.338** | 0% | 26.18 | 88.7% |

- unit 空间：OCC +59%（0.058→0.092），occlusion_boundary 是 LINEMOD 全体水平的 2.3 倍。
- raw 空间：endpoint ≈ 像素到关键点距离（89–92% 超 10px，两 split 几乎相同）——
  由单位方向场架构决定，不反映退化，仅作透明记录。

## 6. Per-Keypoint Analysis

direction error（deg，all fg，OCC vs LIN）：

| kp | LIN mean | OCC mean | OCC/LIN | OCC bad rate |
|---|---|---|---|---|
| kp0 | 2.21 | 4.13 | 1.9× | 8.6% |
| kp1 | 2.32 | 2.96 | 1.3× | 3.0% |
| **kp2** | 2.12 | **11.28** | **5.3×** | **39.4%** |
| **kp3** | 2.18 | **7.76** | **3.6×** | 18.3% |
| kp4 | 2.76 | 4.45 | 1.6× | 9.1% |
| kp5 | 4.33 | 2.36 | 0.5× | 2.8% |
| kp6 | 4.29 | 3.09 | 0.7× | 4.0% |
| **kp7** | 3.03 | **6.51** | **2.2×** | 12.9% |
| kp8 | 3.50 | 3.56 | 1.0× | 6.1% |

endpoint error（unit space）同样模式：OCC kp2 0.196 px（LIN 0.043，4.6×）、kp3 0.134、kp7 0.114；
其余 6 个 kp 与 LINEMOD基本持平（pooled 3.42° vs 3.23°，+6%）。

结论：退化**高度集中在 kp2 / kp3 / kp7**（9 个中 3 个）；LINEMOD自身最难的 kp5/kp6 在 OCC 上反而最好。
聚合层面的 OCC 退化（§3 +72%）大部分由这 3 个 kp贡献。

## 7. Occlusion Proximity Analysis

9/20 张 OCC 图像含 GT 遮挡像素（其余 11 张无遮挡 → 该分析只覆盖 9 张；LINEMOD 无遮挡 → 跳过）。
距离定义为到最近遮挡像素（amodal & ~visible）的 L2 距离（复用 EXP005 距离场）：

| 距遮挡像素 | votes | e_vote mean (px) | direction mean° | mag raw mean (px) | endpoint unit mean |
|---|---|---|---|---|---|
| 20–inf px | 30,306 | 1.243 | 3.52° | 28.32 | 0.067 |
| 10–20 px | 26,284 | 1.490 | 4.40° | 26.50 | 0.081 |
| 5–10 px | 14,040 | 1.942 | 5.89° | 25.99 | 0.104 |
| 2–5 px | 7,775 | 2.410 | 7.33° | 26.13 | 0.126 |
| 0–2 px | 4,654 | **2.711** | **8.20°** | 26.17 | **0.143** |

**方向误差与 vote 误差随接近遮挡边界单调连续增长**（vote error +118%，direction +133%，
endpoint +113%，从 >20px 到 0–2px），而 raw magnitude 完全平坦（26–28 px，几何量）。
这是"方向表示随遮挡接近连续退化"的直接证据。

## 8. Direction-only / Magnitude-only Diagnostic

离线假设分解（**非模型性能，未送入 RANSAC/PnP**）：

| scope | 空间 | original endpoint | direction-only（保预测方向，GT magnitude） | magnitude-only（保 GT 方向，预测 magnitude） |
|---|---|---|---|---|
| LIN all | unit | 0.0577 | 0.0514 | 0.0188 |
| OCC all | unit | 0.0923 | **0.0881** | 0.0203 |
| OCC occlb | unit | 0.1325 | 0.1296 | 0.0273 |
| LIN all | raw | 33.31 | 1.22 | 33.30 |
| OCC all | raw | 27.45 | **1.73** | 27.44 |
| OCC occlb | raw | 26.18 | **2.58** | 26.15 |

解读（按 spec 的提问方式）：

- "如果 magnitude 完美，剩多少误差？" → direction-only 项。unit 空间 0.0881（占原误差 95%）；
  raw 空间 1.73 px（原 27.45 px 的 6%——因为 raw 误差几乎全是"位移量本身没被回归"这一架构事实）。
- "如果 direction 完美，剩多少误差？" → magnitude-only 项。unit 空间仅 0.0203（OCC occlb 也只有 0.0273）。
- 即：**方向分量解释了表示误差的绝大部分**；magnitude 分量在两个空间下都可忽略
  （unit: ~2% 的常量偏差；raw: 架构性缺失，与遮挡无关）。

## 9. Mechanism Interpretation

1. **Magnitude 被排除**：(a) 架构上监督单位向量，pred_mag ≈ 1 是结构性的；
   (b) |m−1| 仅 ~2% 且 LIN/OCC 差异相对 +8%；(c) 对遮挡距离无梯度（§7 平坦）；
   (d) unit 空间中 magnitude 分量只占 endpoint 误差 ~5%。
2. **Direction 是退化轴**：OCC +72%（interior 也差），occlusion_boundary 最差（bad rate 21%），
   随遮挡接近连续单调增长（3.52°→8.20°）。
3. **且退化 keypoint 集中**：kp2/kp3/kp7 贡献了大部分聚合退化；剔除这 3 个 kp后
   其余 6 个 kp的 OCC/LIN 差距仅 +6%。occlusion boundary 的恶化大概率也是这些 kp
   的可见vote 所在（ kp与其被遮挡区域的空间关系决定）。
4. 与 EXP005 一致并深化：EXP005 证明"不是 vote 数量"，本实验定位到"是方向场，
   且集中在特定 kp与遮挡邻域"。

## 10. Decision

预登记规则核对：

- **GO-A 条件全部满足**：
  1. OCC direction error 明显高于 LIN：5.12° vs 2.97°（+72%；p90 10.90° vs 6.05°；bad 11.6% vs 4.3%）✅
  2. occlusion boundary direction error 明显高于 interior：7.65° vs 5.33°（+44%；bad 21.2% vs 12.5%）✅
  3. direction-only endpoint error 明显优于原始 endpoint error：raw 空间 1.73 vs 27.45 px（−94%；
     unit 空间方向分量占 endpoint 误差 ~95%）✅
- GO-D 条件同时部分成立（大部分 kp正常、3 个 kp严重退化），作为 GO-A 的分布特征记录，
  应体现在后续方法设计中（ kp条件化或遮挡上下文化的方向表示）。

→ **DECISION: GO-A — direction-aware / context-aware vertex representation**
（mechanism：DIRECTION，keypoint 集中于 kp2/kp3/kp7；magnitude 排除）

## 11. Limitations

- 样本量 20+20；OCC 仅 9/20 张含 GT 遮挡像素，proximity 分析覆盖这 9 张（46k/250k votes）。
- magnitude 轴的诊断受架构约束：PVNet 不回归 displacement magnitude，"magnitude 退化"
  只能在 |m−1| 偏差意义上诊断；若未来考虑位移回归变体，本实验的 raw 空间数值不可直接引用。
- per-kp 像素池在 LIN/OCC 间分布不同（遮挡改变可见像素分布），跨 split 的 per-kp 对比是近似同分布假设。
- LINEMOD occlusion_boundary 仅 3 张图 1,116 条 vote，其统计不具参考意义。
- direction-only / magnitude-only 为离线假设分解，**不是模型可达到的性能**，未进入 RANSAC/PnP。
- keypoints 为 FPS+center 顺序，无语义标注，仅以 kp0…kp8 报告。
