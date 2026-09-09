# Report — exp001 Reliability-aware Pixel Voting

状态：**训练进行中**（stage1 于 2026-09-09 10:13 启动，本报告的指标栏在训练完成后由
`--test_model` 结果回填）。

## 1. Modified files

全部为新增文件，baseline 已跟踪文件零修改（`git diff HEAD` 为空，`git show --stat 11f6acc`）。

| file | 内容 |
|---|---|
| `lib/networks/model_repository_exp001.py` | `Resnet18_8sReliability(Resnet18_8s)`：`rel_head`（3×3+1×1）输出 `[B,9,H,W]`；baseline 权重 `strict=False` 直接加载 |
| `lib/ransac_voting_gpu_layer/ransac_voting_weighted.py` | `ransac_voting_layer_v3_weighted`：复用同一 CUDA kernel，加权打分 `score(h,k)=Σᵢ rᵢᵏ·inlier(i,h,k)` + 加权 LS 精修 |
| `tools/train_linemod_exp001.py` | 训练/测试入口：two-stage、label clamp、每 epoch label 统计、baseline 开关 |
| `tools/visualize_exp001.py` | Part 7 可视化（heatmap / kp error / weight distribution） |
| `configs/exp001_reliability_voting.json` | 开关与超参（`use_reliability_vote` 等） |

未动：dataset、PnP（`evaluation_utils.py`）、evaluation 指标、CUDA kernel、
`tools/train_linemod.py`、`model_repository.py`、`ransac_voting_gpu.py`。

## 2. Training protocol（two-stage）

| | Stage 1（rel warm-up） | Stage 2（joint fine-tune） |
|---|---|---|
| 初始化 | baseline `199.pth`（strict=False，rel_head 随机） | stage1 最优 |
| 可训练 | 仅 `rel_head`（backbone+vertex head 冻结，含 BN running stats） | 全部参数 |
| loss | L_seg + L_vertex + λ·L_rel（冻结分支无梯度流入） | 同左 |
| epochs | 0–7（`stage1_epochs=8`） | 8–199 |
| checkpoint | `model/stage1_rel_warmup/<ep>.pth` | `model/stage2_joint/<ep>.pth` |

Pseudo label：`r* = exp(-e²/σ²)`，`e`=GT 关键点到**预测**（detach）投票线垂直距离，
clamp：`e≤tau1(0.5px)→1`，`e≥tau2(30px)→0`，σ=3px。
每 epoch 统计写入 `results/reliability_label_stats.jsonl` + `results/label_hist/*.npz`。

推理：RANSAC 框架不变（同假设采样/同置信度停止/同 inlier 阈值 0.99），
仅聚合改为加权；`w<min_weight(0.001)` 剔除；某 kp 全被剔除时回退 uniform。

命令：
```bash
python tools/train_linemod_exp001.py --cfg_file configs/exp001_reliability_voting.json --linemod_cls cat
python tools/train_linemod_exp001.py --cfg_file configs/exp001_reliability_voting.json --linemod_cls cat --test_model --use_test_set
```

## 3. Baseline comparison（LINEMOD cat）

baseline 参考值（exp000，同一评估协议）：

| split | metric | baseline | exp001 |
|---|---|---|---|
| LINEMOD test (1002) | ADD(-S) | 80.64 | _pending_ |
| LINEMOD test (1002) | 2D Projection | 99.80 | _pending_ |
| OCC LINEMOD test | ADD(-S) | 16.85 | _pending_ |
| LINEMOD / OCC | 5cm5deg | _refill_ | _pending_ |

结果文件：`results/result_test_e<ep>.txt`、`results/result_occ_test_e<ep>.txt`、
`results/result_summary.txt`。

## 4. Ablation

| 配置 | 说明 | ADD(-S) LINEMOD | ADD(-S) OCC |
|---|---|---|---|
| PVNet（baseline_v1） | equal-weight voting | 80.64 | 16.85 |
| PVNet + reliability | weighted voting，two-stage | _pending_ | _pending_ |

控制组（诊断实验已有）：oracle 上限 LINEMOD 79.6→99.4（top30%）、OCC 17.9→36.4；
`use_reliability_vote=false` 从同一入口可复现 exact baseline（uniform 权重下与
`ransac_voting_layer_v3` 输出 diff=0.0，见 Part 5）。

## 5. Smoke test 记录（logs/smoke_test.log）

1. uniform weight：weighted vs baseline **max diff = 0.0**（要求：完全一致 ✓）
2. 单 batch forward：seg (1,2,256,256) / vertex (1,18,256,256) / rel_logits (1,9,256,256) ✓
3. backward：rel_head grad 0.3226、backbone conv1 grad 0.7283（L_rel 不回漏 baseline 分支）✓
4. clamp：公式精确匹配（max |r−expected|=0），tau1 饱和 / tau2 置零验证通过 ✓
5. 真实数据 + 199.pth：baseline kp 误差 0.86–1.09px（与 exp000 一致）✓

## 6. 早期信号（stage1 epoch 0）

- label 统计：mean 0.8328 / std 0.2294 / min 0 / max 1 —
  冻结 baseline 的 vertex 质量高，r* 未坍缩到 0（冷启动风险被 stage1 化解）。
- rel_head 随机初始化时四区域权重均 ≈0.50（sigmoid(0)），符合预期；
  训练后 `results/figures/weight_distribution.*` 应呈现 interior > boundary ≈
  occlusion_boundary 的梯度。

## 7. Failure cases（训练完成后生成）

- 收集方式：`results/kp_error_comparison.csv`（vote 级）找 weighted 仍 >20px 的图；
  PnP 级用 `data/record/cat_exp001_reliability_voting.log`（Evaluator per-frame）。
- 重点画像：遮挡边界像素高权重、rel head 给边界高置信的假阳性、
  farthest kp 在自遮挡区的系统性偏差。
- 图：`results/figures/reliability_heatmap_OCC_*.png`。

## 8. 后续可行性

- **instance-aware voting（支持）**：本实现的加权只依赖像素外观证据
  （per-keypoint 通道已隐式区分 kp），下一步把 instance 证据（如 mask id /
  与 object center 的相对几何）注入 rel_head 输入即可，聚合公式不变。
- **amodal voting（支持，需注意约束）**：OCC 的 amodal mask 仅作为训练期
  region 统计/加权监督使用（`amodal_masks/`），推理管线不依赖 amodal 输入，
  不违反"不改 dataset/eval"约束。
- 不引入新网络结构、不改 CUDA kernel、不改 PnP 的约束在本框架下均可保持。

## 9. Git 恢复

```bash
git checkout exp001_reliability_voting   # HEAD 1523815, tag exp001_v0
git checkout baseline_v1                  # 或 pvnet_baseline_verified 分支恢复 baseline
```

checkpoints/logs/npy/figures 均不入 git（`.gitignore`），实验可由
config.json + 代码 commit 完整复现。
