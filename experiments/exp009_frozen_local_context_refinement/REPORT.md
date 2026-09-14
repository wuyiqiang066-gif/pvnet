# EXP009 — Frozen Local Context Direction Refinement

Branch: `exp009_frozen_local_context_refinement` (based on `exp008_oracle_context_diagnosis`)
Cat-only PVNet baseline (`199.pth`), backbone/heads/BN frozen, no baseline modification.

## 1. Research Question

EXP008 showed the GT direction field is locally highly redundant (r=3 oracle
0.569° on OCC vs 5.121° baseline). **Can a tiny learnable local-context
residual module exploit that redundancy while the entire PVNet stays frozen?**

This is a mechanism validation of the EXP008 oracle headroom — not an
architecture proposal.

## 2. Design

- Frozen PVNet 199.pth: entire network frozen (backbone, conv8s/4s/2s,
  convraw incl. vertex head, BN running stats). Trunk forward runs under
  `torch.no_grad()`, all params `requires_grad=False` → zero gradient to the
  baseline; original checkpoint untouched.
- Shared feature F = 32-ch full-resolution feature (output of
  `convraw[0:3]`, the input of the original 1×1 head); frozen vertex head
  output `v_base`.
- Trainable residual module (**914 params**):
  `Δv = PW1×1(DW3×3(F))` (3×3 depthwise, 32ch → 1×1 pointwise 32→18),
  `v_refined = normalize(v_base + Δv)`.
  Pointwise conv **zero-initialized** → step 0 is exactly the baseline.
- Loss: the **original PVNet vertex loss** (`lib.utils.net_utils.smooth_l1_loss`,
  σ=1, fg-mask weighted) on `v_refined` vs the dataset GT vertex targets.
  No new supervision, no new loss terms, no attention/transformer.
- Training: 20 LINEMOD cat images (EXP005/006/008 IDs, asserted),
  augment=False, batch=1, **20 epochs** (400 steps), Adam lr=1e-3, seed=0.
  Train loss 0.00296 → 0.00267 (monotone decrease; see `results/train_log.csv`).
- Evaluation: identical evaluator for Control and EXP009 — same 20 LIN + 20 OCC
  images (asserted against EXP005 image_ids.json), same preprocessing, same
  direction-error metric as EXP006/008 (visible fg pixels, |K−p|≥1px, 9 kps
  pooled), same checkpoint, same region definitions. RANSAC/PnP untouched and
  not exercised (direction-level validation only). Control reproduced
  EXP006/008 exactly in-run (LIN 2.969 / OCC 5.121, counts identical).
- Runtime: 25.7 s total (GPU), within budget. No epochs, kernels, dilation or
  architectures were added after seeing results.

## 3. Required Results

Direction error (deg), `results/direction_error_summary.csv`:

| Method | LIN mean | OCC mean |
|---|---:|---:|
| 199.pth baseline | 2.969 | 5.121 |
| EXP009 | **2.807** | **6.437** |

- OCC relative change: **+25.7% error (degradation)** — fails the ≥10%
  improvement condition (target ≤4.609°).
- LIN relative change: **−5.45% error (improvement)** — satisfies the ≤5%
  degradation condition, but both conditions must hold.

OCC regions (mean direction error, deg), `results/occ_region_direction_error.csv`:

| Region | Control | EXP009 |
|---|---:|---:|
| interior | 5.330 | 6.540 |
| normal boundary | 4.848 | 6.187 |
| occlusion boundary | 7.653 | 9.248 |

Degradation is roughly **uniform across all OCC regions** (+22.7% / +27.6% /
+20.8%), i.e. it is a domain-wide miscalibration of the learned filter, not a
boundary-localized failure.

## 4. GO / STOP Decision

**STOP** (pre-registered rule: GO iff OCC ≤ 4.609° AND LIN ≤ 3.118°; OCC
condition fails by a wide margin). Per the pre-registered protocol: no extra
epochs, no kernel/dilation changes, no attention, no experiment expansion.

## 5. Interpretation (mechanism level)

1. **In-domain, the frozen-trunk protocol works and the signal is real.** The
   914-param spatially-invariant linear filter improved LIN by 5.45% within
   400 steps — while EXP007's tiny-budget whole-network training *degraded*
   even its Control. So (a) frozen-trunk + zero-init tiny residual trains
   stably, and (b) frozen PVNet features do contain locally exploitable
   direction redundancy. This is the positive part of the mechanism check.
2. **But the redundancy is not accessible via a spatially-uniform local
   linear readout under occlusion.** The learned filter applies the same
   3×3 aggregation regardless of neighborhood validity; under occlusion the
   local feature neighborhood is one-sided/truncated, and the filter
   mis-corrects — degrading OCC by +25.7%, uniformly across regions.
3. **Consistency with EXP008.** EXP008's local oracle consumed the *GT
   directions of neighbors* — context that is valid by construction. A
   learned filter reading frozen features has no such validity guarantee.
   The EXP008 headroom is therefore real but is **not** unlocked by
   context-agnostic local aggregation; a context mechanism would need to be
   occlusion/validity-aware to access it.
4. This STOP is more informative than EXP007's: the training protocol here is
   provably stable (in-domain improvement), so the cross-domain failure is
   attributable to the mechanism (spatially-uniform local linear residual),
   not to optimization collapse.

## 6. Limitations

1. Single seed, 20 training images, augment=False, 20 epochs, one linear
   depthwise-separable residual form; the pre-registered budget forbids
   sweeping these.
2. The module reads only the 32-ch frozen feature at 3×3 support; larger or
   validity-aware context was intentionally not tried (budget rule).
3. Direction-level validation only; RANSAC/PnP pose metrics were not part of
   the pre-registered readout and were not run.
4. In-domain LIN improvement (−5.45%) may partly reflect fitting the 20
   training images (the filter is spatially invariant and only 914 params,
   limiting but not eliminating this risk); no held-out LINEMOD split was
   used.

> This experiment is a mechanism validation on the cat-only PVNet baseline and does not establish cross-object generalization.
