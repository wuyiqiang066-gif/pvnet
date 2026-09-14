"""EXP011: Predicted-Direction Global Keypoint Diagnosis (offline, no training).

Research question (EXP008 + EXP010 follow-up):
  EXP008: the GT direction field of visible pixels recovers keypoints almost
          exactly via object-level ray intersection (0.001 deg direction error).
  EXP010: validity-aware LOCAL selection cannot beat blind local aggregation.
  Question: using PVNet's OWN predicted visible-pixel directions (no GT
  directions, no GT keypoint), can object-level geometric aggregation recover
  the keypoint well enough to regenerate per-pixel directions?

Estimators per (image, keypoint), ray set = visible fg pixels with
|K_k - q| >= 1px (identical to the EXP008 global-oracle ray set; only the
ray CONTENT differs: predicted unit directions instead of GT):
  A. Global-LS     : least-squares intersection of all predicted rays
                     (A = sum (I - u u^T), b = sum (I - u u^T) q, tiny ridge).
  B. Global-Robust : the existing PVNet RANSAC voting mechanism
                     (lib.ransac_voting_gpu_layer.ransac_voting_gpu.ransac_voting_layer_v3,
                     baseline parameters round_hyp_num=128, inlier_thresh=0.99,
                     max_num=100, called verbatim on the GT visible mask;
                     torch.manual_seed(0) before each call for determinism;
                     mechanism unchanged).
Then regenerate target-pixel directions: d_hat = normalize(K_hat - p), and
measure angle(d_hat, normalize(K - p)) on the EXP006/008 pool (visible fg,
|K-p| >= 1px, 9 kps pooled). Keypoint localization error = ||K_hat - K|| px.

Protocol: 20 LIN + 20 OCC cat images (EXP005/006/008/009/010 IDs asserted),
seed=0, 199.pth, no training, GPU < 2 min. In-run control: the same LS
estimator on GT rays must reproduce the EXP008 global oracle (~0.001 deg).
Proximity bands (OCC): dist_occ (EXP005/006 definition) [0,5)/[5,20)/[20,inf).

Pre-registered decision (operationalized):
  GO-object-level  iff  min(OCC mean of Global-LS, OCC mean of Global-Robust)
                           <= 0.8 x 5.121 = 4.097 deg   (>=20% relative improvement)
  STOP-object-level otherwise.
Both estimators are reported; the rule uses the better one (stated up front to
avoid cherry-picking).

Run from repo root:
  python experiments/exp011_predicted_direction_global_diagnosis/exp011_global_diagnosis.py
"""
import os, sys, json, re, csv, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import DataLoader, SequentialSampler

from lib.networks.model_repository import Resnet18_8s
from lib.datasets.linemod_dataset import LineModDatasetRealAug, VotingType
from lib.utils.data_utils import LineModImageDB, OcclusionLineModImageDB
from lib.utils.config import cfg
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v3

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(EXP_DIR, 'results')
os.makedirs(RES_DIR, exist_ok=True)

N_IMG = 20
KP = 9
EPS = 1e-9
RIDGE = 1e-9
LIN_BASE, OCC_BASE = 2.969, 5.121                    # EXP006/008 reference
GO_OCC_MAX = 0.8 * OCC_BASE                          # 4.097 deg
BANDS = ((0.0, 5.0), (5.0, 20.0), (20.0, np.inf))
ROUND_HYP_NUM, INLIER_THRESH, MAX_NUM = 128, 0.99, 100   # baseline eval wrapper params

torch.manual_seed(0); np.random.seed(0)
T0 = time.time()
LOG = open(os.path.join(RES_DIR, 'run.log'), 'w', buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + '\n')

json.dump(dict(cls='cat', n_img=N_IMG, ckpt='data/model/cat_linemod_train/199.pth',
               ray_set='visible fg pixels (GT mask) with |K_k - q| >= 1px, identical to '
                       'EXP008 global-oracle ray set; ray content = PREDICTED unit directions',
               estimator_A='Global-LS: LS intersection of predicted rays (ridge 1e-9)',
               estimator_B='Global-Robust: existing PVNet ransac_voting_layer_v3 called '
                           'verbatim (round_hyp_num=128, inlier_thresh=0.99, max_num=100), '
                           'GT visible mask, torch.manual_seed(0) per call, mechanism unchanged',
               target='d_hat = normalize(K_hat - p); angle vs GT direction; EXP006/008 pool',
               control='same LS estimator on GT rays must reproduce EXP008 global oracle',
               go_stop=f'GO-object-level iff min(OCC LS, OCC Robust) <= {GO_OCC_MAX:.3f} '
                       f'(>=20% vs baseline 5.121); else STOP-object-level'),
          open(os.path.join(RES_DIR, 'config.json'), 'w'), indent=2)

device = 'cuda'
net = Resnet18_8s(ver_dim=KP * 2, seg_dim=2)
sd = torch.load(os.path.join(ROOT, 'data/model/cat_linemod_train/199.pth'), map_location='cpu')
net.load_state_dict(sd['net'])
net = net.to(device).eval()

AUG_CFG = json.load(open(os.path.join(
    ROOT, 'configs', 'exp001_reliability_voting_scratch.json')))['aug_cfg']
AMODAL_DIR = {'linemod': os.path.join(cfg.LINEMOD, 'cat', 'amodal_mask'),
              'occ': os.path.join(cfg.OCCLUSION_LINEMOD, 'amodal_masks', 'cat')}
EXP5_IDS = json.load(open(os.path.join(
    ROOT, 'experiments', 'exp005_visibility_diagnosis', 'results', 'image_ids.json')))
EXP5_RGB = {s: [r['rgb'] for r in EXP5_IDS['image_ids'][s]] for s in ('linemod', 'occ')}


def angle_deg(a, b):
    dot = np.clip((a * b).sum(-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def mstats(a):
    a = np.asarray(a, np.float64); a = a[np.isfinite(a)]
    return dict(count=int(a.size), mean=float(a.mean()) if a.size else float('nan'),
                median=float(np.median(a)) if a.size else float('nan'),
                p90=float(np.quantile(a, 0.9)) if a.size else float('nan'))


def ls_intersection(coords_q, u_q, ridge=RIDGE):
    """LS ray intersection: returns (2,) K_hat."""
    M = np.eye(2)[None] - u_q[:, :, None] * u_q[:, None, :]       # (Q,2,2)
    A = M.sum(0)
    b = (M @ coords_q[:, :, None]).sum(0).ravel()
    try:
        return np.linalg.solve(A + ridge * np.eye(2), b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A + ridge * np.eye(2), b, rcond=None)[0]


def build_loader(split):
    if split == 'linemod':
        db = LineModImageDB('cat', has_render_set=False, has_fuse_set=False)
        db_set, prefix = db.val_real_set[:N_IMG], cfg.LINEMOD
    else:
        db = OcclusionLineModImageDB('cat')
        db_set = db.test_real_set[:len(db.test_real_set) // 2][:N_IMG]
        prefix = cfg.OCCLUSION_LINEMOD
    ds = LineModDatasetRealAug(db_set, prefix, VotingType.Farthest,
                               augment=False, cfg=AUG_CFG)
    for i in range(N_IMG):
        rgb = os.path.basename(ds.imagedb[i]['rgb_pth'])
        assert rgb == EXP5_RGB[split][i], f'image mismatch {split}[{i}]: {rgb}'
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'tle', os.path.join(ROOT, 'tools', 'train_linemod_exp001.py'))
    tle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tle)
    loader = DataLoader(ds, batch_sampler=tle.ImageSizeBatchSampler(
        SequentialSampler(ds), 1, False, AUG_CFG), num_workers=6)
    return ds, loader


def main():
    store = {s: {m: [] for m in ('baseline', 'ls_pred', 'robust_pred')}
             for s in ('linemod', 'occ')}
    kpt_err = {s: {'ls_pred': [], 'robust_pred': [], 'ls_gt': []}
               for s in ('linemod', 'occ')}
    prox = {bi: {'baseline': [], 'ls_pred': [], 'robust_pred': []}
            for bi in range(len(BANDS))}
    prox_images = {bi: set() for bi in range(len(BANDS))}
    n_occ_images_with_occl = 0

    for split in ('linemod', 'occ'):
        ds, loader = build_loader(split)
        log(f'--- {split}: {len(ds)} images, IDs verified against EXP005 ---')
        for image_id, data in enumerate(loader):
            image, mask_gt, vertex_gt, vw, pose, gt_kps = [d.to(device) for d in data]
            with torch.no_grad():
                seg_pred, vertex_pred = net(image)
            b, _, h, w = vertex_pred.shape
            ver = vertex_pred.permute(0, 2, 3, 1).contiguous().view(b, h, w, -1, 2)[0]
            gt2 = (gt_kps[0, :, :2] if gt_kps.dim() == 3 else gt_kps[0, 0, :, :2]).cpu().numpy()

            vis_np = (mask_gt[0].cpu().numpy() == 1)
            ys, xs = np.nonzero(vis_np)
            coords = np.stack([xs, ys], 1).astype(np.float64)
            dirs = ver[ys, xs].cpu().numpy().astype(np.float64)
            m_pred = np.linalg.norm(dirs, axis=2)
            u_pred = dirs / (m_pred[..., None] + EPS)

            v_raw = gt2[None, :, :] - coords[:, None, :]
            d_raw = np.linalg.norm(v_raw, axis=2)
            ok = d_raw >= 1.0
            u_gt = v_raw / (d_raw[..., None] + EPS)
            baseline = angle_deg(u_pred, u_gt)

            # B. existing PVNet RANSAC voting (verbatim mechanism); v3 computes
            #    all keypoints in one call from the same [b,h,w] visible mask
            torch.manual_seed(0)
            with torch.no_grad():
                vk = ransac_voting_layer_v3(
                    mask_gt, vertex_pred.permute(0, 2, 3, 1).contiguous()
                    .view(b, h, w, KP, 2), ROUND_HYP_NUM,
                    inlier_thresh=INLIER_THRESH, max_num=MAX_NUM)
            vk = vk[0].cpu().numpy().astype(np.float64)          # (KP,2)
            # ---- per-keypoint global estimation ----
            khat_ls = np.full((KP, 2), np.nan)
            khat_rob = np.full((KP, 2), np.nan)
            khat_ls_gt = np.full((KP, 2), np.nan)
            for k in range(KP):
                elig = ok[:, k]
                q_idx = np.nonzero(elig)[0]
                if len(q_idx) < 2:
                    continue
                cq = coords[q_idx]
                # A. LS on predicted rays
                khat_ls[k] = ls_intersection(cq, u_pred[q_idx, k])
                khat_rob[k] = vk[k]
                # in-run control: LS on GT rays (EXP008 global oracle machinery)
                khat_ls_gt[k] = ls_intersection(cq, u_gt[q_idx, k])

            # ---- keypoint localization error ----
            kpt_err[split]['ls_pred'].append(np.linalg.norm(khat_ls - gt2, axis=1))
            kpt_err[split]['robust_pred'].append(np.linalg.norm(khat_rob - gt2, axis=1))
            kpt_err[split]['ls_gt'].append(np.linalg.norm(khat_ls_gt - gt2, axis=1))

            # ---- regenerate per-pixel directions from global K_hat ----
            for name, khat in (('ls_pred', khat_ls), ('robust_pred', khat_rob)):
                v_hat = khat[None, :, :] - coords[:, None, :]          # (S,9,2)
                n_hat = np.linalg.norm(v_hat, axis=2)
                u_hat = v_hat / (n_hat[..., None] + EPS)
                store[split][name].append(angle_deg(u_hat, u_gt)[ok])
            store[split]['baseline'].append(baseline[ok])

            # ---- occlusion proximity (OCC only) ----
            if split == 'occ':
                mid = re.findall(r'(\d+)', os.path.basename(ds.imagedb[image_id]['rgb_pth']))[0]
                ap = os.path.join(AMODAL_DIR['occ'], str(int(mid)) + '.png')
                amodal = (np.asarray(Image.open(ap)) > 0) if os.path.exists(ap) else None
                if amodal is not None:
                    occ_region = amodal & ~vis_np
                    if occ_region.any():
                        n_occ_images_with_occl += 1
                        dist_occ = cv2.distanceTransform(
                            (~occ_region).astype(np.uint8), cv2.DIST_L2, 3)
                        d_occ = dist_occ[ys, xs]
                        for bi, (lo, hi) in enumerate(BANDS):
                            sel = ok & (d_occ >= lo)[:, None] & (d_occ < hi)[:, None]
                            if sel.any():
                                prox[bi]['baseline'].append(baseline[sel])
                                prox[bi]['ls_pred'].append(
                                    angle_deg((khat_ls[None] - coords[:, None]) /
                                              (np.linalg.norm(khat_ls[None] - coords[:, None],
                                                              axis=2, keepdims=True) + EPS),
                                              u_gt)[sel])
                                prox[bi]['robust_pred'].append(
                                    angle_deg((khat_rob[None] - coords[:, None]) /
                                              (np.linalg.norm(khat_rob[None] - coords[:, None],
                                                              axis=2, keepdims=True) + EPS),
                                              u_gt)[sel])
                                prox_images[bi].add(image_id)
            if (image_id + 1) % 10 == 0:
                log(f'{split} img {image_id+1}/{len(ds)}  t={time.time()-T0:.0f}s')

    # ---------------- sanity: baseline + GT-LS control ----------------
    base_lin = mstats(np.concatenate(store['linemod']['baseline']))
    base_occ = mstats(np.concatenate(store['occ']['baseline']))
    log(f"control baseline LIN {base_lin['mean']:.3f} (ref {LIN_BASE}) "
        f"OCC {base_occ['mean']:.3f} (ref {OCC_BASE})")
    assert abs(base_lin['mean'] - LIN_BASE) < 0.05 and abs(base_occ['mean'] - OCC_BASE) < 0.05
    gt_ctl = {s: mstats(np.concatenate(kpt_err[s]['ls_gt'])) for s in ('linemod', 'occ')}
    log(f"GT-LS control keypoint err px: LIN {gt_ctl['linemod']['mean']:.4f} "
        f"OCC {gt_ctl['occ']['mean']:.4f} (EXP008 oracle machinery)")

    # ---------------- main results CSV ----------------
    with open(os.path.join(RES_DIR, 'global_estimation_results.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['split', 'method', 'count', 'mean_deg', 'median_deg', 'p90_deg'])
        for s in ('linemod', 'occ'):
            for m in ('baseline', 'ls_pred', 'robust_pred'):
                ms = mstats(np.concatenate(store[s][m]))
                wr.writerow([s, m, ms['count'], f"{ms['mean']:.3f}",
                             f"{ms['median']:.3f}", f"{ms['p90']:.3f}"])

    # ---------------- keypoint localization CSV ----------------
    with open(os.path.join(RES_DIR, 'keypoint_localization.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['split', 'estimator', 'count', 'mean_px_err', 'median_px_err'])
        for s in ('linemod', 'occ'):
            for m in ('ls_pred', 'robust_pred', 'ls_gt'):
                ms = mstats(np.concatenate(kpt_err[s][m]))
                wr.writerow([s, m, ms['count'], f"{ms['mean']:.3f}", f"{ms['median']:.3f}"])

    # ---------------- proximity CSV ----------------
    with open(os.path.join(RES_DIR, 'occ_proximity.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['dist_to_occluded_region_px', 'n_images', 'n_entries',
                     'baseline_mean_deg', 'global_ls_mean_deg', 'global_robust_mean_deg'])
        for bi, (lo, hi) in enumerate(BANDS):
            hi_s = f'{hi:g}' if np.isfinite(hi) else 'inf'
            if not prox[bi]['baseline']:
                wr.writerow([f'{lo:g}-{hi_s}', 0, 0] + [''] * 3); continue
            wr.writerow([f'{lo:g}-{hi_s}', len(prox_images[bi]),
                         len(np.concatenate(prox[bi]['baseline'])),
                         f"{np.concatenate(prox[bi]['baseline']).mean():.3f}",
                         f"{np.concatenate(prox[bi]['ls_pred']).mean():.3f}",
                         f"{np.concatenate(prox[bi]['robust_pred']).mean():.3f}"])

    # ---------------- decision ----------------
    ls_occ = mstats(np.concatenate(store['occ']['ls_pred']))['mean']
    rob_occ = mstats(np.concatenate(store['occ']['robust_pred']))['mean']
    ls_lin = mstats(np.concatenate(store['linemod']['ls_pred']))['mean']
    rob_lin = mstats(np.concatenate(store['linemod']['robust_pred']))['mean']
    best_occ = min(ls_occ, rob_occ)
    cond = best_occ <= GO_OCC_MAX
    decision = 'GO-object-level' if cond else 'STOP-object-level'
    log(f'RESULT  LS: LIN {ls_lin:.3f} OCC {ls_occ:.3f} | Robust: LIN {rob_lin:.3f} '
        f'OCC {rob_occ:.3f} | best OCC {best_occ:.3f} vs threshold {GO_OCC_MAX:.3f}')
    log(f'DECISION: {decision}')

    json.dump(dict(baseline=dict(lin=LIN_BASE, occ=OCC_BASE),
                   global_ls=dict(lin=ls_lin, occ=ls_occ),
                   global_robust=dict(lin=rob_lin, occ=rob_occ),
                   exp008_gt_global_reference=dict(lin=0.001, occ=0.001),
                   keypoint_err_px={s: {m: mstats(np.concatenate(kpt_err[s][m]))
                                        for m in kpt_err[s]} for s in kpt_err},
                   n_occ_images_with_occl=n_occ_images_with_occl,
                   decision=decision, go_occ_max=GO_OCC_MAX,
                   runtime_s=time.time() - T0),
              open(os.path.join(RES_DIR, 'final_summary.json'), 'w'), indent=2)
    log(f'DONE in {time.time()-T0:.1f}s')


if __name__ == '__main__':
    main()
