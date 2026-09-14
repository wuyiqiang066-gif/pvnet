"""EXP010: Oracle Validity-Aware Context Aggregation (offline mechanism diagnosis).

Research question (EXP008 + EXP009 follow-up):
  EXP008: blind local aggregation of GT neighbor directions at r=3 reaches
          0.569 deg mean on OCC (vs 5.121 deg baseline prediction).
  EXP009: a blind LEARNED local aggregation on frozen features DEGRADES OCC
          by +25.7% (6.437 deg).
  Question: if the aggregation KNOWS which neighbors are valid/reliable
  (oracle validity from GT geometry), can it avoid error propagation and
  retain the EXP008 oracle benefit?

Validity (oracle, GT-based, no learned component, pre-fixed threshold):
  e_q = angle(v_q_pred, d_q)                # PVNet 199.pth prediction error at q
  valid(q, k) = 1 if e_q <= 5 deg else 0    # per keypoint k, per neighbor q

Aggregation at target (p, k): radius=3 Euclidean disk, q != p, eligible =
visible fg & |K_k - q| >= 1px (same neighbor set as EXP008).
  blind        : d_hat = normalize(mean of GT directions over ALL eligible neighbors)
                 (identical definition to EXP008 local r=3; cross-checked)
  oracle-valid : d_hat = normalize(mean of GT directions over VALID neighbors only)
  If no valid neighbor: the entry is UNAVAILABLE — excluded from oracle-valid
  statistics, never silently replaced by the baseline prediction.
All aggregated content is GT directions (oracle-style, as EXP008); validity
only changes the neighbor SELECTION — this isolates the effect of
validity awareness, holding content fixed.

Protocol: 20 LIN + 20 OCC cat images (EXP005/006/008/009 IDs asserted),
seed=0, 199.pth, single forward pass per image, NO training, GPU < 2 min.
Pool = EXP006/008: visible fg pixels, |K_k - p| >= 1px, all 9 kps pooled.
Proximity bands (OCC only): dist_occ (EXP005/006 definition) [0,5)/[5,20)/[20,inf).
Mechanism metric: Spearman correlation between baseline error at (p,k) and
the r=3 valid-neighbor ratio at (p,k), OCC pool.

Pre-registered decision (operationalized):
  GO  iff  OCC oracle-valid <= 0.9 x OCC blind   (clear improvement over blind)
       AND OCC oracle-valid <= 0.85 deg          (near EXP008 local r=3 oracle 0.569)
  STOP otherwise.

Run from repo root:
  python experiments/exp010_oracle_validity_context/exp010_oracle_validity_context.py
"""
import os, sys, json, re, csv, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torch.utils.data import DataLoader, SequentialSampler
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from lib.networks.model_repository import Resnet18_8s
from lib.datasets.linemod_dataset import LineModDatasetRealAug, VotingType
from lib.utils.data_utils import LineModImageDB, OcclusionLineModImageDB
from lib.utils.config import cfg

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(EXP_DIR, 'results')
os.makedirs(RES_DIR, exist_ok=True)

N_IMG = 20
KP = 9
RADIUS = 3
VALID_THR = 5.0          # pre-fixed oracle validity threshold (deg)
EPS = 1e-9
LIN_BASE, OCC_BASE = 2.969, 5.121                    # EXP006/008 reference
LIN_BLIND_REF, OCC_BLIND_REF = 0.362, 0.569          # EXP008 local r=3 reference
BANDS = ((0.0, 5.0), (5.0, 20.0), (20.0, np.inf))

torch.manual_seed(0); np.random.seed(0)
T0 = time.time()
LOG = open(os.path.join(RES_DIR, 'run.log'), 'w', buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + '\n')

json.dump(dict(cls='cat', n_img=N_IMG, ckpt='data/model/cat_linemod_train/199.pth',
               radius_px=RADIUS, valid_threshold_deg=VALID_THR,
               validity='oracle: e_q = angle(v_q_pred, d_q) <= 5 deg, per (neighbor q, keypoint k)',
               blind='normalize(mean of GT directions of ALL eligible r=3 neighbors, q != p); '
                     'identical to EXP008 local r=3 (fallback: nearest eligible, expected unused)',
               oracle_valid='normalize(mean of GT directions of VALID neighbors only); '
                            'no valid neighbor -> UNAVAILABLE, no fallback to baseline',
               eligible='visible fg & |K_k - q| >= 1px',
               pixel_pool='EXP006/008 pool: visible fg, |K_k - p| >= 1px, 9 kps pooled',
               proximity='OCC only, dist_occ EXP005/006 definition, bands [0,5)/[5,20)/[20,inf) px',
               go_stop='GO iff OCC oracle-valid <= 0.9 x OCC blind AND OCC oracle-valid <= 0.85 deg; '
                       'else STOP'),
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

DISK = None
def disk_kernel():
    global DISK
    if DISK is None:
        k = 2 * RADIUS + 1
        g = torch.arange(k, device=device, dtype=torch.float32) - RADIUS
        yy, xx = torch.meshgrid(g, g, indexing='ij')
        DISK = ((xx * xx + yy * yy) <= RADIUS * RADIUS).float()
    return DISK

def angle_deg(a, b):
    dot = np.clip((a * b).sum(-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def mstats(a):
    a = np.asarray(a, np.float64); a = a[np.isfinite(a)]
    return dict(count=int(a.size), mean=float(a.mean()) if a.size else float('nan'),
                median=float(np.median(a)) if a.size else float('nan'),
                p90=float(np.quantile(a, 0.9)) if a.size else float('nan'))

# pooled storage per split
store = {s: {'baseline': [], 'blind': [], 'oracle_valid': [], 'valid_ratio': [],
             'unavail': [], 'dist_occ': []} for s in ('linemod', 'occ')}
blind_fallback = {s: 0 for s in ('linemod', 'occ')}
prox = {bi: {'blind': [], 'oracle_valid': [], 'unavail': []} for bi in range(len(BANDS))}
prox_images = {bi: set() for bi in range(len(BANDS))}
n_occ_images_with_occl = 0


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
    kern = disk_kernel()
    kw = kern.view(1, 1, kern.shape[0], kern.shape[1])

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

            vis = (mask_gt[0].cpu().numpy() == 1)
            ys, xs = np.nonzero(vis)
            coords = np.stack([xs, ys], 1).astype(np.float64)
            dirs = ver[ys, xs].cpu().numpy().astype(np.float64)

            v_raw = gt2[None, :, :] - coords[:, None, :]
            d_raw = np.linalg.norm(v_raw, axis=2)
            ok = d_raw >= 1.0
            m_pred = np.linalg.norm(dirs, axis=2)
            u_pred = dirs / (m_pred[..., None] + EPS)
            u_gt = v_raw / (d_raw[..., None] + EPS)

            baseline = angle_deg(u_pred, u_gt)              # (S,9)
            e_q = angle_deg(u_pred, u_gt)                   # validity error at each (q,k)
            valid = e_q <= VALID_THR                        # (S,9) oracle validity

            # grid maps: eligibility, validity, GT-direction content
            elig = torch.zeros(1, KP, h, w, device=device)
            elig[0, :, ys, xs] = torch.from_numpy(ok.T.astype(np.float32)).to(device)
            vmap = torch.zeros(1, KP, h, w, device=device)
            vmap[0, :, ys, xs] = torch.from_numpy((ok & valid).T.astype(np.float32)).to(device)
            umap = torch.zeros(1, KP * 2, h, w, device=device)
            ug_t = torch.from_numpy(u_gt.astype(np.float32)).to(device)
            ok_t = torch.from_numpy(ok.astype(np.float32)).to(device)
            va_t = torch.from_numpy((ok & valid).astype(np.float32)).to(device)
            umap[0, 0::2, ys, xs] = (ug_t[..., 0] * ok_t).T
            umap[0, 1::2, ys, xs] = (ug_t[..., 1] * ok_t).T
            vumap = torch.zeros(1, KP * 2, h, w, device=device)
            vumap[0, 0::2, ys, xs] = (ug_t[..., 0] * va_t).T
            vumap[0, 1::2, ys, xs] = (ug_t[..., 1] * va_t).T

            cnt_all = F.conv2d(elig, kw.expand(KP, 1, kw.shape[2], kw.shape[3]),
                               padding=RADIUS, groups=KP)[0]
            cnt_val = F.conv2d(vmap, kw.expand(KP, 1, kw.shape[2], kw.shape[3]),
                               padding=RADIUS, groups=KP)[0]
            sum_all = F.conv2d(umap, kw.expand(KP * 2, 1, kw.shape[2], kw.shape[3]),
                               padding=RADIUS, groups=KP * 2)[0].view(KP, 2, h, w)
            sum_val = F.conv2d(vumap, kw.expand(KP * 2, 1, kw.shape[2], kw.shape[3]),
                               padding=RADIUS, groups=KP * 2)[0].view(KP, 2, h, w)

            cnt_all_at = cnt_all[:, ys, xs].cpu().numpy().astype(np.int64)
            cnt_val_at = cnt_val[:, ys, xs].cpu().numpy().astype(np.int64)
            sum_all_at = sum_all[:, :, ys, xs].permute(2, 0, 1).cpu().numpy().astype(np.float64)
            sum_val_at = sum_val[:, :, ys, xs].permute(2, 0, 1).cpu().numpy().astype(np.float64)

            nb_all = (cnt_all_at - 1).T                                    # (S,9) excl. self
            vself = (ok & valid).astype(np.float64)                        # own flag to subtract
            nb_val = (cnt_val_at.T - vself)                                # (S,9) excl. self
            ssum_blind = sum_all_at - u_gt                                 # (S,9,2), remove self
            ssum_val = sum_val_at - vself[..., None] * u_gt                # (S,9,2), remove self

            # ---- blind aggregation (EXP008 local r=3 definition) ----
            err_blind = np.full(ok.shape, np.nan)
            norm = np.linalg.norm(ssum_blind, axis=2, keepdims=True)
            d_hat = ssum_blind / (norm + EPS)
            err_blind = np.where(nb_all > 0, angle_deg(d_hat, u_gt), np.nan)
            fks, fis = np.nonzero(nb_all <= 0)                             # fallback path
            if len(fks):
                tree = cKDTree(coords)
                for k, i in zip(fks, fis):
                    dists, idxs = tree.query(coords[i], k=min(64, len(coords)))
                    pick = None
                    for j in np.atleast_1d(idxs):
                        if j != i and d_raw[j, k] >= 1.0:
                            pick = j; break
                    if pick is None:
                        continue
                    u = gt2[k] - coords[pick]
                    u = u / max(np.linalg.norm(u), EPS)
                    err_blind[i, k] = angle_deg(u[None, :], u_gt[i:i + 1, k])[0]
                blind_fallback[split] += len(fks)

            # ---- oracle-valid aggregation ----
            err_ovalid = np.full(ok.shape, np.nan)
            has = nb_val > 0
            norm_v = np.linalg.norm(ssum_val, axis=2, keepdims=True)
            d_hat_v = ssum_val / (norm_v + EPS)
            err_ovalid = np.where(has, angle_deg(d_hat_v, u_gt), np.nan)

            ratio = np.where(nb_all > 0, nb_val / np.maximum(nb_all, 1), np.nan)

            store[split]['baseline'].append(baseline[ok])
            store[split]['blind'].append(err_blind[ok])
            store[split]['oracle_valid'].append(err_ovalid[ok])
            store[split]['valid_ratio'].append(ratio[ok])
            store[split]['unavail'].append((~has & ok).astype(np.float64)[ok])

            # occlusion proximity (OCC only, images with actual occlusion)
            if split == 'occ':
                mid = re.findall(r'(\d+)', os.path.basename(ds.imagedb[image_id]['rgb_pth']))[0]
                ap = os.path.join(AMODAL_DIR['occ'], str(int(mid)) + '.png')
                amodal = (np.asarray(Image.open(ap)) > 0) if os.path.exists(ap) else None
                if amodal is not None:
                    occ_region = amodal & ~vis
                    if occ_region.any():
                        global n_occ_images_with_occl
                        n_occ_images_with_occl += 1
                        dist_occ = cv2.distanceTransform(
                            (~occ_region).astype(np.uint8), cv2.DIST_L2, 3)
                        d_occ = dist_occ[ys, xs]
                        for bi, (lo, hi) in enumerate(BANDS):
                            sel = ok & (d_occ >= lo)[:, None] & (d_occ < hi)[:, None]
                            if sel.any():
                                prox[bi]['blind'].append(err_blind[sel])
                                prox[bi]['oracle_valid'].append(err_ovalid[sel])
                                prox[bi]['unavail'].append((~has & sel).astype(np.float64)[sel])
                                prox_images[bi].add(image_id)
            if (image_id + 1) % 20 == 0:
                log(f'{split} img {image_id+1}/{len(ds)}  t={time.time()-T0:.0f}s')

    # ---------------- sanity checks ----------------
    log('--- cross-checks ---')
    for s in ('linemod', 'occ'):
        bm = mstats(np.concatenate(store[s]['baseline']))
        ref = LIN_BASE if s == 'linemod' else OCC_BASE
        log(f"{s} baseline {bm['mean']:.3f} vs ref {ref:.3f} (d={abs(bm['mean']-ref):.4f}) "
            f"count {bm['count']}")
        assert abs(bm['mean'] - ref) < 0.05
    for s in ('linemod', 'occ'):
        bl = mstats(np.concatenate(store[s]['blind']))
        ref = LIN_BLIND_REF if s == 'linemod' else OCC_BLIND_REF
        log(f"{s} blind r3 {bl['mean']:.3f} vs EXP008 local_r3 {ref:.3f} "
            f"(d={abs(bl['mean']-ref):.4f}) fallbacks {blind_fallback[s]}")
        assert abs(bl['mean'] - ref) < 0.02, f'blind mismatch vs EXP008 on {s}'
        assert blind_fallback[s] == 0

    # ---------------- context_aggregation_results.csv ----------------
    with open(os.path.join(RES_DIR, 'context_aggregation_results.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['split', 'method', 'count', 'mean_deg', 'median_deg', 'p90_deg'])
        for s in ('linemod', 'occ'):
            for m in ('baseline', 'blind', 'oracle_valid'):
                ms = mstats(np.concatenate(store[s][m]))
                wr.writerow([s, m, ms['count'], f"{ms['mean']:.3f}",
                             f"{ms['median']:.3f}", f"{ms['p90']:.3f}"])

    # ---------------- validity_statistics.csv ----------------
    vstat = {}
    for s in ('linemod', 'occ'):
        r = np.asarray(np.concatenate(store[s]['valid_ratio']), np.float64)
        r = r[np.isfinite(r)]
        u = np.asarray(np.concatenate(store[s]['unavail']), np.float64)
        vstat[s] = dict(mean_valid_ratio=float(r.mean()), median_valid_ratio=float(np.median(r)),
                        unavailable_pct=float(u.mean() * 100.0))
        log(f"{s} valid ratio mean {r.mean():.4f} median {np.median(r):.4f} "
            f"unavailable {u.mean()*100:.2f}%")
    with open(os.path.join(RES_DIR, 'validity_statistics.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['split', 'mean_valid_ratio', 'median_valid_ratio', 'unavailable_pct'])
        for s in ('linemod', 'occ'):
            wr.writerow([s, f"{vstat[s]['mean_valid_ratio']:.4f}",
                         f"{vstat[s]['median_valid_ratio']:.4f}",
                         f"{vstat[s]['unavailable_pct']:.3f}"])

    # ---------------- occ_proximity.csv ----------------
    with open(os.path.join(RES_DIR, 'occ_proximity.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['dist_to_occluded_region_px', 'n_images', 'n_entries',
                     'blind_mean_deg', 'oracle_valid_mean_deg', 'oracle_valid_n_available',
                     'oracle_valid_unavailable_pct'])
        for bi, (lo, hi) in enumerate(BANDS):
            hi_s = f'{hi:g}' if np.isfinite(hi) else 'inf'
            if not prox[bi]['blind']:
                wr.writerow([f'{lo:g}-{hi_s}', 0, 0] + [''] * 4); continue
            bl = np.concatenate(prox[bi]['blind'])
            ov = np.concatenate(prox[bi]['oracle_valid'])
            un = np.concatenate(prox[bi]['unavail'])
            ov_fin = ov[np.isfinite(ov)]
            wr.writerow([f'{lo:g}-{hi_s}', len(prox_images[bi]), bl.size,
                         f'{bl.mean():.3f}', f'{ov_fin.mean():.3f}' if ov_fin.size else 'nan',
                         int(ov_fin.size), f'{un.mean()*100:.2f}'])

    # ---------------- mechanism metric: Spearman(OCC baseline error, valid ratio) ----------------
    bl_pool = np.concatenate(store['occ']['baseline'])
    ra_pool = np.asarray(np.concatenate(store['occ']['valid_ratio']), np.float64)
    fin = np.isfinite(ra_pool)
    rho, pval = spearmanr(bl_pool[fin], ra_pool[fin])
    log(f'OCC Spearman(baseline error, valid ratio) = {rho:.4f} (p={pval:.3e}, n={fin.sum()})')

    # ---------------- decision ----------------
    blind_occ = mstats(np.concatenate(store['occ']['blind']))['mean']
    ov_occ = mstats(np.concatenate(store['occ']['oracle_valid']))['mean']
    ov_lin = mstats(np.concatenate(store['linemod']['oracle_valid']))['mean']
    blind_lin = mstats(np.concatenate(store['linemod']['blind']))['mean']
    cond1 = ov_occ <= 0.9 * blind_occ
    cond2 = ov_occ <= 0.85
    decision = 'GO' if (cond1 and cond2) else 'STOP'
    log(f'RESULT blind OCC {blind_occ:.3f} | oracle-valid OCC {ov_occ:.3f} | '
        f'cond1(<=0.9xblind={0.9*blind_occ:.3f}): {cond1} | cond2(<=0.85): {cond2}')
    log(f'DECISION: {decision}')

    json.dump(dict(baseline=dict(lin=LIN_BASE, occ=OCC_BASE),
                   blind_r3=dict(lin=blind_lin, occ=blind_occ),
                   oracle_valid_r3=dict(lin=ov_lin, occ=ov_occ),
                   validity_stats=vstat,
                   spearman_occ=dict(rho=float(rho), p=float(pval), n=int(fin.sum())),
                   n_occ_images_with_occl=n_occ_images_with_occl,
                   decision=decision, cond1_clear_improvement=bool(cond1),
                   cond2_near_oracle=bool(cond2), runtime_s=time.time() - T0),
              open(os.path.join(RES_DIR, 'final_summary.json'), 'w'), indent=2)
    log(f'DONE in {time.time()-T0:.1f}s')


if __name__ == '__main__':
    main()
