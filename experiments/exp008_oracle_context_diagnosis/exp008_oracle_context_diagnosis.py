"""EXP008: Oracle context diagnosis — can visible-object context recover the
correct vertex direction at occluded/degraded pixels? (offline, inference-only)

Research question (cat-only PVNet baseline, 199.pth, NO training):
  For a foreground pixel p, does the GT geometric information of OTHER VISIBLE
  foreground pixels contain enough information to recover p's GT direction
  normalize(K - p)?

Estimators (all oracle-style: no training, no learned model, no PVNet output
used as context, no test-time optimization):
  baseline : angle between PVNet 199.pth predicted unit direction and GT
             direction. Pixel pool identical to EXP006 ('all': visible
             foreground pixels with |K - p| >= 1px, all 9 keypoints pooled).
  local r  : mean of GT unit directions of eligible visible pixels within
             Euclidean radius r in {3,7,15,30} (p itself excluded). If no
             eligible neighbor exists, fall back to the nearest eligible
             visible pixel's GT direction (deterministic; fallback rate
             reported). Simple estimator by design (spec): the goal is the
             recoverability-vs-context-scale trend, not the best estimator.
  global   : object-level readout. The keypoint location K_hat is estimated
             from the GT direction rays of all OTHER eligible visible pixels
             (least-squares ray intersection; p's own ray excluded), then
             d_hat = normalize(K_hat - p). This is the "whole visible object
             geometry" upper bound. The trivial oracle normalize(K_exact - p)
             is NOT used (error 0 by construction, non-diagnostic): under
             occlusion K itself must be inferred from the visible context,
             which is exactly what this oracle measures.

GT directions of neighbors are the idealized stand-in for a perfect readout
of visible object geometry (see REPORT.md Limitations).

Protocol: 20 LINEMOD + 20 OCC cat images identical to EXP005/EXP006 (asserted
against exp005 image_ids.json), seed=0, GPU inference, offline oracle
computation. Proximity bands (OCC only): distance to occluded region
(dist_occ, EXP005/006 definition) in [0,5) / [5,20) / [20,inf) px; only
images with a non-empty occluded region contribute.

Run from repo root:
  python experiments/exp008_oracle_context_diagnosis/exp008_oracle_context_diagnosis.py
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

from lib.networks.model_repository import Resnet18_8s
from lib.datasets.linemod_dataset import LineModDatasetRealAug, VotingType
from lib.utils.data_utils import LineModImageDB, OcclusionLineModImageDB
from lib.utils.config import cfg

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(EXP_DIR, 'results')
os.makedirs(RES_DIR, exist_ok=True)

N_IMG = 20
KP = 9
RADII = (3, 7, 15, 30)
EPS = 1e-9
RIDGE = 1e-9
BANDS = ((0.0, 5.0), (5.0, 20.0), (20.0, np.inf))

torch.manual_seed(0); np.random.seed(0)
T0 = time.time()
LOG = open(os.path.join(RES_DIR, 'run.log'), 'w', buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + '\n')

CFG = dict(
    cls='cat', n_img_per_split=N_IMG, ckpt='data/model/cat_linemod_train/199.pth',
    images='identical to EXP005/EXP006 (verified against exp005 image_ids.json)',
    radii_px=list(RADII),
    local_estimator='mean of neighbor GT unit directions within Euclidean radius r '
                    '(p excluded); fallback = nearest eligible visible pixel GT direction',
    global_estimator='least-squares ray intersection of GT direction rays of all other '
                     'eligible visible pixels -> K_hat; d_hat = normalize(K_hat - p); '
                     "p's own ray excluded; trivial exact-K oracle intentionally NOT used",
    eligibility='visible foreground pixel (GT visible mask) with |K_k - q| >= 1px',
    pixel_pool='same as EXP006 all-region pool: visible fg pixels, |K_k - p| >= 1px, '
               'all 9 keypoints pooled',
    proximity='OCC only, dist_occ = distanceTransform(~(amodal & ~vis)) as EXP005/006; '
              'bands [0,5) / [5,20) / [20,inf) px; only images with non-empty occluded region',
    metrics='direction error in degrees; mean/median/p90; recoverable gap = baseline - oracle',
)
json.dump(CFG, open(os.path.join(RES_DIR, 'config.json'), 'w'), indent=2)

# ---------------- model ----------------
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

DISK = {r: None for r in RADII}   # lazily built on GPU

def disk_kernel(r):
    if DISK[r] is None:
        k = 2 * r + 1
        g = torch.arange(k, device=device, dtype=torch.float32) - r
        yy, xx = torch.meshgrid(g, g, indexing='ij')
        DISK[r] = ((xx * xx + yy * yy) <= r * r).float()
    return DISK[r]

def angle_deg(a, b):
    """elementwise angle (deg) between two unit-vector arrays [...,2]"""
    dot = np.clip((a * b).sum(-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def mstats(a):
    a = np.asarray(a, np.float64); a = a[np.isfinite(a)]
    return dict(count=int(a.size), mean=float(a.mean()) if a.size else float('nan'),
                median=float(np.median(a)) if a.size else float('nan'),
                p90=float(np.quantile(a, 0.9)) if a.size else float('nan'))

# storage: per split, per method -> list of per-image flat arrays
METHODS = ['baseline'] + [f'local_r{r}' for r in RADII] + ['global']
store = {s: {m: [] for m in METHODS} for s in ('linemod', 'occ')}
fb_count = {s: {f'local_r{r}': [0, 0] for r in RADII} for s in ('linemod', 'occ')}  # [fallbacks, pool_entries]
prox = {b: {'baseline': [], 'global': [], 'gap': []} for b in range(len(BANDS))}
prox_images = {b: set() for b in range(len(BANDS))}
n_occ_images_with_occl = 0


def local_oracle(err, u_gt, ok, coords, gt2, d_raw, tree, H, W, ys, xs, split):
    """fill err[:,k] for each radius via disk-mean of neighbor GT directions."""
    elig = torch.zeros(1, KP, H, W, device=device)
    umap = torch.zeros(1, KP * 2, H, W, device=device)
    elig[0, :, ys, xs] = torch.from_numpy(ok.T.astype(np.float32)).to(device)
    ug_t = torch.from_numpy(u_gt.astype(np.float32)).to(device)      # (S,9,2)
    ok_t = torch.from_numpy(ok.astype(np.float32)).to(device)
    umap[0, 0::2, ys, xs] = (ug_t[..., 0] * ok_t).T
    umap[0, 1::2, ys, xs] = (ug_t[..., 1] * ok_t).T

    for r in RADII:
        kern = disk_kernel(r)
        kw = kern.view(1, 1, kern.shape[0], kern.shape[1])
        cnt = F.conv2d(elig, kw.expand(KP, 1, kw.shape[2], kw.shape[3]),
                       padding=r, groups=KP)[0]                     # (9,H,W)
        ssum = F.conv2d(umap, kw.expand(KP * 2, 1, kw.shape[2], kw.shape[3]),
                        padding=r, groups=KP * 2)[0].view(KP, 2, H, W)
        cnt_at = cnt[:, ys, xs].cpu().numpy().astype(np.int64)       # (9,S) includes self
        nb_cnt = (cnt_at - 1).T                                      # (S,9)
        nb_sum = ssum[:, :, ys, xs].permute(2, 0, 1).cpu().numpy().astype(np.float64) - u_gt
        norm = np.linalg.norm(nb_sum, axis=2, keepdims=True)
        d_hat = nb_sum / (norm + EPS)
        e = angle_deg(d_hat, u_gt)                                   # (S,9)
        e = np.where(nb_cnt > 0, e, np.nan)

        # fallback: nearest eligible visible pixel (deterministic KDTree)
        fis, fks = np.nonzero(nb_cnt <= 0)                           # (S,9) -> pixel, kp
        for i, k in zip(fis, fks):
            dists, idxs = tree.query(coords[i], k=min(64, len(coords)))
            pick = None
            for j in np.atleast_1d(idxs):
                if j != i and d_raw[j, k] >= 1.0:
                    pick = j; break
            if pick is None:
                continue   # stays NaN, excluded from stats
            u = gt2[k] - coords[pick]
            u = u / max(np.linalg.norm(u), EPS)
            e[i, k] = angle_deg(u[None, :], u_gt[i:i + 1, k])[0]
        fb_count[split][f'local_r{r}'][0] += len(fks)
        fb_count[split][f'local_r{r}'][1] += int(ok.sum())
        err[f'local_r{r}'] = e


def global_oracle(err, u_gt, ok, coords, gt2):
    """fill err['global'] via LS ray intersection of all other eligible visible rays."""
    e = np.full(ok.shape, np.nan)
    for k in range(KP):
        q = np.nonzero(ok[:, k])[0]
        if len(q) < 2:
            continue
        cq = coords[q]                                               # (Q,2)
        uq = u_gt[q, k]                                              # (Q,2)
        M = np.eye(2)[None, :, :] - uq[:, :, None] * uq[:, None, :]  # (Q,2,2)
        Mcq = M @ cq[:, :, None]                                     # (Q,2,1)
        A_full = M.sum(0)
        b_full = Mcq.sum(0).ravel()                                  # (2,)
        A_p = A_full[None] - M                                       # exclude own ray
        b_p = b_full[None] - Mcq[:, :, 0]
        try:
            Khat = np.linalg.solve(A_p + RIDGE * np.eye(2),
                                   b_p[..., None])[..., 0]           # (Q,2)
        except np.linalg.LinAlgError:
            Khat = np.stack([np.linalg.lstsq(A_p[i] + RIDGE * np.eye(2), b_p[i], rcond=None)[0]
                             for i in range(len(q))])
        dh = Khat - cq
        dh = dh / (np.linalg.norm(dh, axis=1, keepdims=True) + EPS)
        e[q, k] = angle_deg(dh, uq)
    err['global'] = e


def main():
    from torch.utils.data import DataLoader as DL
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'tle', os.path.join(ROOT, 'tools', 'train_linemod_exp001.py'))
    tle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tle)
    ImageSizeBatchSampler = tle.ImageSizeBatchSampler

    for split in ('linemod', 'occ'):
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
        log(f'--- {split}: {len(ds)} images, IDs verified against EXP005 ---')
        loader = DL(ds, batch_sampler=ImageSizeBatchSampler(
            SequentialSampler(ds), 1, False, AUG_CFG), num_workers=6)

        for image_id, data in enumerate(loader):
            image, mask_gt, vertex_gt, vw, pose, gt_kps = [d.to(device) for d in data]
            with torch.no_grad():
                seg_pred, vertex_pred = net(image)
            b, _, h, w = vertex_pred.shape
            ver = vertex_pred.permute(0, 2, 3, 1).contiguous().view(b, h, w, -1, 2)[0]
            gt2 = (gt_kps[0, :, :2] if gt_kps.dim() == 3 else gt_kps[0, 0, :, :2]).cpu().numpy()

            vis = (mask_gt[0].cpu().numpy() == 1)
            H, W = vis.shape
            ys, xs = np.nonzero(vis)
            coords = np.stack([xs, ys], 1).astype(np.float64)        # (S,2)
            dirs = ver[ys, xs].cpu().numpy().astype(np.float64)      # (S,9,2)

            v_raw = gt2[None, :, :] - coords[:, None, :]             # (S,9,2)
            d_raw = np.linalg.norm(v_raw, axis=2)
            ok = d_raw >= 1.0
            m_pred = np.linalg.norm(dirs, axis=2)
            u_pred = dirs / (m_pred[..., None] + EPS)
            u_gt = v_raw / (d_raw[..., None] + EPS)

            err = {'baseline': angle_deg(u_pred, u_gt)}
            tree = cKDTree(coords)
            local_oracle(err, u_gt, ok, coords, gt2, d_raw, tree, H, W, ys, xs, split)
            global_oracle(err, u_gt, ok, coords, gt2)
            for m in METHODS:
                store[split][m].append(err[m][ok])                   # pool = ok entries

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
                                prox[bi]['baseline'].append(err['baseline'][sel])
                                prox[bi]['global'].append(err['global'][sel])
                                prox[bi]['gap'].append((err['baseline'] - err['global'])[sel])
                                prox_images[bi].add(image_id)
            if (image_id + 1) % 10 == 0:
                log(f'{split} img {image_id+1}/{len(ds)}  t={time.time()-T0:.0f}s')

    # ---------------- sanity check vs EXP006 ----------------
    log('--- baseline cross-check vs EXP006 direction_error.csv (all region) ---')
    exp6_csv = os.path.join(ROOT, 'experiments', 'exp006_vertex_representation_diagnosis',
                            'results', 'direction_error.csv')
    ref = {}
    with open(exp6_csv) as f:
        for row in csv.DictReader(f):
            if row['region'] == 'all':
                ref[row['split']] = (float(row['mean_deg']), float(row['median_deg']),
                                     float(row['p90_deg']), int(row['count']))
    for s in ('linemod', 'occ'):
        a = np.concatenate(store[s]['baseline'])
        ms = mstats(a)
        d_mean = abs(ms['mean'] - ref[s][0])
        log(f"{s}: baseline mean {ms['mean']:.3f} vs EXP006 {ref[s][0]:.3f} "
            f"(d={d_mean:.4f}) count {ms['count']} vs {ref[s][3]}")
        assert d_mean < 0.05, f'baseline mismatch vs EXP006 on {s}'

    # ---------------- oracle_context_results.csv ----------------
    with open(os.path.join(RES_DIR, 'oracle_context_results.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['split', 'method', 'count', 'mean_deg', 'median_deg', 'p90_deg',
                     'fallback_frac'])
        for s in ('linemod', 'occ'):
            for m in METHODS:
                ms = mstats(np.concatenate(store[s][m]))
                fb = ''
                if m.startswith('local'):
                    nfb, tot = fb_count[s][m]
                    fb = f'{nfb / max(tot, 1):.5f}'
                wr.writerow([s, m, ms['count'], f"{ms['mean']:.3f}", f"{ms['median']:.3f}",
                             f"{ms['p90']:.3f}", fb])

    # ---------------- recoverable_gap.csv ----------------
    with open(os.path.join(RES_DIR, 'recoverable_gap.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['split', 'method', 'baseline_mean', 'oracle_mean', 'gap_mean',
                     'baseline_median', 'oracle_median', 'gap_median'])
        for s in ('linemod', 'occ'):
            bl = np.concatenate(store[s]['baseline'])
            for m in METHODS[1:]:
                oc = np.concatenate(store[s][m])
                wr.writerow([s, m, f'{bl.mean():.3f}', f'{oc.mean():.3f}',
                             f'{bl.mean() - oc.mean():.3f}',
                             f'{np.median(bl):.3f}', f'{np.median(oc):.3f}',
                             f'{np.median(bl) - np.median(oc):.3f}'])

    # ---------------- occlusion_proximity.csv ----------------
    with open(os.path.join(RES_DIR, 'occlusion_proximity.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['dist_to_occluded_region_px', 'n_images', 'n_entries',
                     'baseline_mean', 'baseline_median', 'global_mean', 'global_median',
                     'gap_mean', 'gap_median'])
        for bi, (lo, hi) in enumerate(BANDS):
            hi_s = f'{hi:g}' if np.isfinite(hi) else 'inf'
            if not prox[bi]['baseline']:
                wr.writerow([f'{lo:g}-{hi_s}', 0, 0] + [''] * 6); continue
            bl = np.concatenate(prox[bi]['baseline'])
            gl = np.concatenate(prox[bi]['global'])
            gp = np.concatenate(prox[bi]['gap'])
            wr.writerow([f'{lo:g}-{hi_s}', len(prox_images[bi]), bl.size,
                         f'{bl.mean():.3f}', f'{np.median(bl):.3f}',
                         f'{gl.mean():.3f}', f'{np.median(gl):.3f}',
                         f'{gp.mean():.3f}', f'{np.median(gp):.3f}'])

    log(f'OCC images with occluded region: {n_occ_images_with_occl}/20 '
        f'(proximity bands limited to these)')
    log(f'DONE in {time.time()-T0:.1f}s')


if __name__ == '__main__':
    main()
