"""EXP006: Vertex Representation Failure Analysis under occlusion (inference-only).

Question: under OCC, is the PVNet vertex representation degraded in DIRECTION,
in MAGNITUDE, or both? Is degradation uniform across keypoints or concentrated?

Strictly reuses EXP005's protocol: same 20+20 deterministic images (verified
against EXP005 image_ids.json), 199.pth baseline inference, same region labels
(interior / normal_boundary / occlusion_boundary, GT amodal masks), same
invalid handling (norms clamped at 1e-9; pixels <1px from GT kp excluded).

IMPORTANT architectural fact (verified in lib/datasets/linemod_dataset.py,
compute_vertex_hcoords): PVNet supervises UNIT direction vectors. Hence
pred_mag ~= 1.0 by construction and the supervision-space GT magnitude is 1.
All metrics are reported in BOTH spaces:
  raw space : v_gt = gt_kp - pixel (spec formula); magnitude/endpoint errors
              are dominated by the geometric displacement magnitude.
  unit space: v_gt = unit(gt_kp - pixel) = the supervision target; errors
              isolate the learned representation. Mechanism conclusions are
              drawn here (clearly labelled).

Direction-only / magnitude-only hypotheticals (offline diagnostics, NEVER fed
into RANSAC/PnP) are likewise computed in both spaces.

Run from repo root: python experiments/exp006_vertex_representation_diagnosis/exp006_diag.py
"""
import os, sys, json, re, csv, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, SequentialSampler

from lib.networks.model_repository import Resnet18_8s
from lib.datasets.linemod_dataset import LineModDatasetRealAug, VotingType
from lib.utils.data_utils import LineModImageDB, OcclusionLineModImageDB
from lib.utils.config import cfg

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(EXP_DIR, 'results')
os.makedirs(RES_DIR, exist_ok=True)

N_IMG = 20
KP = 9
DIR_BAD_DEG = 10.0        # spec: bad direction if angle > 10 deg
ENDP_BAD_PX = 10.0        # spec: bad endpoint if > 10 px
PROX_BINS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, np.inf)]
EPS = 1e-9

torch.manual_seed(0); np.random.seed(0)
T0 = time.time()
LOG = open(os.path.join(RES_DIR, 'run.log'), 'w', buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + '\n')

json.dump(dict(cls='cat', n_img_per_split=N_IMG, ckpt='data/model/cat_linemod_train/199.pth',
               images='identical to EXP005 (verified against exp005 image_ids.json)',
               dir_bad_deg=DIR_BAD_DEG, endp_bad_px=ENDP_BAD_PX, prox_bins=str(PROX_BINS),
               supervision='PVNet supervises UNIT direction vectors (compute_vertex_hcoords); '
                           'pred_mag~=1 by construction. Metrics reported in raw space '
                           '(v_gt = gt_kp - pixel, spec formula) and unit/supervision space '
                           '(v_gt = unit vector); mechanism conclusions use unit space.',
               invalid_handling='norms clamped at 1e-9 (no NaN); pixels with |gt_kp - pixel| < 1px '
                                'excluded from all metrics (direction undefined)',
               diagnostics='direction-only / magnitude-only hypothetical endpoint errors, '
                           'offline only, never fed into RANSAC/PnP'),
          open(os.path.join(RES_DIR, 'config.json'), 'w'), indent=2)

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

# EXP005 image ids (must match exactly)
EXP5_IDS = json.load(open(os.path.join(
    ROOT, 'experiments', 'exp005_visibility_diagnosis', 'results', 'image_ids.json')))
EXP5_RGB = {s: [r['rgb'] for r in EXP5_IDS['image_ids'][s]] for s in ('linemod', 'occ')}


def region_labels(vis_mask, amodal_mask):
    """Same definitions as diagnosis_vote_analysis / EXP005."""
    m = (vis_mask > 0)
    interior = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8),
                         iterations=5).astype(bool)
    out = {'interior': interior, 'boundary': m & ~interior}
    if amodal_mask is not None:
        occ_region = amodal_mask & ~m
        out['occluded_unvotable'] = occ_region
        dist_occ = cv2.distanceTransform((~occ_region).astype(np.uint8), cv2.DIST_L2, 3)
        occl_b = m & (dist_occ <= 5)
        out['occl_boundary'] = occl_b
        out['interior'] = out['interior'] & ~occl_b
        out['boundary'] = out['boundary'] & ~occl_b
        out['dist_occ'] = dist_occ
    else:
        out['occluded_unvotable'] = None
        out['occl_boundary'] = np.zeros_like(interior)
        out['dist_occ'] = None
    return out


def line_err(dirs_unit, coords, kp):
    v = kp[None, :] - coords
    t = (v * dirs_unit).sum(1, keepdims=True)
    return np.linalg.norm(v - t * dirs_unit, axis=1)


def mstats(a):
    a = np.asarray(a, np.float64)
    return dict(count=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.quantile(a, 0.9)))


def bad_rate(a, thr):
    a = np.asarray(a, np.float64)
    return float((a > thr).mean()) if a.size else None


REGIONS = ('all', 'interior', 'normal_boundary', 'occlusion_boundary')
RMAP = {'interior': 'interior', 'normal_boundary': 'boundary',
        'occlusion_boundary': 'occl_boundary'}

store = {s: {m: {r: [] for r in REGIONS} for m in
             ('dir', 'mag_raw', 'mag_norm_raw', 'mag_dev_unit', 'endp_raw', 'endp_unit',
              'dironly_raw', 'magonly_raw', 'dironly_unit', 'magonly_unit', 'e_vote',
              'pred_mag', 'dist_kp')}
         for s in ('linemod', 'occ')}
kp_store = {s: {m: {k: [] for k in range(KP)} for m in
                ('dir', 'mag_dev_unit', 'mag_raw', 'endp_unit', 'endp_raw')}
            for s in ('linemod', 'occ')}
prox_store = {b: {m: [] for m in ('e_vote', 'dir', 'mag_raw', 'endp_unit')}
              for b in range(len(PROX_BINS))}
prox_img_count = {b: set() for b in range(len(PROX_BINS))}
occ_images_with_occl = 0


def main():
    from torch.utils.data import DataLoader as DL
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'tle', os.path.join(ROOT, 'tools', 'train_linemod_exp001.py'))
    tle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tle)
    ImageSizeBatchSampler = tle.ImageSizeBatchSampler

    global occ_images_with_occl

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
        # verify image identity against EXP005
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
                gt2 = gt_kps[0, :, :2] if gt_kps.dim() == 3 else gt_kps[0, 0, :, :2]

                vis = (mask_gt[0].cpu().numpy() == 1)
                rgb_pth = ds.imagedb[image_id]['rgb_pth']
                mid = re.findall(r'(\d+)', os.path.basename(rgb_pth))[0]
                ap = os.path.join(AMODAL_DIR[split], str(int(mid)) + '.png')
                amodal = (np.asarray(Image.open(ap)) > 0) if os.path.exists(ap) else None
                regions = region_labels(vis, amodal)

                ys, xs = np.nonzero(vis)
                coords = np.stack([xs, ys], 1).astype(np.float64)
                dirs = ver[ys, xs].cpu().numpy().astype(np.float64)      # [S,9,2] raw pred
                gt_np = gt2.cpu().numpy()

                v_raw = gt_np[None, :, :] - coords[:, None, :]           # [S,9,2]
                d_raw = np.linalg.norm(v_raw, axis=2)                    # gt_mag (raw)
                ok = d_raw >= 1.0                                        # direction undefined otherwise
                m_pred = np.linalg.norm(dirs, axis=2)                    # pred_mag
                u_pred = dirs / (m_pred[..., None] + EPS)
                u_gt = v_raw / (d_raw[..., None] + EPS)
                dot = np.clip((u_pred * u_gt).sum(-1), -1.0, 1.0)
                ang = np.degrees(np.arccos(dot))

                # metrics
                e_vote = np.stack([line_err(u_pred[:, k], coords, gt_np[k])
                                   for k in range(KP)], 1)
                mag_raw = np.abs(m_pred - d_raw)
                mag_norm_raw = mag_raw / (d_raw + EPS)
                mag_dev_unit = np.abs(m_pred - 1.0)                      # unit-space magnitude error
                endp_raw = np.linalg.norm(dirs - v_raw, axis=2)
                endp_unit = np.linalg.norm(dirs - u_gt, axis=2)
                dironly_raw = np.linalg.norm(u_pred * d_raw[..., None] - v_raw, axis=2)
                magonly_raw = np.linalg.norm(u_gt * m_pred[..., None] - v_raw, axis=2)
                dironly_unit = np.linalg.norm(u_pred - u_gt, axis=2)     # GT mag=1 in unit space
                magonly_unit = np.abs(m_pred - 1.0)

                st = store[split]
                selmap = {r: regions[RMAP[r]][ys, xs][:, None] & ok
                          for r in REGIONS if r in RMAP}
                selmap['all'] = ok  # all fg pixels with valid direction
                arrs = dict(dir=ang, mag_raw=mag_raw, mag_norm_raw=mag_norm_raw,
                            mag_dev_unit=mag_dev_unit, endp_raw=endp_raw, endp_unit=endp_unit,
                            dironly_raw=dironly_raw, magonly_raw=magonly_raw,
                            dironly_unit=dironly_unit, magonly_unit=magonly_unit,
                            e_vote=e_vote, pred_mag=m_pred, dist_kp=d_raw)
                for r, sel in selmap.items():
                    for m, a in arrs.items():
                        if sel.any():
                            st[m][r].append(a[sel].ravel())
                for k in range(KP):
                    kst = kp_store[split]
                    for m, a in (('dir', ang), ('mag_dev_unit', mag_dev_unit),
                                 ('mag_raw', mag_raw), ('endp_unit', endp_unit),
                                 ('endp_raw', endp_raw)):
                        kst[m][k].append(a[ok[:, k], k])

                # occlusion proximity (only images that actually have occlusion)
                if split == 'occ' and regions['occluded_unvotable'] is not None \
                        and regions['occluded_unvotable'].any():
                    occ_images_with_occl += 1
                    dist_occ = regions['dist_occ'][ys, xs]               # [S]
                    for bi, (lo, hi) in enumerate(PROX_BINS):
                        sel = ok & (dist_occ >= lo)[:, None] & (dist_occ < hi)[:, None]
                        if sel.any():
                            prox_store[bi]['e_vote'].append(e_vote[sel].ravel())
                            prox_store[bi]['dir'].append(ang[sel].ravel())
                            prox_store[bi]['mag_raw'].append(mag_raw[sel].ravel())
                            prox_store[bi]['endp_unit'].append(endp_unit[sel].ravel())
                            prox_img_count[bi].add(image_id)
            if (image_id + 1) % 10 == 0:
                log(f'{split} img {image_id+1}/{len(ds)}  t={time.time()-T0:.0f}s')

    # ---------------- direction_error.csv ----------------
    with open(os.path.join(RES_DIR, 'direction_error.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['split', 'region', 'count', 'mean_deg', 'median_deg', 'p90_deg',
                     f'bad_rate_gt_{int(DIR_BAD_DEG)}deg'])
        for s in ('linemod', 'occ'):
            for r in REGIONS:
                es = store[s]['dir'][r]
                if not es:
                    wr.writerow([s, r, 0, '', '', '', '']); continue
                a = np.concatenate(es); ms = mstats(a)
                wr.writerow([s, r, ms['count'], f"{ms['mean']:.3f}", f"{ms['median']:.3f}",
                             f"{ms['p90']:.3f}", f"{bad_rate(a, DIR_BAD_DEG):.4f}"])

    # ---------------- magnitude_error.csv ----------------
    with open(os.path.join(RES_DIR, 'magnitude_error.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['split', 'region', 'count',
                     'abs_raw_mean', 'abs_raw_median', 'abs_raw_p90',
                     'norm_raw_mean', 'norm_raw_median', 'norm_raw_p90',
                     'unit_dev_mean(|m-1|)', 'unit_dev_median', 'unit_dev_p90',
                     'pred_mag_mean', 'pred_mag_median', 'pred_mag_p90'])
        for s in ('linemod', 'occ'):
            for r in REGIONS:
                es = store[s]['mag_raw'][r]
                if not es:
                    wr.writerow([s, r, 0] + [''] * 12); continue
                raw = np.concatenate(es); nrm = np.concatenate(store[s]['mag_norm_raw'][r])
                dev = np.concatenate(store[s]['mag_dev_unit'][r])
                pm = np.concatenate(store[s]['pred_mag'][r])
                wr.writerow([s, r, raw.size,
                             f"{raw.mean():.3f}", f"{np.median(raw):.3f}", f"{np.quantile(raw, .9):.3f}",
                             f"{nrm.mean():.4f}", f"{np.median(nrm):.4f}", f"{np.quantile(nrm, .9):.4f}",
                             f"{dev.mean():.4f}", f"{np.median(dev):.4f}", f"{np.quantile(dev, .9):.4f}",
                             f"{pm.mean():.4f}", f"{np.median(pm):.4f}", f"{np.quantile(pm, .9):.4f}"])

    # ---------------- endpoint_error.csv ----------------
    with open(os.path.join(RES_DIR, 'endpoint_error.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['split', 'region', 'count',
                     'unit_mean', 'unit_median', 'unit_p90', f'unit_bad_rate_gt_{int(ENDP_BAD_PX)}px',
                     'raw_mean', 'raw_median', 'raw_p90', f'raw_bad_rate_gt_{int(ENDP_BAD_PX)}px',
                     'mean_dist_to_kp_px'])
        for s in ('linemod', 'occ'):
            for r in REGIONS:
                es = store[s]['endp_unit'][r]
                if not es:
                    wr.writerow([s, r, 0] + [''] * 10); continue
                eu = np.concatenate(es); er = np.concatenate(store[s]['endp_raw'][r])
                dk = np.concatenate(store[s]['dist_kp'][r])
                wr.writerow([s, r, eu.size,
                             f"{eu.mean():.3f}", f"{np.median(eu):.3f}", f"{np.quantile(eu, .9):.3f}",
                             f"{bad_rate(eu, ENDP_BAD_PX):.4f}",
                             f"{er.mean():.2f}", f"{np.median(er):.2f}", f"{np.quantile(er, .9):.2f}",
                             f"{bad_rate(er, ENDP_BAD_PX):.4f}",
                             f"{dk.mean():.2f}"])

    # ---------------- keypoint_error.csv ----------------
    with open(os.path.join(RES_DIR, 'keypoint_error.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['split', 'kp', 'count', 'dir_mean_deg', 'dir_p90_deg', 'dir_bad_rate',
                     'mag_unit_dev_mean', 'mag_abs_raw_mean', 'endp_unit_mean', 'endp_unit_p90',
                     'endp_raw_mean'])
        for s in ('linemod', 'occ'):
            for k in range(KP):
                d = np.concatenate(kp_store[s]['dir'][k])
                wr.writerow([s, f'kp{k}', d.size,
                             f"{d.mean():.3f}", f"{np.quantile(d, .9):.3f}",
                             f"{bad_rate(d, DIR_BAD_DEG):.4f}",
                             f"{np.concatenate(kp_store[s]['mag_dev_unit'][k]).mean():.4f}",
                             f"{np.concatenate(kp_store[s]['mag_raw'][k]).mean():.2f}",
                             f"{np.concatenate(kp_store[s]['endp_unit'][k]).mean():.3f}",
                             f"{np.quantile(np.concatenate(kp_store[s]['endp_unit'][k]), .9):.3f}",
                             f"{np.concatenate(kp_store[s]['endp_raw'][k]).mean():.2f}"])

    # ---------------- direction_magnitude_diagnostic.csv ----------------
    with open(os.path.join(RES_DIR, 'direction_magnitude_diagnostic.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['split', 'scope', 'endp_raw_original', 'endp_raw_direction_only',
                     'endp_raw_magnitude_only', 'endp_unit_original',
                     'endp_unit_direction_only', 'endp_unit_magnitude_only'])
        for s in ('linemod', 'occ'):
            for r in REGIONS:
                if not store[s]['endp_raw'][r]:
                    continue
                er = np.concatenate(store[s]['endp_raw'][r]).mean()
                dr = np.concatenate(store[s]['dironly_raw'][r]).mean()
                mr = np.concatenate(store[s]['magonly_raw'][r]).mean()
                eu = np.concatenate(store[s]['endp_unit'][r]).mean()
                du = np.concatenate(store[s]['dironly_unit'][r]).mean()
                mu = np.concatenate(store[s]['magonly_unit'][r]).mean()
                wr.writerow([s, r, f'{er:.2f}', f'{dr:.2f}', f'{mr:.2f}',
                             f'{eu:.4f}', f'{du:.4f}', f'{mu:.4f}'])

    # ---------------- occlusion_proximity.csv ----------------
    with open(os.path.join(RES_DIR, 'occlusion_proximity.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['dist_to_occlusion_bin_px', 'n_images', 'n_votes',
                     'e_vote_mean', 'e_vote_median', 'dir_mean_deg', 'dir_median_deg',
                     'mag_abs_raw_mean', 'endp_unit_mean', 'endp_unit_median'])
        for bi, (lo, hi) in enumerate(PROX_BINS):
            if not prox_store[bi]['e_vote']:
                wr.writerow([f'{lo}-{hi if np.isfinite(hi) else "inf"}', 0, 0] + [''] * 7)
                continue
            ev = np.concatenate(prox_store[bi]['e_vote'])
            dg = np.concatenate(prox_store[bi]['dir'])
            mr = np.concatenate(prox_store[bi]['mag_raw'])
            eu = np.concatenate(prox_store[bi]['endp_unit'])
            wr.writerow([f'{lo}-{hi if np.isfinite(hi) else "inf"}',
                         len(prox_img_count[bi]), ev.size,
                         f'{ev.mean():.3f}', f'{np.median(ev):.3f}',
                         f'{dg.mean():.3f}', f'{np.median(dg):.3f}',
                         f'{mr.mean():.3f}', f'{eu.mean():.3f}', f'{np.median(eu):.3f}'])

    # ---------------- figures (3 max) ----------------
    # Fig 1: LIN vs OCC direction / magnitude / endpoint
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    lin = lambda s, m, r='all': np.concatenate(store[s][m][r])
    pairs = [('LINEMOD', 'linemod'), ('OCC', 'occ')]
    d_l, d_o = lin('linemod', 'dir'), lin('occ', 'dir')
    m_l, m_o = lin('linemod', 'mag_dev_unit'), lin('occ', 'mag_dev_unit')
    e_l, e_o = lin('linemod', 'endp_unit'), lin('occ', 'endp_unit')
    axs[0].bar(['LIN', 'OCC'], [d_l.mean(), d_o.mean()], color=['steelblue', 'indianred'])
    axs[0].set_title(f'direction error (deg)  median {np.median(d_l):.2f} vs {np.median(d_o):.2f}')
    axs[1].bar(['LIN', 'OCC'], [m_l.mean(), m_o.mean()], color=['steelblue', 'indianred'])
    axs[1].set_title(f'magnitude |m-1| (unit space)  median {np.median(m_l):.4f} vs {np.median(m_o):.4f}')
    axs[2].bar(['LIN', 'OCC'], [e_l.mean(), e_o.mean()], color=['steelblue', 'indianred'])
    axs[2].set_title(f'endpoint error (unit space, px)  median {np.median(e_l):.3f} vs {np.median(e_o):.3f}')
    for a in axs:
        a.grid(alpha=.3, axis='y')
        for j, v in enumerate([a.patches[0].get_height(), a.patches[1].get_height()]):
            a.text(j, v, f'{v:.3f}', ha='center', va='bottom')
    plt.tight_layout(); plt.savefig(os.path.join(RES_DIR, 'fig1_lin_vs_occ.png'), dpi=120); plt.close()

    # Fig 2: OCC regions
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    regs3 = ['interior', 'normal_boundary', 'occlusion_boundary']
    vals = {m: [np.concatenate(store['occ'][m][r]).mean() for r in regs3]
            for m in ('dir', 'mag_dev_unit', 'endp_unit')}
    med = {m: [np.median(np.concatenate(store['occ'][m][r])) for r in regs3] for m in vals}
    titles = {'dir': 'direction error (deg)', 'mag_dev_unit': 'magnitude |m-1| (unit space)',
              'endp_unit': 'endpoint error (unit space, px)'}
    for a, m in zip(axs, ('dir', 'mag_dev_unit', 'endp_unit')):
        a.bar(range(3), vals[m], color='indianred')
        a.set_xticks(range(3)); a.set_xticklabels(regs3, rotation=12)
        a.set_title(f'OCC {titles[m]}  (median {med[m][0]:.3f}/{med[m][1]:.3f}/{med[m][2]:.3f})')
        for j, v in enumerate(vals[m]):
            a.text(j, v, f'{v:.3f}', ha='center', va='bottom')
        a.grid(alpha=.3, axis='y')
    plt.tight_layout(); plt.savefig(os.path.join(RES_DIR, 'fig2_occ_regions.png'), dpi=120); plt.close()

    # Fig 3: per-keypoint LIN vs OCC endpoint error (unit space)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(KP); wd = 0.38
    kp_mean = lambda s: [np.concatenate(kp_store[s]['endp_unit'][k]).mean() for k in range(KP)]
    ax.bar(x - wd/2, kp_mean('linemod'), wd, label='LINEMOD', color='steelblue')
    ax.bar(x + wd/2, kp_mean('occ'), wd, label='OCC', color='indianred')
    ax.set_xticks(x); ax.set_xticklabels([f'kp{k}' for k in range(KP)])
    ax.set_ylabel('endpoint error (unit space, px)'); ax.legend()
    ax.set_title('per-keypoint endpoint error (all fg pixels)')
    ax.grid(alpha=.3, axis='y')
    plt.tight_layout(); plt.savefig(os.path.join(RES_DIR, 'fig3_per_keypoint.png'), dpi=120); plt.close()

    # ---------------- summary ----------------
    log(f"OCC images with occluded region: {occ_images_with_occl}/20 "
        f"(proximity analysis limited to these; LINEMOD has no occlusion -> skipped)")
    for s in ('linemod', 'occ'):
        log(f"{s}: dir {mstats(np.concatenate(store[s]['dir']['all']))} "
            f"mag_dev {np.concatenate(store[s]['mag_dev_unit']['all']).mean():.4f} "
            f"endp_unit {np.concatenate(store[s]['endp_unit']['all']).mean():.4f} "
            f"endp_raw {np.concatenate(store[s]['endp_raw']['all']).mean():.2f}")
    log(f'DONE in {time.time()-T0:.1f}s')


if __name__ == '__main__':
    main()
