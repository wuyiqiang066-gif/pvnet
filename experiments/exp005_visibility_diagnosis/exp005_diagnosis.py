"""EXP005: Visibility vs Representation Failure Diagnosis (inference-only).

Question: is OCC degradation mainly (A) fewer visible votes, or (B) worse
vertex representation of the remaining visible pixels?

Strict diagnosis: no training, no model/baseline/RANSAC/CUDA/PnP/evaluator
changes. Data: 20 LINEMOD val + 20 OCC val images (same deterministic first-20
IDs as EXP004), checkpoint 199.pth, baseline PVNet inference only.

Region classes (reused from diagnosis_vote_analysis, GT-derived, no guessing):
  interior          = visible, eroded 5x by 3x3 kernel
  normal_boundary   = visible, within 5px of silhouette, not near occlusion
  occlusion_boundary= visible, within 5px of occluded (amodal & ~visible) region
  occluded_unvotable= amodal & ~visible (no vote possible)
If an image's amodal mask file is missing -> occlusion classes unavailable
for that image (marked, excluded from occlusion stats; never guessed).

Run from repo root: python experiments/exp005_visibility_diagnosis/exp005_diagnosis.py
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
from lib.utils.data_utils import LineModImageDB, OcclusionLineModImageDB, LineModModelDB
from lib.utils.config import cfg
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v3
from lib.utils.evaluation_utils import pnp

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(EXP_DIR, 'results')
os.makedirs(RES_DIR, exist_ok=True)

N_IMG = 20
KP = 9
BAD_VOTE_PX = 10.0        # spec: bad vote if e > 10 px
ANGLE_BAD_DEG = 30.0      # bad vertex direction if angular error > 30 deg
BOUNDARY_PX = 5           # same as diagnosis_vote_analysis (comparable defs)
NEAR_OCC_PX = 5
TOP_FRAC = 0.5            # oracle top-50% reliable votes
N_REPS = 3                # equal-count subsampling repetitions per image
K_LINEMOD = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
MODEL_SUB = 1000

torch.manual_seed(0); np.random.seed(0)
T0 = time.time()
LOG = open(os.path.join(RES_DIR, 'run.log'), 'w', buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + '\n')

json.dump(dict(cls='cat', n_img_per_split=N_IMG, ckpt='data/model/cat_linemod_train/199.pth',
               image_selection='LINEMOD val_real_set[:20]; OCC test_real_set[:len//2][:20] '
                               '(identical to EXP004 deterministic first-20)',
               boundary_px=BOUNDARY_PX, near_occ_px=NEAR_OCC_PX, bad_vote_px=BAD_VOTE_PX,
               angle_bad_deg=ANGLE_BAD_DEG, top_frac=TOP_FRAC, n_reps=N_REPS,
               vote_def='e_ik = perpendicular distance(GT kp_k, vote LINE of pixel i) '
                        '[unit-direction field, same as diagnosis/EXP004]',
               vertex_def='angular error acos(clamp(dot(v_pred_hat, v_gt_hat))), '
                          'v_gt = unit(GT kp - pixel); pixels within 1px of kp skipped',
               equal_count='LINEMOD image i subsampled to OCC image i visible-fg count; '
                           'LS line-intersection estimator on both sides; 3 reps',
               pose_sensitivity='stock ransac_voting_layer_v3 + stock pnp on GT visible '
                                'mask vs occlusion-boundary-removed mask (OCC only)'),
          open(os.path.join(RES_DIR, 'config.json'), 'w'), indent=2)

# ---------------- model / geometry ----------------
device = 'cuda'
net = Resnet18_8s(ver_dim=KP * 2, seg_dim=2)
sd = torch.load(os.path.join(ROOT, 'data/model/cat_linemod_train/199.pth'), map_location='cpu')
net.load_state_dict(sd['net'])
net = net.to(device).eval()

MODELDB = LineModModelDB()
POINTS_3D = np.concatenate([MODELDB.get_farthest_3d('cat'),
                            MODELDB.get_centers_3d('cat')[None, :]], 0)   # [9,3]
DIAMETER = MODELDB.get_diameter('cat')
mesh, _ = MODELDB.get_ply_mesh('cat')
MESH = mesh[np.random.RandomState(0).choice(len(mesh), MODEL_SUB, replace=False)]

AUG_CFG = json.load(open(os.path.join(
    ROOT, 'configs', 'exp001_reliability_voting_scratch.json')))['aug_cfg']

AMODAL_DIR = {'linemod': os.path.join(cfg.LINEMOD, 'cat', 'amodal_mask'),
              'occ': os.path.join(cfg.OCCLUSION_LINEMOD, 'amodal_masks', 'cat')}


def region_labels(vis_mask, amodal_mask):
    """Same definitions as diagnosis_vote_analysis/diagnosis.py."""
    m = (vis_mask > 0)
    interior = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8),
                         iterations=BOUNDARY_PX).astype(bool)
    out = {'interior': interior, 'boundary': m & ~interior}
    if amodal_mask is not None:
        occ_region = amodal_mask & ~m
        out['occluded_unvotable'] = occ_region
        dist_occ = cv2.distanceTransform((~occ_region).astype(np.uint8), cv2.DIST_L2, 3)
        occl_b = m & (dist_occ <= NEAR_OCC_PX)
        out['occl_boundary'] = occl_b
        out['interior'] = out['interior'] & ~occl_b
        out['boundary'] = out['boundary'] & ~occl_b
    else:
        out['occluded_unvotable'] = None
        out['occl_boundary'] = np.zeros_like(interior)
    return out


def line_err(dirs, coords, kp):
    """perpendicular distance from kp to each line (coords[i], dirs[i]); dirs unit."""
    v = kp[None, :] - coords
    t = (v * dirs).sum(1, keepdims=True)
    return np.linalg.norm(v - t * dirs, axis=1)


def ls_intersect(dirs, coords):
    """least-squares intersection of vote lines (same as diagnosis)."""
    P = np.eye(2)[None] - dirs[:, :, None] * dirs[:, None, :]
    A = P.sum(0); b = (P @ coords[:, :, None]).sum(0)
    try:
        return np.linalg.solve(A + 1e-6 * np.eye(2), b).ravel()
    except np.linalg.LinAlgError:
        return coords.mean(0)


def pose_add(pose_pred, pose_gt):
    R, t = pose_pred[:, :3], pose_pred[:, 3]
    Rg, tg = pose_gt[:, :3], pose_gt[:, 3]
    m_pred = MESH @ R.T + t; m_gt = MESH @ Rg.T + tg
    add = np.linalg.norm(m_pred - m_gt, axis=1).mean()
    d2 = ((m_pred[:, None, :] - m_gt[None, :, :]) ** 2).sum(-1)
    adds = np.sqrt(d2.min(1)).mean()
    return add, adds


def stat_row(e):
    e = np.asarray(e, np.float64)
    return dict(count=int(e.size), mean=float(e.mean()), median=float(np.median(e)),
                p90=float(np.quantile(e, 0.9)),
                bad_rate=float((e > BAD_VOTE_PX).mean()))


def main():
    from torch.utils.data import DataLoader as DL
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'tle', os.path.join(ROOT, 'tools', 'train_linemod_exp001.py'))
    tle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tle)
    ImageSizeBatchSampler = tle.ImageSizeBatchSampler

    # pooled per-split stores
    store = {s: dict(e={r: [] for r in ('all', 'interior', 'normal_boundary', 'occlusion_boundary')},
                     ang={r: [] for r in ('all', 'interior', 'normal_boundary', 'occlusion_boundary')},
                     oracle=[], px_rows=[], ids=[])
             for s in ('linemod', 'occ')}
    equal_rows = []
    pose_rows = []
    amodal_missing = {s: [] for s in ('linemod', 'occ')}

    pools = {}   # per split per image: coords, dirs (unit), gt2 -> for equal-count
    gt_masks = {}  # occ region masks needed for pose sensitivity (per occ img)

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
        loader = DL(ds, batch_sampler=ImageSizeBatchSampler(
            SequentialSampler(ds), 1, False, AUG_CFG), num_workers=6)
        log(f'--- {split}: {len(ds)} images ---')

        for image_id, data in enumerate(loader):
            image, mask_gt, vertex_gt, vw, pose, gt_kps = [d.to(device) for d in data]
            with torch.no_grad():
                seg_pred, vertex_pred = net(image)
                b, _, h, w = vertex_pred.shape
                ver = vertex_pred.permute(0, 2, 3, 1).contiguous().view(b, h, w, -1, 2)[0]
                gt2 = gt_kps[0, :, :2] if gt_kps.dim() == 3 else gt_kps[0, 0, :, :2]
                pose_gt = pose[0].cpu().numpy()                      # [3,4]

                vis = (mask_gt[0].cpu().numpy() == 1)

                # ---- amodal mask (occlusion classes; never guessed) ----
                rgb_pth = ds.imagedb[image_id]['rgb_pth']
                mid = re.findall(r'(\d+)', os.path.basename(rgb_pth))[0]
                ap = os.path.join(AMODAL_DIR[split], str(int(mid)) + '.png')
                amodal = None
                if os.path.exists(ap):
                    amodal = (np.asarray(Image.open(ap)) > 0)
                else:
                    amodal_missing[split].append(rgb_pth)
                regions = region_labels(vis, amodal)

                # ---- pixel pools on GT visible mask ----
                ys, xs = np.nonzero(vis)
                coords = np.stack([xs, ys], 1).astype(np.float64)
                dirs = ver[ys, xs].cpu().numpy().astype(np.float64)   # [S,9,2]
                nrm = np.linalg.norm(dirs, axis=2, keepdims=True); nrm[nrm == 0] = 1
                dirs = dirs / nrm
                S = len(coords)
                E = np.stack([line_err(dirs[:, k], coords, gt2[k].cpu().numpy())
                              for k in range(KP)], 1)                # [S,9]

                # ---- vertex angular error vs analytical GT direction ----
                gt_np = gt2.cpu().numpy()
                diff = gt_np[None, :, :] - coords[:, None, :]        # [S,9,2]
                dn = np.linalg.norm(diff, axis=2, keepdims=True)
                ok_diff = (dn[:, :, 0] >= 1.0)
                vgt = diff / np.maximum(dn, 1e-9)
                dot = (dirs * vgt).sum(-1)                           # [S,9]
                ang = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
                ang[~ok_diff] = np.nan

                # ---- region stats ----
                st = store[split]
                rmask = {r: regions[r][ys, xs] for r in
                         ('interior', 'boundary', 'occl_boundary')}
                st['e']['all'].append(E.ravel())
                a_all = ang.ravel()
                st['ang']['all'].append(a_all[~np.isnan(a_all)])
                for r_out, r_in in (('interior', 'interior'),
                                    ('normal_boundary', 'boundary'),
                                    ('occlusion_boundary', 'occl_boundary')):
                    sel = rmask[r_in]
                    if sel.sum():
                        st['e'][r_out].append(E[sel].ravel())
                        a = ang[sel].ravel()
                        st['ang'][r_out].append(a[~np.isnan(a)])
                if amodal is not None:
                    st['oracle'].append((rmask['interior'], rmask['boundary'],
                                         rmask['occl_boundary'], E))

                st['ids'].append(dict(img_id=image_id, rgb=os.path.basename(rgb_pth),
                                      n_vis=int(vis.sum()),
                                      n_amodal=int(amodal.sum()) if amodal is not None else -1,
                                      n_occluded=int(regions['occluded_unvotable'].sum())
                                      if amodal is not None else -1,
                                      n_interior_px=int(rmask['interior'].sum()),
                                      n_normal_boundary_px=int(rmask['boundary'].sum()),
                                      n_occl_boundary_px=int(rmask['occl_boundary'].sum())))

                # ---- keep pools for equal-count (linemod) and pose sens (occ) ----
                if split == 'linemod':
                    pools[image_id] = (coords, dirs, gt_np)
                else:
                    gt_masks[image_id] = dict(vis=vis, occlb=regions['occl_boundary'],
                                              ver=ver, gt2=gt2, pose_gt=pose_gt,
                                              vertex_pred=vertex_pred, h=h, w=w)
            if (image_id + 1) % 10 == 0:
                log(f'{split} img {image_id+1}/{len(ds)}  t={time.time()-T0:.0f}s')

    # ---------------- pixel_statistics.csv ----------------
    with open(os.path.join(RES_DIR, 'pixel_statistics.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['split', 'img_id', 'rgb', 'visible_fg_px', 'occluded_px',
                     'amodal_fg_px', 'interior_px', 'normal_boundary_px', 'occlusion_boundary_px'])
        for s in ('linemod', 'occ'):
            for r in store[s]['ids']:
                wr.writerow([s, r['img_id'], r['rgb'], r['n_vis'], r['n_occluded'],
                             r['n_amodal'], r['n_interior_px'], r['n_normal_boundary_px'],
                             r['n_occl_boundary_px']])
        for s in ('linemod', 'occ'):
            rows = store[s]['ids']
            for key, name in (('n_vis', 'visible_fg_px'), ('n_occluded', 'occluded_px'),
                              ('n_amodal', 'amodal_fg_px')):
                vals = [r[key] for r in rows if r[key] >= 0]
                if vals:
                    wr.writerow([s, 'MEAN', name, f'{np.mean(vals):.1f}', '',
                                 f'median={np.median(vals):.1f}'])

    # ---------------- vote_error_by_region.csv ----------------
    with open(os.path.join(RES_DIR, 'vote_error_by_region.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['split', 'region', 'votes', 'mean_e_px', 'median_e_px', 'p90_e_px',
                     f'bad_vote_rate_e_gt_{int(BAD_VOTE_PX)}px'])
        for s in ('linemod', 'occ'):
            for r in ('all', 'interior', 'normal_boundary', 'occlusion_boundary'):
                es = store[s]['e'][r]
                if not es:
                    wr.writerow([s, r, 0, '', '', '', ''])
                    continue
                e = np.concatenate(es)
                st = stat_row(e)
                wr.writerow([s, r, st['count'], f"{st['mean']:.3f}", f"{st['median']:.3f}",
                             f"{st['p90']:.3f}", f"{st['bad_rate']:.4f}"])

    # ---------------- vertex_error_by_region.csv ----------------
    with open(os.path.join(RES_DIR, 'vertex_error_by_region.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['split', 'region', 'samples', 'mean_angle_deg', 'median_angle_deg',
                     'p90_angle_deg', f'bad_rate_gt_{int(ANGLE_BAD_DEG)}deg'])
        for s in ('linemod', 'occ'):
            for r in ('all', 'interior', 'normal_boundary', 'occlusion_boundary'):
                as_ = store[s]['ang'][r]
                if not as_:
                    wr.writerow([s, r, 0, '', '', '', ''])
                    continue
                a = np.concatenate(as_)
                st = stat_row(a)
                wr.writerow([s, r, st['count'], f"{st['mean']:.3f}", f"{st['median']:.3f}",
                             f"{st['p90']:.3f}", f"{st['bad_rate']:.4f}"])

    # ---------------- oracle_region_analysis.csv ----------------
    # per (image, keypoint): top-50% / bottom-50% by e_ik -> region composition
    with open(os.path.join(RES_DIR, 'oracle_region_analysis.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['split', 'region', 'vote_fraction_all', 'vote_fraction_top50',
                     'vote_fraction_bottom50'])
        for s in ('linemod', 'occ'):
            tot = {r: 0.0 for r in ('interior', 'normal_boundary', 'occlusion_boundary')}
            top = {r: 0.0 for r in tot}; bot = {r: 0.0 for r in tot}; total = 0.0
            for (ri, rb, ro, E) in store[s]['oracle']:
                selmap = {'interior': ri, 'normal_boundary': rb, 'occlusion_boundary': ro}
                for k in range(KP):
                    e_k = E[:, k]
                    n = len(e_k)
                    if n < 2:
                        continue
                    order = np.argsort(e_k)
                    n_top = max(1, int(round(TOP_FRAC * n)))
                    for r, sel in selmap.items():
                        tot[r] += sel.sum()
                        top[r] += sel[order[:n_top]].sum()
                        bot[r] += sel[order[n - n_top:]].sum()
                    total += n
            for r in tot:
                wr.writerow([s, r, f'{tot[r]/max(total,1):.4f}',
                             f'{top[r]/max(total,1):.4f}', f'{bot[r]/max(total,1):.4f}'])

    # ---------------- equal-count counterfactual ----------------
    # OCC image i vs LINEMOD image i subsampled to the same visible-fg count.
    for i in range(N_IMG):
        coords_o = None
        if i in gt_masks:
            vm = gt_masks[i]['vis']
            ys, xs = np.nonzero(vm)
            coords_o = np.stack([xs, ys], 1).astype(np.float64)
            dirs_o = gt_masks[i]['ver'][ys, xs].cpu().numpy().astype(np.float64)
            nrm = np.linalg.norm(dirs_o, axis=2, keepdims=True); nrm[nrm == 0] = 1
            dirs_o = dirs_o / nrm
            gt_o = gt_masks[i]['gt2'].cpu().numpy()
        if coords_o is None or i not in pools:
            continue
        coords_l, dirs_l, gt_l = pools[i]
        n_occ = len(coords_o)

        def kp_err(coords, dirs, gt):
            kp = np.stack([ls_intersect(dirs[:, k], coords, ) for k in range(KP)], 0)
            return np.linalg.norm(kp - gt, axis=1)   # [9]

        err_occ = kp_err(coords_o, dirs_o, gt_o)
        n_lin_use = min(n_occ, len(coords_l))
        sub_errs = []
        for rep in range(N_REPS):
            rng = np.random.RandomState(1000 + i * 10 + rep)
            idx = rng.choice(len(coords_l), n_lin_use, replace=False)
            sub_errs.append(kp_err(coords_l[idx], dirs_l[idx], gt_l))
        sub_errs = np.stack(sub_errs, 0)             # [3,9]
        err_lin_full = kp_err(coords_l, dirs_l, gt_l)
        equal_rows.append(dict(img=i, n_occ=n_occ, n_lin_full=len(coords_l),
                               n_lin_sub=n_lin_use,
                               kp_err_lin_full=float(err_lin_full.mean()),
                               kp_err_lin_sub_mean=float(sub_errs.mean()),
                               kp_err_lin_sub_std=float(sub_errs.std()),
                               kp_err_occ=float(err_occ.mean()),
                               kp_err_lin_sub_per_rep=[float(x.mean()) for x in sub_errs]))
    with open(os.path.join(RES_DIR, 'equal_count_analysis.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['img', 'n_occ_visible', 'n_lin_full', 'n_lin_sub',
                     'kp_err_lin_full_mean_px', 'kp_err_lin_sub_mean_px',
                     'kp_err_lin_sub_std_px', 'kp_err_occ_mean_px',
                     'kp_err_lin_sub_per_rep_px'])
        for r in equal_rows:
            wr.writerow([r['img'], r['n_occ'], r['n_lin_full'], r['n_lin_sub'],
                         f"{r['kp_err_lin_full']:.2f}", f"{r['kp_err_lin_sub_mean']:.2f}",
                         f"{r['kp_err_lin_sub_std']:.2f}", f"{r['kp_err_occ']:.2f}",
                         ';'.join(f'{x:.2f}' for x in r['kp_err_lin_sub_per_rep'])])

    # ---------------- pose sensitivity (OCC only): remove occlusion boundary ----
    log('pose sensitivity: stock RANSAC + stock pnp, GT visible mask vs occl-boundary-removed')
    for i in range(N_IMG):
        if i not in gt_masks:
            continue
        g = gt_masks[i]
        m_all = torch.from_numpy(g['vis'].astype(np.int64))[None].to(device)
        m_nocb = m_all.clone()
        if g['occlb'].any():
            m_nocb[0][torch.from_numpy(g['occlb']).to(device)] = 0
        ver_t = g['vertex_pred'].permute(0, 2, 3, 1).view(1, g['h'], g['w'], KP, 2)
        adds = {}
        for name, mk in (('all_visible', m_all), ('no_occlusion_boundary', m_nocb)):
            with torch.no_grad():
                kp = ransac_voting_layer_v3(mk, ver_t, 512, inlier_thresh=0.99)[0]
                kp = kp.cpu().numpy()
                if not np.isfinite(kp).all():
                    adds[name] = dict(add_s=-1.0, add_s_pass=False)
                    continue
                pose_pred = pnp(POINTS_3D, kp, K_LINEMOD)
                add, adds_v = pose_add(pose_pred, g['pose_gt'])
                adds[name] = dict(add_s=float(adds_v), add_s_pass=bool(adds_v < 0.1 * DIAMETER))
        pose_rows.append(dict(img=i, **adds['all_visible'], **{f'nocb_{k}': v for k, v in adds['no_occlusion_boundary'].items()}))
    with open(os.path.join(RES_DIR, 'pose_sensitivity.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['img', 'add_s_all_visible', 'add_s_pass_all_visible',
                     'add_s_no_occlusion_boundary', 'add_s_pass_no_occlusion_boundary'])
        for r in pose_rows:
            wr.writerow([r['img'], f"{r['add_s']:.4f}", r['add_s_pass'],
                         f"{r['nocb_add_s']:.4f}", r['nocb_add_s_pass']])

    # ---------------- figures (3 max) ----------------
    e_lin = np.concatenate(store['linemod']['e']['all'])
    e_occ = np.concatenate(store['occ']['e']['all'])
    n_lin_px = np.mean([r['n_vis'] for r in store['linemod']['ids']])
    n_occ_px = np.mean([r['n_vis'] for r in store['occ']['ids']])

    # Fig 1: vote error by region
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    regs = ['interior', 'normal_boundary', 'occlusion_boundary']
    for j, s in enumerate(('linemod', 'occ')):
        means, p90s, meds = [], [], []
        for r in regs:
            es = store[s]['e'][r]
            if es:
                e = np.concatenate(es); means.append(e.mean()); meds.append(np.median(e)); p90s.append(np.quantile(e, .9))
            else:
                means.append(np.nan); meds.append(np.nan); p90s.append(np.nan)
        x = np.arange(3)
        axs[0].bar(x, means, 0.55, label=s); axs[0].set_xticks(x); axs[0].set_xticklabels(regs, rotation=15)
        axs[0].set_title(f'{s}: mean vote error (px)')
        axs[1].bar(x, meds, 0.55); axs[1].set_xticks(x); axs[1].set_xticklabels(regs, rotation=15)
        axs[1].set_title(f'{s}: median vote error (px)')
        axs[2].bar(x, p90s, 0.55); axs[2].set_xticks(x); axs[2].set_xticklabels(regs, rotation=15)
        axs[2].set_title(f'{s}: p90 vote error (px)')
    plt.tight_layout(); plt.savefig(os.path.join(RES_DIR, 'fig1_vote_error_by_region.png'), dpi=120); plt.close()

    # Fig 2: LIN vs OCC vote count + vote error
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].boxplot([[r['n_vis'] for r in store['linemod']['ids']],
                    [r['n_vis'] for r in store['occ']['ids']]], tick_labels=['LINEMOD', 'OCC'])
    axs[0].set_title(f'visible fg pixels/image (mean {n_lin_px:.0f} vs {n_occ_px:.0f})')
    axs[0].grid(alpha=.3)
    stats = ['mean', 'median', 'p90']
    el, eo = stat_row(e_lin), stat_row(e_occ)
    x = np.arange(3); wd = 0.35
    axs[1].bar(x - wd/2, [el['mean'], el['median'], el['p90']], wd, label='LINEMOD')
    axs[1].bar(x + wd/2, [eo['mean'], eo['median'], eo['p90']], wd, label='OCC')
    axs[1].set_xticks(x); axs[1].set_xticklabels(stats); axs[1].legend()
    axs[1].set_title('vote line error e_ik (px, all fg votes)')
    axs[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(RES_DIR, 'fig2_lin_vs_occ.png'), dpi=120); plt.close()

    # Fig 3: equal-count counterfactual
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lf = np.mean([r['kp_err_lin_full'] for r in equal_rows])
    ls_ = np.mean([r['kp_err_lin_sub_mean'] for r in equal_rows])
    ls_sd = np.std([r['kp_err_lin_sub_mean'] for r in equal_rows])
    oc = np.mean([r['kp_err_occ'] for r in equal_rows])
    oc_sd = np.std([r['kp_err_occ'] for r in equal_rows])
    bars = ax.bar(['LIN full', 'LIN subsampled\nto OCC count', 'OCC all visible'],
                  [lf, ls_, oc], yerr=[0, ls_sd, oc_sd], capsize=5,
                  color=['steelblue', 'skyblue', 'indianred'])
    for b, v in zip(bars, [lf, ls_, oc]):
        ax.text(b.get_x() + b.get_width()/2, v, f'{v:.2f}', ha='center', va='bottom')
    ax.set_ylabel('keypoint error (px, mean over 9 kps)')
    ax.set_title('Equal-count counterfactual (LS estimator, 20 img pairs)')
    ax.grid(alpha=.3, axis='y')
    plt.tight_layout(); plt.savefig(os.path.join(RES_DIR, 'fig3_equal_count.png'), dpi=120); plt.close()

    # ---------------- summary log ----------------
    json.dump(dict(amodal_missing={s: amodal_missing[s] for s in amodal_missing},
                   image_ids={s: store[s]['ids'] for s in store},
                   elapsed_s=time.time() - T0),
              open(os.path.join(RES_DIR, 'image_ids.json'), 'w'), indent=2)
    log(f"LIN votes={len(e_lin)}  OCC votes={len(e_occ)}")
    log(f"LIN e: {stat_row(e_lin)}")
    log(f"OCC e: {stat_row(e_occ)}")
    if equal_rows:
        log(f"equal-count: LIN_sub mean kp err = "
            f"{np.mean([r['kp_err_lin_sub_mean'] for r in equal_rows]):.2f}px, "
            f"OCC = {np.mean([r['kp_err_occ'] for r in equal_rows]):.2f}px")
    if pose_rows:
        log(f"pose OCC: ADD(-S) all={np.mean([r['add_s'] for r in pose_rows])*1000:.1f}mm "
            f"no_occlb={np.mean([r['nocb_add_s'] for r in pose_rows])*1000:.1f}mm "
            f"(pass@0.1d: {np.mean([r['add_s_pass'] for r in pose_rows]):.2f} vs "
            f"{np.mean([r['nocb_add_s_pass'] for r in pose_rows]):.2f})")
    log(f'DONE in {time.time()-T0:.1f}s')


if __name__ == '__main__':
    main()
