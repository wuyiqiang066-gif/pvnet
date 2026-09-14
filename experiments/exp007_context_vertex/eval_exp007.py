"""EXP007 evaluation: pose metrics + EXP006-style vertex diagnosis for one model.

Evaluates on the EXP005/EXP006 deterministic image sets (20 LINEMOD val +
20 OCC test), using baseline RANSAC voting + baseline pnp ONLY.
Vertex diagnosis metrics are identical to EXP006 (direction error, endpoint
error in supervision space, per-keypoint, occlusion proximity).

Run from repo root:
  python experiments/exp007_context_vertex/eval_exp007.py --model baseline --ckpt data/model/cat_linemod_train/199.pth --tag ref199
  python experiments/exp007_context_vertex/eval_exp007.py --model baseline --ckpt experiments/exp007_context_vertex/checkpoints/controlA/last.pth --tag controlA
  python experiments/exp007_context_vertex/eval_exp007.py --model context  --ckpt experiments/exp007_context_vertex/checkpoints/exp007/last.pth --tag exp007
"""
import os, sys, json, re, csv, time, argparse
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import DataLoader, SequentialSampler

from lib.networks.model_repository import Resnet18_8s
from lib.datasets.linemod_dataset import LineModDatasetRealAug, ImageSizeBatchSampler, VotingType
from lib.utils.data_utils import LineModImageDB, OcclusionLineModImageDB, LineModModelDB
from lib.utils.config import cfg
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v3
from lib.utils.evaluation_utils import pnp

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(EXP_DIR, 'results')
os.makedirs(RES_DIR, exist_ok=True)

N_IMG = 20
KP = 9
DIR_BAD_DEG = 10.0
PROX_BINS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, np.inf)]
EPS = 1e-9
K_LINEMOD = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
MODEL_SUB = 1000
TRAIN_CFG = json.load(open(os.path.join(ROOT, 'configs', 'linemod_train.json')))

MODELDB = LineModModelDB()
POINTS_3D = np.concatenate([MODELDB.get_farthest_3d('cat'),
                            MODELDB.get_centers_3d('cat')[None, :]], 0)
CORNERS_3D = MODELDB.get_corners_3d('cat')
DIAMETER = MODELDB.get_diameter('cat')
mesh, _ = MODELDB.get_ply_mesh('cat')
MESH = mesh[np.random.RandomState(0).choice(len(mesh), MODEL_SUB, replace=False)]

AMODAL_DIR = {'linemod': os.path.join(cfg.LINEMOD, 'cat', 'amodal_mask'),
              'occ': os.path.join(cfg.OCCLUSION_LINEMOD, 'amodal_masks', 'cat')}
REF_IDS = json.load(open(os.path.join(EXP_DIR, 'exp005_image_ids_reference.json')))
REF_RGB = {s: [r['rgb'] for r in REF_IDS['image_ids'][s]] for s in ('linemod', 'occ')}


def build_model(model_type, ckpt):
    if model_type == 'baseline':
        net = Resnet18_8s(ver_dim=KP * 2, seg_dim=2)
        sd = torch.load(ckpt, map_location='cpu')['net']
        net.load_state_dict(sd)
    else:
        from context_model import Resnet18_8sContext, load_from_baseline
        net = Resnet18_8sContext(ver_dim=KP * 2, seg_dim=2)
        sd = torch.load(ckpt, map_location='cpu')['net']
        if any(k.startswith('base.') for k in sd):
            net.load_state_dict(sd)          # already a context checkpoint
        else:
            load_from_baseline(net, sd)      # raw 199.pth into context model
    return net.cuda().eval()


def region_labels(vis_mask, amodal_mask):
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


def pose_metrics(pose_pred, pose_gt):
    R, t = pose_pred[:, :3], pose_pred[:, 3]
    Rg, tg = pose_gt[:, :3], pose_gt[:, 3]
    m_pred = MESH @ R.T + t; m_gt = MESH @ Rg.T + tg
    add = float(np.linalg.norm(m_pred - m_gt, axis=1).mean())
    d2 = ((m_pred[:, None, :] - m_gt[None, :, :]) ** 2).sum(-1)
    adds = float(np.sqrt(d2.min(1)).mean())
    pp = (CORNERS_3D @ R.T + t) @ K_LINEMOD.T; pp = pp[:, :2] / pp[:, 2:3]
    pg = (CORNERS_3D @ Rg.T + tg) @ K_LINEMOD.T; pg = pg[:, :2] / pg[:, 2:3]
    proj = float(np.linalg.norm(pp - pg, axis=1).mean())
    return add, adds, proj


REGIONS = ('all', 'interior', 'normal_boundary', 'occlusion_boundary')
RMAP = {'interior': 'interior', 'normal_boundary': 'boundary',
        'occlusion_boundary': 'occl_boundary'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['baseline', 'context'], required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    t0 = time.time()
    net = build_model(args.model, args.ckpt)
    aug_cfg = TRAIN_CFG['aug_cfg']

    pose_rows, dir_rows, endp_rows, kp_rows, prox_rows = [], [], [], [], []
    # accumulators per split
    acc = {s: {m: {r: [] for r in REGIONS} for m in
               ('dir', 'endp_unit', 'e_vote')}
           for s in ('linemod', 'occ')}
    kp_acc = {s: {k: {'dir': [], 'endp': []} for k in range(KP)} for s in ('linemod', 'occ')}
    prox_acc = {b: {'dir': [], 'e_vote': [], 'endp': [], 'imgs': set()} for b in range(len(PROX_BINS))}

    for split in ('linemod', 'occ'):
        if split == 'linemod':
            db = LineModImageDB('cat', has_render_set=False, has_fuse_set=False)
            db_set, prefix = db.val_real_set[:N_IMG], cfg.LINEMOD
        else:
            db = OcclusionLineModImageDB('cat')
            db_set = db.test_real_set[:len(db.test_real_set) // 2][:N_IMG]
            prefix = cfg.OCCLUSION_LINEMOD
        ds = LineModDatasetRealAug(db_set, prefix, VotingType.Farthest,
                                   augment=False, cfg=aug_cfg)
        for i in range(N_IMG):
            rgb = os.path.basename(ds.imagedb[i]['rgb_pth'])
            assert rgb == REF_RGB[split][i], f'image mismatch {split}[{i}]: {rgb}'
        loader = DataLoader(ds, batch_sampler=ImageSizeBatchSampler(
            SequentialSampler(ds), 1, False, aug_cfg), num_workers=0)

        with torch.no_grad():
            for image_id, data in enumerate(loader):
                image, mask_gt, vertex_gt, vw, pose, gt_kps = [d.cuda() for d in data]
                seg_pred, vertex_pred = net(image)
                b, _, h, w = vertex_pred.shape
                ver = vertex_pred.permute(0, 2, 3, 1).contiguous().view(b, h, w, -1, 2)[0]
                gt2 = gt_kps[0, :, :2] if gt_kps.dim() == 3 else gt_kps[0, 0, :, :2]
                pose_gt = pose[0].cpu().numpy()
                vis = (mask_gt[0].cpu().numpy() == 1)

                # ---- pose: predicted mask + baseline RANSAC + baseline pnp ----
                mask_pred = torch.argmax(seg_pred, 1)
                ver_t = vertex_pred.permute(0, 2, 3, 1).view(b, h, w, KP, 2)
                kp_pred = ransac_voting_layer_v3(mask_pred, ver_t, 128,
                                                 inlier_thresh=0.99, max_num=100)[0]
                kp_pred = kp_pred.cpu().numpy()
                if np.isfinite(kp_pred).all():
                    pose_pred = pnp(POINTS_3D, kp_pred, K_LINEMOD)
                    add, adds, proj = pose_metrics(pose_pred, pose_gt)
                else:
                    add = adds = proj = -1.0
                pose_rows.append([args.tag, split, image_id,
                                  f'{add:.4f}', f'{adds:.4f}', f'{proj:.2f}',
                                  adds < 0.1 * DIAMETER, add < 0.1 * DIAMETER, proj < 5.0])

                # ---- vertex diagnosis on GT visible pool (EXP006 definitions) ----
                rgb_pth = ds.imagedb[image_id]['rgb_pth']
                mid = re.findall(r'(\d+)', os.path.basename(rgb_pth))[0]
                apth = os.path.join(AMODAL_DIR[split], str(int(mid)) + '.png')
                amodal = (np.asarray(Image.open(apth)) > 0) if os.path.exists(apth) else None
                regions = region_labels(vis, amodal)

                ys, xs = np.nonzero(vis)
                coords = np.stack([xs, ys], 1).astype(np.float64)
                dirs = ver[ys, xs].cpu().numpy().astype(np.float64)
                gt_np = gt2.cpu().numpy()
                v_raw = gt_np[None, :, :] - coords[:, None, :]
                d_raw = np.linalg.norm(v_raw, axis=2)
                ok = d_raw >= 1.0
                m_pred = np.linalg.norm(dirs, axis=2)
                u_pred = dirs / (m_pred[..., None] + EPS)
                u_gt = v_raw / (d_raw[..., None] + EPS)
                ang = np.degrees(np.arccos(np.clip((u_pred * u_gt).sum(-1), -1, 1)))
                e_vote = np.stack([line_err(u_pred[:, k], coords, gt_np[k]) for k in range(KP)], 1)
                endp_unit = np.linalg.norm(dirs - u_gt, axis=2)

                st = acc[split]
                selmap = {r: regions[RMAP[r]][ys, xs][:, None] & ok
                          for r in REGIONS if r in RMAP}
                selmap['all'] = ok
                for r, sel in selmap.items():
                    if sel.any():
                        st['dir'][r].append(ang[sel].ravel())
                        st['endp_unit'][r].append(endp_unit[sel].ravel())
                        st['e_vote'][r].append(e_vote[sel].ravel())
                for k in range(KP):
                    kp_acc[split][k]['dir'].append(ang[ok[:, k], k])
                    kp_acc[split][k]['endp'].append(endp_unit[ok[:, k], k])

                if split == 'occ' and regions['occluded_unvotable'] is not None \
                        and regions['occluded_unvotable'].any():
                    dist_occ = regions['dist_occ'][ys, xs]
                    for bi, (lo, hi) in enumerate(PROX_BINS):
                        sel = ok & (dist_occ >= lo)[:, None] & (dist_occ < hi)[:, None]
                        if sel.any():
                            prox_acc[bi]['dir'].append(ang[sel].ravel())
                            prox_acc[bi]['e_vote'].append(e_vote[sel].ravel())
                            prox_acc[bi]['endp'].append(endp_unit[sel].ravel())
                            prox_acc[bi]['imgs'].add(image_id)

    def w(s, m, r, arrs):
        a = np.concatenate(arrs) if arrs else np.array([])
        return [s, m, r, a.size,
                f'{a.mean():.4f}' if a.size else '', f'{np.median(a):.4f}' if a.size else '',
                f'{np.quantile(a, .9):.4f}' if a.size else '',
                f'{(a > DIR_BAD_DEG).mean():.4f}' if a.size else '']

    with open(os.path.join(RES_DIR, f'direction_error_{args.tag}.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['tag', 'split', 'region', 'count', 'mean_deg', 'median_deg',
                     'p90_deg', 'bad_rate_gt10deg'])
        for s in ('linemod', 'occ'):
            for r in REGIONS:
                wr.writerow(w(args.tag, s, r, acc[s]['dir'][r]))
    with open(os.path.join(RES_DIR, f'endpoint_error_{args.tag}.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['tag', 'split', 'region', 'count', 'mean_px', 'median_px', 'p90_px'])
        for s in ('linemod', 'occ'):
            for r in REGIONS:
                a = np.concatenate(acc[s]['endp_unit'][r]) if acc[s]['endp_unit'][r] else np.array([])
                wr.writerow([args.tag, s, r, a.size,
                             f'{a.mean():.4f}' if a.size else '',
                             f'{np.median(a):.4f}' if a.size else '',
                             f'{np.quantile(a, .9):.4f}' if a.size else ''])
    with open(os.path.join(RES_DIR, f'keypoint_error_{args.tag}.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['tag', 'split', 'kp', 'count', 'dir_mean_deg', 'dir_p90_deg',
                     'dir_bad_rate', 'endp_mean_px', 'endp_p90_px'])
        for s in ('linemod', 'occ'):
            for k in range(KP):
                d = np.concatenate(kp_acc[s][k]['dir'])
                e = np.concatenate(kp_acc[s][k]['endp'])
                wr.writerow([args.tag, s, f'kp{k}', d.size, f'{d.mean():.3f}',
                             f'{np.quantile(d, .9):.3f}', f'{(d > DIR_BAD_DEG).mean():.4f}',
                             f'{e.mean():.4f}', f'{np.quantile(e, .9):.4f}'])
    with open(os.path.join(RES_DIR, f'occlusion_proximity_{args.tag}.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['tag', 'dist_bin_px', 'n_images', 'n_votes', 'dir_mean_deg',
                     'dir_median_deg', 'e_vote_mean', 'endp_mean_px'])
        for bi, (lo, hi) in enumerate(PROX_BINS):
            pa = prox_acc[bi]
            if not pa['dir']:
                wr.writerow([args.tag, f'{lo}-{hi if np.isfinite(hi) else "inf"}', 0, 0] + [''] * 4)
                continue
            dg = np.concatenate(pa['dir']); ev = np.concatenate(pa['e_vote'])
            ep = np.concatenate(pa['endp'])
            wr.writerow([args.tag, f'{lo}-{hi if np.isfinite(hi) else "inf"}',
                         len(pa['imgs']), dg.size, f'{dg.mean():.3f}', f'{np.median(dg):.3f}',
                         f'{ev.mean():.3f}', f'{ep.mean():.4f}'])
    with open(os.path.join(RES_DIR, f'pose_metrics_{args.tag}.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['tag', 'split', 'img', 'ADD_m', 'ADDS_m', 'proj_px',
                     'adds_pass_0.1d', 'add_pass_0.1d', 'proj_pass_5px'])
        wr.writerows(pose_rows)

    print(f'[{args.tag}] pose summary:')
    for s in ('linemod', 'occ'):
        rs = [r for r in pose_rows if r[1] == s]
        print(f"  {s}: ADD-S mean {np.mean([float(r[4]) for r in rs])*1000:.1f}mm "
              f"pass@0.1d {np.mean([r[6] for r in rs]):.3f} "
              f"| ADD mean {np.mean([float(r[3]) for r in rs])*1000:.1f}mm pass {np.mean([r[7] for r in rs]):.3f} "
              f"| proj mean {np.mean([float(r[5]) for r in rs]):.2f}px pass5px {np.mean([r[8] for r in rs]):.3f}")
    print(f'[{args.tag}] done in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
