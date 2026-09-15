"""EXP016: PnP Stability Diagnosis.

Research question (EXP014/EXP015 follow-up): EXP015 proved that the EXP013 D1
decoder genuinely and stably improves OCC keypoint localization on held-out
images (median 3.150 -> 2.143 px; paired 12 better / 7 worse; non-catastrophic
common subset 2.877 -> 1.948), yet EXP014 showed pose performance does NOT
improve and the standard-PnP flip count increases (OCC D0 4/20 vs D1 14/20
flips >90 deg, from EXP014's stored standard-PnP outputs). The single
question here: WHY does better keypoint localization produce MORE PnP flips?

This is a diagnosis-only observation experiment. The FULL original pipeline is
executed and observed (image -> segmentation -> vertex -> original PVNet RANSAC
voting -> predicted keypoints -> original standard PnP -> pose), but NOTHING is
modified: no training, no D0/D1 change, no voting/RANSAC change, no PnP solver
or threshold change, no intrinsics/GT change, no data expansion.

Arms:
  D0: 199.pth original 1x1 vertex head.  D1: EXP013 final D1 decoder loaded
  from EXP014's persisted states (direction gate re-verified against EXP013
  recorded held-out direction numbers; NO re-fitting).

Causal isolation (asserted): D0 vs original net.forward pipeline identity
(seg/vertex/mask/keypoints, <1e-5); D0 vs D1 segmentation logits BITWISE
identical and argmax masks identical; identical voting RNG (seed 0 re-applied
before EVERY voting call); identical intrinsics/GT/evaluation.

Flip definition (reused from EXP015's reuse of EXP014, not re-invented):
rotation_error = acos((trace(R_pred @ R_gt^T) - 1) / 2);
flip = rotation_error > 90 deg (primary); >120/>150/>170 deg also recorded.

Reprojection error: the original Evaluator's mean 2D reprojection distance
(px) of the predicted pose. "Low reprojection" uses the project's OWN
official 2D-projection success threshold (< 5 px) -- no new threshold is
invented. Case 1 (reprojection ambiguity) = flip(>90deg) AND reprojection < 5px.

Voting catastrophic (EXP015 definition, reused): image max keypoint error
> 100 px.

Pre-registered decision rules (fixed BEFORE the run):
  reproduction_ok := OCC flip(>90deg) counts reproduce EXP015's stored
                     reference exactly (D0 4/20, D1 14/20).
  GO-B (PnP stability) iff reproduction_ok AND D1 majority KP improved AND
       D1 non-catastrophic flip rate >= 25% AND at least one D1 flip survives
       excluding catastrophic images AND >= 30% of D1 flipped non-cat poses
       have reprojection < 5 px (keypoint accuracy -> pose stability mismatch).
  GO-A (voting robustness) iff D1 non-cat flip rate < 25% AND D1 majority KP
       improved (flips mostly disappear without catastrophic images).
  STOP otherwise (flip difference not reproducible, or all flips explained
       by catastrophic voting and no reprojection ambiguity).
Statistics: counts/percentages/paired differences only (20 images/split);
optional simple Spearman rank correlation computed manually (no p-values).

Run from repo root:
  python experiments/exp016_pnp_stability_diagnosis/run_exp016.py
"""
import os, sys, json, csv, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler

from lib.networks.model_repository import Resnet18_8s
from lib.datasets.linemod_dataset import LineModDatasetRealAug, VotingType
from lib.utils.data_utils import LineModImageDB, OcclusionLineModImageDB
from lib.utils.config import cfg
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v3
from lib.utils.evaluation_utils import Evaluator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(EXP_DIR, 'results')
FIG_DIR = os.path.join(RES_DIR, 'figures')
RAW_DIR = os.path.join(RES_DIR, 'raw_predictions')
LOG_DIR = os.path.join(EXP_DIR, 'logs')
for d in (RES_DIR, FIG_DIR, RAW_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)
EXP13_DIR = os.path.join(ROOT, 'experiments', 'exp013_heldout_decoder_generalization')
EXP14_DIR = os.path.join(ROOT, 'experiments', 'exp014_end_to_end_pose_validation')
EXP15_DIR = os.path.join(ROOT, 'experiments', 'exp015_keypoint_localization_diagnosis')

N_IMG = 20
KP = 9
SEED = 0
EPS = 1e-9
VOTE_TYPE = VotingType.Farthest          # train_linemod_exp001.py verbatim
TRAIN_CFG = json.load(open(os.path.join(ROOT, 'configs',
                                        'exp001_reliability_voting_scratch.json')))
ROUND_HYP, MAX_NUM = TRAIN_CFG['vote_round_hyp_num'], TRAIN_CFG['vote_max_num']
INLIER_THRESH = 0.99                     # EvalWrapperBaseline verbatim
assert (ROUND_HYP, MAX_NUM) == (128, 100), 'vote params changed -> STOP'
FLIP_THRESH = (90.0, 120.0, 150.0, 170.0)
CAT_IMG = 100.0                          # voting catastrophic: max kp error > 100 px (EXP015 def)
TIE_EPS = 1e-6
REPROJ_LOW = 5.0                         # project's official 2D-projection success threshold (px)
GOB_FLIP_RATE = 0.25                     # 'high flip rate' / 'mostly disappear' boundary
GOB_LOWREPROJ_FRAC = 0.30                # min fraction of D1 flipped non-cat poses with reproj<5px

# EXP015-stored flip reference (from EXP014 standard-PNP outputs, >90 deg)
FLIP_REF = {('linemod', 'D0'): 0, ('linemod', 'D1'): 0,
            ('occ', 'D0'): 4, ('occ', 'D1'): 14}

IDS15 = json.load(open(os.path.join(EXP15_DIR, 'results', 'image_ids.json')))
IDS13 = json.load(open(os.path.join(EXP13_DIR, 'results', 'image_ids.json')))
for s in ('linemod', 'occ'):
    assert IDS15['heldout'][s]['rgb'] == IDS13['heldout'][s]['rgb'], \
        f'EXP015/EXP013 image IDs differ for {s} -> STOP'
    assert set(IDS15['fit'][s]['rgb']).isdisjoint(set(IDS15['heldout'][s]['rgb']))
HELD_RGB = {s: IDS15['heldout'][s]['rgb'] for s in ('linemod', 'occ')}

EXP13_SUM = json.load(open(os.path.join(EXP13_DIR, 'results', 'final_summary.json')))
DIR_REF = {s: {'D0': EXP13_SUM['d0_heldout'][s if s != 'linemod' else 'lin'],
               'D1': EXP13_SUM['mean']['heldout'][s]['D1']}
           for s in ('linemod', 'occ')}

D1_STATES = {s: os.path.join(EXP14_DIR, 'results', f'd1_decoder_{s}.pth')
             for s in ('linemod', 'occ')}
for s in ('linemod', 'occ'):
    assert os.path.exists(D1_STATES[s]), \
        f'D1 state missing: {D1_STATES[s]} -> STOP (re-fitting is forbidden)'

torch.manual_seed(SEED); np.random.seed(SEED)
T0 = time.time()
LOG = open(os.path.join(LOG_DIR, 'run.log'), 'w', buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + '\n')

log('HELD-OUT image IDs == EXP015 == EXP014 == EXP013 (asserted):')
log(f'LIN image IDs: {[os.path.splitext(r)[0] for r in HELD_RGB["linemod"]]}')
log(f'OCC image IDs: {[os.path.splitext(r)[0] for r in HELD_RGB["occ"]]}')

json.dump(dict(
    cls='cat', n_img=N_IMG, ckpt='data/model/cat_linemod_train/199.pth',
    variable='vertex prediction ONLY: D0 original convraw[3] 1x1 head vs EXP013 '
             'final D1 decoder loaded from EXP014 persisted states; NO re-fitting; '
             'full original pipeline observed (voting AND original standard PnP), '
             'nothing modified',
    frozen='199.pth entire PVNet frozen; D1 verified against EXP013 recorded '
           'held-out direction numbers (tol 0.01 deg) before use',
    data='EXP015 results/image_ids.json heldout (asserted == EXP013 == EXP014 '
         'usage): 20 LIN val_real_set[20:40], 20 OCC test_real_set[20:40]; '
         'FIT-excluded asserted',
    voting='original ransac_voting_layer_v3 (round_hyp_num=128, inlier_thresh=0.99, '
           f'max_num=100, asserted); seed {SEED} re-applied before every call; '
           'NO modification; internal vote-count statistics NOT available without '
           'modifying voting -> not recorded (per spec)',
    pnp='original Evaluator standard PnP path (pnp, SOLVEPNP_ITERATIVE, linemod '
        'intrinsics, Farthest 3D points) EXACTLY as EXP014; NO solver/threshold '
        'change; rotation error via trace formula; flip = rot_err > 90 deg '
        '(>120/>150/>170 also recorded); reprojection = Evaluator mean 2D '
        'reprojection px; "low reprojection" = < 5 px (project official 2D '
        'threshold)',
    definitions=dict(flip_deg=list(FLIP_THRESH),
                     voting_catastrophic=f'max kp error > {CAT_IMG:.0f} px (EXP015 def)',
                     reproj_low_px=REPROJ_LOW, tie_eps=TIE_EPS),
    decision_rules='Pre-registered: reproduction_ok = OCC >90deg flips reproduce '
                   'EXP015 reference (D0 4/20, D1 14/20); GO-B iff reproduction_ok '
                   'AND D1 majority KP improved AND D1 non-cat flip rate >= 25% AND '
                   '>=1 D1 flip survives excluding catastrophic AND >=30% of D1 '
                   'flipped non-cat poses have reprojection < 5px; GO-A iff D1 '
                   'non-cat flip rate < 25% AND D1 majority KP improved; STOP '
                   'otherwise',
    reference_flip_counts={f'{s}_{n}': v for (s, n), v in FLIP_REF.items()}),
    open(os.path.join(RES_DIR, 'config.json'), 'w'), indent=2)

device = 'cuda'
net = Resnet18_8s(ver_dim=KP * 2, seg_dim=2)
sd = torch.load(os.path.join(ROOT, 'data/model/cat_linemod_train/199.pth'), map_location='cpu')
net.load_state_dict(sd['net'])
net = net.to(device).eval()
for p in net.parameters():
    p.requires_grad_(False)

AUG_CFG = TRAIN_CFG['aug_cfg']


def normalize_vertices(t):
    b = t.view(t.shape[0], KP, 2, *t.shape[2:])
    n = torch.clamp(b.pow(2).sum(2, keepdim=True).sqrt(), min=EPS)
    return (b / n).view(t.shape)


def build_loader(split, lo, hi, expect):
    if split == 'linemod':
        db = LineModImageDB('cat', has_render_set=False, has_fuse_set=False)
        db_set, prefix = db.val_real_set[lo:hi], cfg.LINEMOD
    else:
        db = OcclusionLineModImageDB('cat')
        half = db.test_real_set[:len(db.test_real_set) // 2]
        db_set, prefix = half[lo:hi], cfg.OCCLUSION_LINEMOD
    ds = LineModDatasetRealAug(db_set, prefix, VOTE_TYPE, augment=False, cfg=AUG_CFG)
    ids = []
    for i in range(len(ds)):
        rgb = os.path.basename(ds.imagedb[i]['rgb_pth'])
        ids.append(rgb)
        assert rgb == expect[i], f'image mismatch {split}[{lo+i}]: {rgb}'
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'tle', os.path.join(ROOT, 'tools', 'train_linemod_exp001.py'))
    tle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tle)
    loader = DataLoader(ds, batch_sampler=tle.ImageSizeBatchSampler(
        SequentialSampler(ds), 1, False, AUG_CFG), num_workers=6)
    return ds, loader


def trunk_feat(image):
    """Manual frozen trunk forward -> shared 32-ch feature F (EXP014 verbatim)."""
    with torch.no_grad():
        x2s, x4s, x8s, x16s, x32s, xfc = net.resnet18_8s(image)
        fm = net.conv8s(torch.cat([xfc, x8s], 1)); fm = net.up8sto4s(fm)
        fm = net.conv4s(torch.cat([fm, x4s], 1)); fm = net.up4sto2s(fm)
        fm = net.conv2s(torch.cat([fm, x2s], 1)); fm = net.up2storaw(fm)
        return net.convraw[0:3](torch.cat([fm, image], 1))   # (1,32,H,W)


def cache_split(split, ds, loader, id_offset):
    items = []
    for i, data in enumerate(loader):
        image, mask_gt, vertex_gt, vw, pose, gt_kps = [d.to(device) for d in data]
        assert vertex_gt.shape[1] == KP * 2 and vw.shape[1] == 1
        feat = trunk_feat(image)
        gt2 = (gt_kps[0, :, :2] if gt_kps.dim() == 3 else gt_kps[0, 0, :, :2]).cpu().numpy()
        items.append(dict(image_id=id_offset + i, image=image, feat=feat,
                          gt_kps=gt2, pose=pose[0].cpu().numpy(),
                          vis=(mask_gt[0].cpu().numpy() == 1)))
    return items


CACHE = {}
for split in ('linemod', 'occ'):
    ds, loader = build_loader(split, N_IMG, 2 * N_IMG, expect=HELD_RGB[split])
    CACHE[split] = cache_split(split, ds, loader, N_IMG)
    log(f'--- {split}: {len(CACHE[split])} HELD-OUT images cached, IDs asserted '
        f'({time.time()-T0:.1f}s) ---')


# ---------------- identity checks (before any evaluation) ----------------
def d0_outputs(feat):
    with torch.no_grad():
        out = net.convraw[3](feat)
        return out[:, :net.seg_dim], out[:, net.seg_dim:]


def vote_keypoints(seg, ver):
    """EvalWrapperBaseline verbatim, seed re-applied per call (identical RNG
    stream for both branches)."""
    vertex = ver.permute(0, 2, 3, 1)
    b, h, w, vn_2 = vertex.shape
    vertex = vertex.view(b, h, w, vn_2 // 2, 2)
    mask = torch.argmax(seg, 1)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    return ransac_voting_layer_v3(mask, vertex, ROUND_HYP,
                                  inlier_thresh=INLIER_THRESH, max_num=MAX_NUM)


max_seg = max_ver = 0.0
mask_all_eq = True
max_kp = 0.0
for split in ('linemod', 'occ'):
    for item in CACHE[split]:
        with torch.no_grad():
            seg_f, ver_f = net(item['image'])                 # original PVNet outputs
        seg0, ver0 = d0_outputs(item['feat'])                 # D0 branch outputs
        max_seg = max(max_seg, float((seg_f - seg0).abs().max()))
        max_ver = max(max_ver, float((ver_f - ver0).abs().max()))
        mask_all_eq &= bool((torch.argmax(seg_f, 1) == torch.argmax(seg0, 1)).all())
        kp_f = vote_keypoints(seg_f, ver_f)
        kp_0 = vote_keypoints(seg0, ver0)
        max_kp = max(max_kp, float((kp_f - kp_0).abs().max()))
log(f'identity check (D0 vs original pipeline): max|seg| {max_seg:.2e}  '
    f'max|vertex| {max_ver:.2e}  mask identical {mask_all_eq}  max|kp| {max_kp:.2e}')
assert max_seg < 1e-5 and max_ver < 1e-5 and mask_all_eq and max_kp < 1e-5, \
    'D0 branch does not reproduce original PVNet -> STOP'
log('identity check (a) PASSED')


# ---------------- D1 load + direction gate ----------------
class ResidualDecoder(nn.Module):
    """EXP012/013 final D1 verbatim: D0(F) + zero-init Conv3x3(32->32) ->
    LeakyReLU(0.1) -> Conv1x1(32->18). Segmentation untouched."""

    def __init__(self, cin=32, kp=KP):
        super().__init__()
        self.conv3 = nn.Conv2d(cin, cin, 3, padding=1, bias=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.c1 = nn.Conv2d(cin, kp * 2, 1, bias=True)
        nn.init.zeros_(self.c1.weight)
        nn.init.zeros_(self.c1.bias)

    def forward(self, f):
        return net.convraw[3](f)[:, net.seg_dim:] + self.c1(self.act(self.conv3(f)))


MODELS = {}
for s in ('linemod', 'occ'):
    dec = ResidualDecoder().to(device)
    dec.load_state_dict(torch.load(D1_STATES[s], map_location=device))
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    MODELS[s] = dec
    log(f'D1-{s} state loaded from {D1_STATES[s]} '
        f'({sum(p.numel() for p in dec.parameters())} params, frozen)')


def angle_deg(a, b):
    dot = np.clip((a * b).sum(-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def d0_decode_only(feat):
    with torch.no_grad():
        return net.convraw[3](feat)[:, net.seg_dim:]


DIR = {}
for split in ('linemod', 'occ'):
    res = {'D0': [], 'D1': []}
    for item in CACHE[split]:
        feat = item['feat']
        H, W = feat.shape[2], feat.shape[3]
        # EXP014/015 visible-fg pool geometry verbatim (GT seg mask, |K-p| >= 1px)
        ys, xs = np.nonzero(item['vis'])
        coords = np.stack([xs, ys], 1).astype(np.float64)
        gt2 = item['gt_kps']
        v_raw = gt2[None, :, :] - coords[:, None, :]
        d_raw = np.linalg.norm(v_raw, axis=2)
        ok = d_raw >= 1.0
        u_gt = v_raw / (d_raw[..., None] + EPS)
        for name in ('D0', 'D1'):
            with torch.no_grad():
                v = (d0_decode_only(feat) if name == 'D0' else MODELS[split](feat))
                v = normalize_vertices(v)
            ver = v.permute(0, 2, 3, 1).contiguous().view(1, H, W, KP, 2)[0]
            dirs = ver[ys, xs].cpu().numpy().astype(np.float64)
            u_pred = dirs / (np.linalg.norm(dirs, axis=2)[..., None] + EPS)
            res[name].append(angle_deg(u_pred, u_gt)[ok])
    DIR[split] = {n: float(np.concatenate(res[n]).mean()) for n in ('D0', 'D1')}
    log(f"direction gate [{split}]: D0 {DIR[split]['D0']:.4f} "
        f"(EXP013 {DIR_REF[split]['D0']:.4f})  D1 {DIR[split]['D1']:.4f} "
        f"(EXP013 {DIR_REF[split]['D1']:.4f})")
    assert abs(DIR[split]['D0'] - DIR_REF[split]['D0']) < 0.01 and \
           abs(DIR[split]['D1'] - DIR_REF[split]['D1']) < 0.01, \
        'loaded D1 state does not reproduce EXP013 held-out direction -> STOP'
log('direction gate PASSED (loaded D1 state == EXP013 final state)')


# ---------------- main loop: voting + ORIGINAL standard PnP (nothing modified) ----------------
EVALUATORS = {(s, n): Evaluator() for s in ('linemod', 'occ') for n in ('D0', 'D1')}
SEG_ID_OK = True
REC = {(s, n): [] for s in ('linemod', 'occ') for n in ('D0', 'D1')}   # per-image records
RAW = {(s, n): dict(pred_kps=[], pred_pose=[], gt_kps=[], gt_pose=[])
       for s in ('linemod', 'occ') for n in ('D0', 'D1')}

for split in ('linemod', 'occ'):
    for item in CACHE[split]:
        seg0, ver0 = d0_outputs(item['feat'])
        with torch.no_grad():
            ver1 = MODELS[split](item['feat'])
            seg1 = net.convraw[3](item['feat'])[:, :net.seg_dim]
        seg_bit_eq = bool((seg1 == seg0).all())
        mask_eq = bool((torch.argmax(seg1, 1) == torch.argmax(seg0, 1)).all())
        SEG_ID_OK &= seg_bit_eq and mask_eq
        assert seg_bit_eq and mask_eq, 'D1 branch changed segmentation -> STOP'
        kp = {'D0': vote_keypoints(seg0, ver0), 'D1': vote_keypoints(seg1, ver1)}
        for n in ('D0', 'D1'):
            kpn = kp[n][0].cpu().numpy().astype(np.float64)          # (9,2) voted kps
            err = np.linalg.norm(kpn - item['gt_kps'], axis=1)       # (9,) px
            fin = err[np.isfinite(err)]
            ev = EVALUATORS[(split, n)]
            pnp_ok, pose_pred = True, None
            try:
                pose_pred = ev.evaluate(kpn, item['pose'], 'cat', 'linemod',
                                        VOTE_TYPE, intri_matrix=None)
            except Exception as e:                                    # record only
                pnp_ok = False
                log(f'PnP EXCEPTION [{split} {n} img{item["image_id"]}]: {e}')
            if pnp_ok:
                rot = float(np.rad2deg(np.arccos(np.clip((np.trace(
                    pose_pred[:, :3] @ item['pose'][:, :3].T) - 1) / 2, -1, 1))))
                trans = float(np.linalg.norm(pose_pred[:, 3] - item['pose'][:, 3]))
                add_mm = float(ev.add_dists[-1]) * 1000.0
                add_pass = bool(ev.add_recorder[-1])
                proj_px = float(ev.proj_mean_diffs[-1])
                proj_pass = bool(ev.projection_2d_recorder[-1])
                cm_pass = bool(ev.cm_degree_5_recorder[-1])
                RAW[(split, n)]['pred_pose'].append(pose_pred)
            else:
                rot = trans = add_mm = proj_px = float('nan')
                add_pass = proj_pass = cm_pass = False
                RAW[(split, n)]['pred_pose'].append(np.full((3, 4), np.nan))
            REC[(split, n)].append(dict(
                image_id=item['image_id'], rgb=HELD_RGB[split][item['image_id'] - N_IMG],
                kp_mean=float(fin.mean()) if fin.size else float('nan'),
                kp_median=float(np.median(fin)) if fin.size else float('nan'),
                kp_max=float(fin.max()) if fin.size else float('nan'),
                nan_kp=int(err.size - fin.size),
                kp_err=err, pnp_ok=pnp_ok, rot_err=rot, trans_err=trans,
                add_mm=add_mm, add_pass=add_pass, proj_px=proj_px,
                proj_pass=proj_pass, cm_pass=cm_pass))
            RAW[(split, n)]['pred_kps'].append(kpn)
            RAW[(split, n)]['gt_kps'].append(item['gt_kps'])
            RAW[(split, n)]['gt_pose'].append(item['pose'])
log(f'segmentation identity D0 vs D1 (logits bitwise + mask): {SEG_ID_OK}')
assert SEG_ID_OK, 'D0/D1 segmentation mismatch -> STOP'

# flip flags at all thresholds
for (s, n), rows in REC.items():
    for r in rows:
        for t in FLIP_THRESH:
            r[f'flip_gt{int(t)}'] = bool(r['pnp_ok'] and r['rot_err'] > t)
        r['voting_catastrophic'] = bool(r['nan_kp'] > 0 or r['kp_max'] > CAT_IMG)

# ---------------- raw predictions ----------------
for (s, n), d in RAW.items():
    np.savez_compressed(
        os.path.join(RAW_DIR, f'raw_{s}_{n}.npz'),
        pred_kps=np.stack(d['pred_kps']), pred_pose=np.stack(d['pred_pose']),
        gt_kps=np.stack(d['gt_kps']), gt_pose=np.stack(d['gt_pose']),
        image_ids=np.arange(N_IMG, 2 * N_IMG))
log(f'raw predictions saved to {RAW_DIR} (npz compressed)')


# ---------------- Analysis A: flip rates vs EXP015 reference ----------------
FLIP = {}
for s in ('linemod', 'occ'):
    for n in ('D0', 'D1'):
        FLIP[(s, n)] = {f'gt{int(t)}': int(sum(r[f'flip_gt{int(t)}'] for r in REC[(s, n)]))
                        for t in FLIP_THRESH}
for s in ('linemod', 'occ'):
    for n in ('D0', 'D1'):
        log(f"flip [{s} {n}]: " + '  '.join(f'>{int(t)}deg {FLIP[(s,n)][f"gt{int(t)}"]}/20'
                                            for t in FLIP_THRESH))
reproduction_ok = all(FLIP[(s, n)]['gt90'] == FLIP_REF[(s, n)]
                      for s in ('linemod', 'occ') for n in ('D0', 'D1'))
log(f'reproduction of EXP015-stored flip counts (D0 0/0, D1 0/0, OCC D0 4/20, '
    f'OCC D1 14/20): {reproduction_ok}')


# ---------------- Analysis B: keypoint improvement vs D1 flip ----------------
def cls_delta(d):
    return 'improved' if d < -TIE_EPS else ('worsened' if d > TIE_EPS else 'tie')


CONTAB = {}
for s in ('linemod', 'occ'):
    rows0, rows1 = REC[(s, 'D0')], REC[(s, 'D1')]
    cats = []
    for r0, r1 in zip(rows0, rows1):
        assert r0['image_id'] == r1['image_id']
        cats.append(dict(
            image_id=r0['image_id'],
            delta_kp_mean=r1['kp_mean'] - r0['kp_mean'],
            delta_kp_median=r1['kp_median'] - r0['kp_median'],
            cls_mean=cls_delta(r1['kp_mean'] - r0['kp_mean']),
            cls_median=cls_delta(r1['kp_median'] - r0['kp_median']),
            d1_flip=r1['flip_gt90'], d0_flip=r0['flip_gt90']))
    for c in cats:
        c['delta_kp_mean'] = float(c['delta_kp_mean']); c['delta_kp_median'] = float(c['delta_kp_median'])
    tabs = {}
    for basis in ('cls_mean', 'cls_median'):
        tab = {k: dict(stable=0, flip=0) for k in ('improved', 'worsened', 'tie')}
        for c in cats:
            tab[c[basis]]['flip' if c['d1_flip'] else 'stable'] += 1
        tabs[basis] = tab
    n_imp = tabs['cls_mean']['improved']['stable'] + tabs['cls_mean']['improved']['flip']
    n_wor = tabs['cls_mean']['worsened']['stable'] + tabs['cls_mean']['worsened']['flip']
    CONTAB[s] = dict(
        table=tabs,
        p_flip_given_improved=(tabs['cls_mean']['improved']['flip'] / n_imp) if n_imp else None,
        p_flip_given_worsened=(tabs['cls_mean']['worsened']['flip'] / n_wor) if n_wor else None,
        per_image=cats)
    log(f'contingency [{s}] (mean-based, D1 flip rows): '
        f"improved {tabs['cls_mean']['improved']['flip']}/{n_imp} flip  "
        f"worsened {tabs['cls_mean']['worsened']['flip']}/{n_wor} flip  "
        f"tie {tabs['cls_mean']['tie']['flip']}/{tabs['cls_mean']['tie']['stable'] + tabs['cls_mean']['tie']['flip']}")
    log(f"P(flip|KP improved) {CONTAB[s]['p_flip_given_improved']}  "
        f"P(flip|KP worsened) {CONTAB[s]['p_flip_given_worsened']}")


# ---------------- Analysis C: KP error by flip group ----------------
GRP = {}
for s in ('linemod', 'occ'):
    for n in ('D0', 'D1'):
        g = {'flip': [], 'stable': []}
        for r in REC[(s, n)]:
            g['flip' if r['flip_gt90'] else 'stable'].append(r)
        GRP[f'{s}_{n}'] = {}
        for k, rows in g.items():
            if rows:
                GRP[f'{s}_{n}'][k] = dict(
                    n=len(rows),
                    median_of_kp_median=float(np.median([r['kp_median'] for r in rows])),
                    median_of_kp_mean=float(np.median([r['kp_mean'] for r in rows])),
                    median_of_kp_max=float(np.median([r['kp_max'] for r in rows])),
                    median_rot_err=float(np.nanmedian([r['rot_err'] for r in rows])),
                    median_reproj_px=float(np.nanmedian([r['proj_px'] for r in rows])))
            else:
                GRP[f'{s}_{n}'][k] = dict(n=0)
        log(f"KP-by-flip [{s} {n}]: flip n={GRP[f'{s}_{n}']['flip']['n']} "
            f"median kp_median {GRP[f'{s}_{n}']['flip'].get('median_of_kp_median', float('nan')):.3f}px | "
            f"stable n={GRP[f'{s}_{n}']['stable']['n']} "
            f"median kp_median {GRP[f'{s}_{n}']['stable'].get('median_of_kp_median', float('nan')):.3f}px")


# ---------------- Analysis D/E: reprojection ambiguity ----------------
AMBIG = {}
for s in ('linemod', 'occ'):
    for n in ('D0', 'D1'):
        rows = REC[(s, n)]
        flips = [r for r in rows if r['flip_gt90']]
        stabs = [r for r in rows if not r['flip_gt90']]
        def rep(rows):
            v = np.array([r['proj_px'] for r in rows], np.float64)
            v = v[np.isfinite(v)]
            return dict(mean=float(v.mean()), median=float(np.median(v)),
                        p90=float(np.quantile(v, 0.9))) if v.size else dict(mean=None, median=None, p90=None)
        low_flip = [r for r in flips if r['proj_px'] < REPROJ_LOW]
        AMBIG[f'{s}_{n}'] = dict(
            reproj_all=rep(rows), reproj_flip=rep(flips), reproj_stable=rep(stabs),
            n_flip=len(flips), n_flip_low_reproj=len(low_flip),
            frac_flip_low_reproj=(len(low_flip) / len(flips)) if flips else None,
            flip_low_reproj_image_ids=[r['image_id'] for r in low_flip])
        log(f"ambiguity [{s} {n}]: flips {len(flips)}, low-reproj(<{REPROJ_LOW:.0f}px) flips "
            f"{len(low_flip)} ({AMBIG[f'{s}_{n}']['frac_flip_low_reproj']})")


# ---------------- Analysis F: voting catastrophic vs flip separation ----------------
CATX = {}
for s in ('linemod', 'occ'):
    for n in ('D0', 'D1'):
        rows = REC[(s, n)]
        cat = [r for r in rows if r['voting_catastrophic']]
        ncat = [r for r in rows if not r['voting_catastrophic']]
        CATX[f'{s}_{n}'] = dict(
            n_cat=len(cat), n_ncat=len(ncat),
            cat_ids=[r['image_id'] for r in cat],
            flip_among_cat=int(sum(r['flip_gt90'] for r in cat)),
            flip_among_ncat=int(sum(r['flip_gt90'] for r in ncat)),
            ncat_flip_rate=(sum(r['flip_gt90'] for r in ncat) / len(ncat)) if ncat else None,
            cat_and_flip=[r['image_id'] for r in cat if r['flip_gt90']],
            ncat_flip_ids=[r['image_id'] for r in ncat if r['flip_gt90']])
        log(f"catx [{s} {n}]: cat {len(cat)}/20 (flip {CATX[f'{s}_{n}']['flip_among_cat']}), "
            f"non-cat flip {CATX[f'{s}_{n}']['flip_among_ncat']}/{len(ncat)} "
            f"({CATX[f'{s}_{n}']['ncat_flip_rate']:.2%}) ids {CATX[f'{s}_{n}']['ncat_flip_ids']}")


# ---------------- non-catastrophic OCC comparison (common subset) ----------------
NC_OCC = {}
rows0, rows1 = REC[('occ', 'D0')], REC[('occ', 'D1')]
com = [i for i in range(N_IMG)
       if not rows0[i]['voting_catastrophic'] and not rows1[i]['voting_catastrophic']]
for n in ('D0', 'D1'):
    rows = [REC[('occ', n)][i] for i in com]
    NC_OCC[n] = dict(
        n=len(com),
        kp_median_median=float(np.median([r['kp_median'] for r in rows])),
        kp_mean_median=float(np.median([r['kp_mean'] for r in rows])),
        rot_err_median=float(np.nanmedian([r['rot_err'] for r in rows])),
        flip_rate=float(sum(r['flip_gt90'] for r in rows) / len(rows)),
        reproj_median_px=float(np.nanmedian([r['proj_px'] for r in rows])),
        add_pass_frac=float(np.mean([r['add_pass'] for r in rows])),
        add_mm_median=float(np.nanmedian([r['add_mm'] for r in rows])))
    log(f"non-cat OCC [{n}] n={len(com)}: kp_median {NC_OCC[n]['kp_median_median']:.3f}px  "
        f"rot_med {NC_OCC[n]['rot_err_median']:.2f}deg  flip {NC_OCC[n]['flip_rate']:.2%}  "
        f"reproj_med {NC_OCC[n]['reproj_median_px']:.2f}px  ADD pass {NC_OCC[n]['add_pass_frac']:.2f}")


# ---------------- simple Spearman (manual, no p-values) ----------------
def spearman(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


occ_d1 = REC[('occ', 'D1')]; occ_d0 = REC[('occ', 'D0')]
CORR = dict(
    occ_d1_improvement_vs_rot=spearman([r0['kp_mean'] - r1['kp_mean']
                                        for r0, r1 in zip(occ_d0, occ_d1)],
                                       [r['rot_err'] for r in occ_d1]),
    occ_d1_kpmean_vs_rot=spearman([r['kp_mean'] for r in occ_d1],
                                  [r['rot_err'] for r in occ_d1]),
    occ_d1_reproj_vs_rot=spearman([r['proj_px'] for r in occ_d1],
                                  [r['rot_err'] for r in occ_d1]))
log(f"Spearman (OCC): D1 improvement vs rot {CORR['occ_d1_improvement_vs_rot']}  "
    f"D1 kp_mean vs rot {CORR['occ_d1_kpmean_vs_rot']}  "
    f"D1 reproj vs rot {CORR['occ_d1_reproj_vs_rot']}")


# ---------------- decision (pre-registered) ----------------
tabm = CONTAB['occ']['table']['cls_mean']
d1_imp_n = tabm['improved']['stable'] + tabm['improved']['flip']
d1_wor_n = tabm['worsened']['stable'] + tabm['worsened']['flip']
majority_improved = d1_imp_n > d1_wor_n
d1_ncat_rate = CATX['occ_D1']['ncat_flip_rate']
d1_ncat_flips = CATX['occ_D1']['flip_among_ncat']
amb = AMBIG['occ_D1']
lowreproj_frac = amb['frac_flip_low_reproj'] or 0.0
if not reproduction_ok:
    decision = 'STOP'
elif majority_improved and d1_ncat_rate is not None and d1_ncat_rate >= GOB_FLIP_RATE \
        and d1_ncat_flips >= 1 and lowreproj_frac >= GOB_LOWREPROJ_FRAC:
    decision = 'GO-B'
elif majority_improved and d1_ncat_rate is not None and d1_ncat_rate < GOB_FLIP_RATE:
    decision = 'GO-A'
else:
    decision = 'STOP'
log(f"DECISION: {decision} (reproduction_ok {reproduction_ok}, majority_improved "
    f"{majority_improved}, D1 non-cat flip rate {d1_ncat_rate}, low-reproj flip frac "
    f"{lowreproj_frac})")


# ---------------- per_image.csv ----------------
with open(os.path.join(RES_DIR, 'per_image.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['dataset', 'image_id', 'rgb',
                 'D0_kp_mean', 'D1_kp_mean', 'D0_kp_median', 'D1_kp_median',
                 'D0_kp_max', 'D1_kp_max',
                 'D0_rotation_error', 'D1_rotation_error',
                 'D0_translation_error', 'D1_translation_error',
                 'D0_reprojection_error', 'D1_reprojection_error',
                 'D0_flip', 'D1_flip',
                 'D0_voting_catastrophic', 'D1_voting_catastrophic',
                 'delta_kp_mean', 'delta_kp_median'])
    for i in range(N_IMG):
        for s in ('linemod', 'occ'):
            r0, r1 = REC[(s, 'D0')][i], REC[(s, 'D1')][i]
            wr.writerow([s, r0['image_id'], r0['rgb'],
                         f"{r0['kp_mean']:.4f}", f"{r1['kp_mean']:.4f}",
                         f"{r0['kp_median']:.4f}", f"{r1['kp_median']:.4f}",
                         f"{r0['kp_max']:.2f}", f"{r1['kp_max']:.2f}",
                         f"{r0['rot_err']:.3f}" if r0['pnp_ok'] else 'nan',
                         f"{r1['rot_err']:.3f}" if r1['pnp_ok'] else 'nan',
                         f"{r0['trans_err'] * 1000:.2f}" if r0['pnp_ok'] else 'nan',
                         f"{r1['trans_err'] * 1000:.2f}" if r1['pnp_ok'] else 'nan',
                         f"{r0['proj_px']:.3f}" if r0['pnp_ok'] else 'nan',
                         f"{r1['proj_px']:.3f}" if r1['pnp_ok'] else 'nan',
                         int(r0['flip_gt90']), int(r1['flip_gt90']),
                         int(r0['voting_catastrophic']), int(r1['voting_catastrophic']),
                         f"{r1['kp_mean'] - r0['kp_mean']:.4f}",
                         f"{r1['kp_median'] - r0['kp_median']:.4f}"])


# ---------------- figures ----------------
occ_x = np.arange(N_IMG, N_IMG * 2)
# Fig 1: D0 KP median vs D1 KP median (OCC, per image)
fig, ax = plt.subplots(figsize=(5.2, 5.2))
m_cat = np.array([r['voting_catastrophic'] for r in rows0])
ax.scatter([r['kp_median'] for r in occ_d0], [r['kp_median'] for r in occ_d1],
           c=np.where(m_cat, 'red', 'tab:blue'), s=42,
           edgecolors=np.where(m_cat, 'darkred', 'navy'), zorder=3)
lim = max(max(r['kp_median'] for r in occ_d0), max(r['kp_median'] for r in occ_d1)) * 1.08
ax.plot([0, lim], [0, lim], 'k--', lw=1, label='y = x')
ax.set_xlabel('D0 KP median error (px)'); ax.set_ylabel('D1 KP median error (px)')
ax.set_title('OCC per-image: D0 vs D1 keypoint median error\n(red = voting-catastrophic)')
ax.legend(); ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect('equal')
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'fig1_occ_d0_vs_d1_kp_median.png'), dpi=150); plt.close(fig)

# Fig 2: D1 KP improvement vs D1 rotation error
imp = np.array([r0['kp_mean'] - r1['kp_mean'] for r0, r1 in zip(occ_d0, occ_d1)])
rot1 = np.array([r['rot_err'] for r in occ_d1])
fig, ax = plt.subplots(figsize=(6.4, 4.8))
ax.scatter(imp[~m_cat], rot1[~m_cat], c='tab:blue', s=42, label='non-catastrophic')
ax.scatter(imp[m_cat], rot1[m_cat], c='red', s=52, marker='x', label='voting-catastrophic')
ax.axhline(90, color='k', ls='--', lw=1, label='flip = 90$^\\circ$')
ax.set_xlabel('D1 KP improvement  D0$-$D1 mean error (px, >0 = D1 better)')
ax.set_ylabel('D1 rotation error (deg)')
ax.set_title('OCC: keypoint improvement vs D1 rotation error')
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'fig2_occ_improvement_vs_rotation.png'), dpi=150); plt.close(fig)

# Fig 3: reprojection vs rotation, flip marked
proj1 = np.array([r['proj_px'] for r in occ_d1])
fl1 = np.array([r['flip_gt90'] for r in occ_d1])
fig, ax = plt.subplots(figsize=(6.4, 4.8))
ax.scatter(proj1[~fl1], rot1[~fl1], c='tab:blue', s=42, label='D1 stable')
ax.scatter(proj1[fl1], rot1[fl1], c='red', s=52, marker='x', label='D1 flip (>90$^\\circ$)')
ax.axvline(REPROJ_LOW, color='gray', ls=':', lw=1, label='reproj = 5 px (official 2D thr.)')
ax.set_xlabel('D1 mean reprojection error (px)'); ax.set_ylabel('D1 rotation error (deg)')
ax.set_title('OCC: reprojection vs rotation error (ambiguity check)')
ax.set_xscale('log'); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'fig3_occ_reproj_vs_rotation.png'), dpi=150); plt.close(fig)
log('3 figures saved')


# ---------------- summary.json ----------------
def rec_pub(r):
    return {k: (v if not isinstance(v, np.ndarray) else None)
            for k, v in r.items() if k != 'kp_err'}


summary = dict(
    identity_check=dict(max_seg_diff=max_seg, max_vertex_diff=max_ver,
                        mask_identical=mask_all_eq, max_kp_diff=max_kp,
                        d0_vs_d1_seg_bitwise_and_mask=SEG_ID_OK),
    direction_gate=dict(computed=DIR, reference_exp013=DIR_REF, tol_deg=0.01, passed=True),
    image_ids=dict(lin=HELD_RGB['linemod'], occ=HELD_RGB['occ'],
                   source='EXP015 results/image_ids.json (asserted == EXP013/EXP014)'),
    vote_params=dict(round_hyp_num=ROUND_HYP, inlier_thresh=INLIER_THRESH,
                     max_num=MAX_NUM, vote_type=str(VOTE_TYPE), seed=SEED),
    flip_counts={f'{s}_{n}': FLIP[(s, n)] for s in ('linemod', 'occ') for n in ('D0', 'D1')},
    reproduction_ok=bool(reproduction_ok),
    contingency={s: {k: v for k, v in CONTAB[s].items() if k != 'per_image'}
                 for s in ('linemod', 'occ')},
    kp_by_flip_group=GRP,
    reproj_ambiguity=AMBIG,
    catastrophic_vs_flip=CATX,
    noncat_occ_common=NC_OCC,
    spearman=CORR,
    decision=dict(majority_improved=bool(majority_improved),
                  d1_ncat_flip_rate=d1_ncat_rate,
                  d1_ncat_flips=d1_ncat_flips,
                  d1_flip_low_reproj_frac=amb['frac_flip_low_reproj'],
                  decision=decision,
                  rules='pre-registered in config.json'),
    per_image_records={f'{s}_{n}': [rec_pub(r) for r in REC[(s, n)]]
                       for s in ('linemod', 'occ') for n in ('D0', 'D1')},
    runtime_s=time.time() - T0)
json.dump(summary, open(os.path.join(RES_DIR, 'summary.json'), 'w'), indent=2)
log(f'DONE in {time.time()-T0:.1f}s')
