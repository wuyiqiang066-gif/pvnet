"""EXP015: Keypoint Localization Causal Diagnosis.

Research question (EXP013/EXP014 follow-up): EXP013 showed the D1 decoder
improves HELD-OUT OCC direction error (6.156 -> 4.936 deg, +19.8%) and EXP014
showed this does NOT propagate to pose (OCC ADD(-S) 0.05 -> 0.00, STOP-C;
PnP ~180-deg flip mode + shared catastrophic voting failures). The open
question: did the D1 OCC direction gain ACTUALLY and STABLY transfer to
keypoint localization, or was EXP014's keypoint-median improvement
(3.150 -> 2.143 px) a limited-sample phenomenon?

This is a measurement experiment: image -> segmentation -> vertex prediction
-> original PVNet RANSAC voting -> predicted keypoints, and STOP there.
NO PnP, NO ADD, NO pose of any kind is computed here. The only variable is
the vertex prediction (D0 original 1x1 head vs EXP013 final D1 decoder);
everything else (input, backbone, shared features, segmentation, voting
implementation, RANSAC parameters, RNG seeding, intrinsics, GT) is identical.

Strict isolation:
  - 199.pth frozen; D1 loaded from EXP014's persisted decoder states
    (d1_decoder_{linemod,occ}.pth), which are the EXP013 final states
    regenerated deterministically with the EXP013 fitting protocol and
    verified against EXP013's recorded held-out direction numbers before
    EXP014's pose evaluation. NO re-fitting here; the direction sanity gate
    re-verifies the loaded state against EXP013 references before use.
  - Voting: original lib ransac_voting_layer_v3 with the project's original
    parameters (round_hyp_num=128, inlier_thresh=0.99, max_num=100 from
    configs/exp001_reliability_voting_scratch.json, asserted); seed 0
    re-applied before EVERY voting call so both branches see the identical
    RNG stream.
  - Identity checks before evaluation: (a) D0 branch (manual trunk ->
    convraw[3]) must reproduce the original net.forward segmentation
    logits/mask and vertex outputs and voting keypoints (<1e-5) on every
    held-out image; (b) the D1 branch must leave segmentation logits AND
    argmax mask bit-identical to D0 (pure vertex intervention) -- abort
    otherwise.

Data: EXP013 HELD-OUT set only (20 LIN val_real_set[20:40] + 20 OCC
test_real_set[20:40]); IDs loaded from EXP013 results/image_ids.json (the
same set EXP014 used), FIT-excluded asserted, and re-saved to this
experiment's results/image_ids.json.

Metrics (per split x method, keypoint-level pooled over 9 kps x 20 images):
  mean/median/p75/p90/p95/max pixel error (finite errors; NaN predictions
  reported separately and counted as catastrophic);
  keypoint-level catastrophic rates (>20/>50/>100 px) and image-level
  catastrophic images (any keypoint error > 100 px or NaN);
  paired per-image analysis: per image mean/median over 9 kps, delta = D1-D0,
  better/worse/tie (tie: |delta| <= 1e-6), mean/median paired delta;
  non-catastrophic subset analysis (image max finite error <= 100 px):
  branch-specific subsets AND the common subset (non-catastrophic in BOTH
  branches) -- distinguishes case A (general improvement, catastrophic
  failures unresolved) from case B (no stable improvement on normal images).

Optional PnP failure diagnosis (EXPLANATION ONLY, no PnP is run here):
EXP014's per_image_pose.csv (rotation/translation/reprojection/ADD per
image, from the unmodified standard-PnP baseline evaluator) is reused to
count ~180-deg flip cases (rot_err > 90 deg) and cross-tabulate them with
this experiment's image-level keypoint catastrophic status.

Pre-registered decision (operationalization fixed BEFORE the run):
  GO   iff OCC pooled median D1 < D0
            AND paired mean-based better_count > worse_count (OCC)
            AND OCC common non-catastrophic subset pooled median D1 < D0
            (catastrophic failures are not dominating the conclusion)
  STOP iff OCC pooled median D1 >= D0
            AND paired mean-based better_count <= worse_count (OCC)
  WEAK otherwise (pooled-median improvement not stably paired, or
        driven by catastrophic-image changes, or median flat but paired
        suggests improvement).

Run from repo root:
  python experiments/exp015_keypoint_localization_diagnosis/run_exp015.py
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

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(EXP_DIR, 'results')
LOG_DIR = os.path.join(EXP_DIR, 'logs')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
EXP13_DIR = os.path.join(ROOT, 'experiments', 'exp013_heldout_decoder_generalization')
EXP14_DIR = os.path.join(ROOT, 'experiments', 'exp014_end_to_end_pose_validation')

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
CAT_KP = (20.0, 50.0, 100.0)             # keypoint-level catastrophic thresholds (px)
CAT_IMG = 100.0                          # image-level: any kp error > 100 px (or NaN)
TIE_EPS = 1e-6                           # paired tie tolerance (px)

EXP13_IDS = json.load(open(os.path.join(EXP13_DIR, 'results', 'image_ids.json')))
FIT_RGB = {s: EXP13_IDS['fit'][s]['rgb'] for s in ('linemod', 'occ')}
HELD_RGB = {s: EXP13_IDS['heldout'][s]['rgb'] for s in ('linemod', 'occ')}
for s in ('linemod', 'occ'):
    assert set(FIT_RGB[s]).isdisjoint(set(HELD_RGB[s])), f'FIT/HELD-OUT overlap in {s}'
    assert EXP13_IDS['disjoint_assert'][s] is True, f'EXP013 disjoint flag false for {s}'

# EXP013 recorded HELD-OUT direction references (D1-state verification gate)
EXP13_SUM = json.load(open(os.path.join(EXP13_DIR, 'results', 'final_summary.json')))
DIR_REF = {s: {'D0': EXP13_SUM['d0_heldout'][s if s != 'linemod' else 'lin'],
               'D1': EXP13_SUM['mean']['heldout'][s]['D1']}
           for s in ('linemod', 'occ')}

# EXP014 persisted D1 decoder states (EXP013 final states; NO re-fitting here)
D1_STATES = {s: os.path.join(EXP14_DIR, 'results', f'd1_decoder_{s}.pth')
             for s in ('linemod', 'occ')}
for s in ('linemod', 'occ'):
    assert os.path.exists(D1_STATES[s]), \
        f'D1 state missing: {D1_STATES[s]} -> STOP (re-fitting is forbidden)'
EXP14_PERIMG = os.path.join(EXP14_DIR, 'results', 'per_image_pose.csv')

torch.manual_seed(SEED); np.random.seed(SEED)
T0 = time.time()
LOG = open(os.path.join(LOG_DIR, 'run.log'), 'w', buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + '\n')

log('HELD-OUT image IDs reused from EXP013/EXP014 (no re-sampling):')
log(f'LIN image IDs: {[os.path.splitext(r)[0] for r in HELD_RGB["linemod"]]}')
log(f'OCC image IDs: {[os.path.splitext(r)[0] for r in HELD_RGB["occ"]]}')
json.dump(dict(
    source='EXP013 results/image_ids.json (heldout), identical to EXP014 usage',
    fit=EXP13_IDS['fit'], heldout=EXP13_IDS['heldout'],
    disjoint_assert=EXP13_IDS['disjoint_assert'],
    leakage_check='FIT vs HELD-OUT disjoint asserted in run_exp015.py'),
    open(os.path.join(RES_DIR, 'image_ids.json'), 'w'), indent=2)

json.dump(dict(
    cls='cat', n_img=N_IMG, ckpt='data/model/cat_linemod_train/199.pth',
    variable='vertex prediction ONLY: D0 original convraw[3] 1x1 head vs '
             'EXP013 final D1 decoder (zero-init 3x3 cross-channel residual), '
             'loaded from EXP014 persisted states; NO re-fitting',
    frozen='199.pth entire PVNet frozen; D1 state loaded and re-verified against '
           'EXP013 recorded held-out direction numbers (tol 0.01 deg) before use',
    data='EXP013 HELD-OUT set only, IDs loaded from EXP013 results/image_ids.json '
         '(identical to EXP014 usage): 20 LIN val_real_set[20:40], 20 OCC '
         'test_real_set[20:40]; FIT-excluded asserted',
    voting='original ransac_voting_layer_v3, params from '
           'configs/exp001_reliability_voting_scratch.json '
           f'(round_hyp_num={ROUND_HYP}, inlier_thresh={INLIER_THRESH}, '
           f'max_num={MAX_NUM}); seed {SEED} re-applied before every call for '
           'both branches; NO modification',
    pnp='NOT run in this experiment (keypoint localization only). Optional '
        'explanation-only diagnosis reuses EXP014 per_image_pose.csv '
        '(standard PnP outputs) to count ~180-deg flip cases',
    metrics='keypoint pixel error vs GT projected keypoints: pooled '
            'mean/median/p75/p90/p95/max, keypoint-level catastrophic rates '
            '(>20/>50/>100 px), image-level catastrophic images (>100 px max or '
            'NaN), paired per-image better/worse/tie (tie <=1e-6), non-'
            'catastrophic subset (branch-specific + common) mean/median/p90',
    identity_check='(a) D0 branch == original net.forward pipeline (<1e-5) on '
                   'every held-out image (seg logits/vertex/mask/voted kps); '
                   '(b) D1 branch seg logits and argmax mask BIT-IDENTICAL to D0',
    decision_rules='Pre-registered BEFORE the run: GO iff OCC pooled median D1<D0 '
                   'AND OCC paired mean-based better>worse AND OCC common non-'
                   'catastrophic subset median D1<D0; STOP iff OCC median D1>=D0 '
                   'AND paired better<=worse; WEAK otherwise',
    cat_thresholds=dict(keypoint_px=list(CAT_KP), image_px=CAT_IMG, tie_eps=TIE_EPS)),
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
                          vertex_gt=vertex_gt, vw=vw, gt_kps=gt2,
                          vis=(mask_gt[0].cpu().numpy() == 1)))
    return items


# ---------------- Step 1: cache HELD-OUT (FIT images never loaded) ----------------
CACHE = {}
for split in ('linemod', 'occ'):
    ds, loader = build_loader(split, N_IMG, 2 * N_IMG, expect=HELD_RGB[split])
    CACHE[split] = cache_split(split, ds, loader, N_IMG)
    log(f'--- {split}: {len(CACHE[split])} HELD-OUT images cached, IDs asserted '
        f'({time.time()-T0:.1f}s) ---')


# ---------------- Step 2: identity checks (before any evaluation) ----------------
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
max_feat = 0.0
for split in ('linemod', 'occ'):
    for item in CACHE[split]:
        with torch.no_grad():
            seg_f, ver_f = net(item['image'])                 # original PVNet outputs
        feat_f = trunk_feat(item['image'])                    # fresh trunk forward
        max_feat = max(max_feat, float((feat_f - item['feat']).abs().max()))
        seg0, ver0 = d0_outputs(item['feat'])                 # D0 branch outputs
        max_seg = max(max_seg, float((seg_f - seg0).abs().max()))
        max_ver = max(max_ver, float((ver_f - ver0).abs().max()))
        mask_all_eq &= bool((torch.argmax(seg_f, 1) == torch.argmax(seg0, 1)).all())
        kp_f = vote_keypoints(seg_f, ver_f)                   # original pipeline keypoints
        kp_0 = vote_keypoints(seg0, ver0)                     # D0-branch keypoints
        max_kp = max(max_kp, float((kp_f - kp_0).abs().max()))
log(f'identity check (D0 vs original pipeline): max|seg| {max_seg:.2e}  '
    f'max|vertex| {max_ver:.2e}  mask identical {mask_all_eq}  '
    f'max|kp| {max_kp:.2e}  max|feat cache| {max_feat:.2e}')
assert max_seg < 1e-5 and max_ver < 1e-5 and mask_all_eq and max_kp < 1e-5 \
       and max_feat < 1e-5, 'D0 branch does not reproduce original PVNet -> STOP'
log('identity check (a) PASSED')


# ---------------- Step 3: D1 state load + direction verification gate ----------------
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
        # EXP014 visible-fg pool geometry verbatim (GT seg mask, |K-p| >= 1px)
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


# ---------------- Step 4: keypoint localization (voting ONLY, NO PnP) ----------------
ROWS_KP = []                                   # per_keypoint.csv rows
PER = {(s, n): [] for s in ('linemod', 'occ') for n in ('D0', 'D1')}
SEG_ID_OK = True
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
            kpn = kp[n][0].cpu().numpy().astype(np.float64)
            err = np.linalg.norm(kpn - item['gt_kps'], axis=1)   # (9,) px
            PER[(split, n)].append(dict(
                image_id=item['image_id'],
                rgb=HELD_RGB[split][item['image_id'] - N_IMG], err=err))
            for k in range(KP):
                ROWS_KP.append([split, item['image_id'], n, k, err[k]])
log(f'segmentation identity D0 vs D1 (logits bitwise + mask): {SEG_ID_OK} '
    f'-> pure vertex causal intervention confirmed')
assert SEG_ID_OK, 'D0/D1 segmentation mismatch -> STOP'


def img_cat(err):
    """Image-level catastrophic: any keypoint NaN or max finite error > CAT_IMG."""
    fin = err[np.isfinite(err)]
    return bool(fin.size < err.size or (fin.size > 0 and fin.max() > CAT_IMG))


def img_max_finite(err):
    fin = err[np.isfinite(err)]
    return float(fin.max()) if fin.size else float('inf')


def stats_ext(a):
    a = np.asarray(a, np.float64)
    fin = a[np.isfinite(a)]
    d = dict(count=int(fin.size), nan_count=int(a.size - fin.size))
    if fin.size:
        q = np.quantile(fin, [0.75, 0.90, 0.95])
        d.update(mean=float(fin.mean()), median=float(np.median(fin)),
                 p75=float(q[0]), p90=float(q[1]), p95=float(q[2]), max=float(fin.max()))
    else:
        d.update(mean=None, median=None, p75=None, p90=None, p95=None, max=None)
    return d


SUM = {}
for s in ('linemod', 'occ'):
    for n in ('D0', 'D1'):
        errs = np.concatenate([r['err'] for r in PER[(s, n)]])
        st = stats_ext(errs)
        st['cat_rate_kp'] = {f'gt{int(t)}px': float(((errs > t) | ~np.isfinite(errs)).mean())
                             for t in CAT_KP}
        st['cat_images'] = int(sum(img_cat(r['err']) for r in PER[(s, n)]))
        st['n_images'] = len(PER[(s, n)])
        per_kp = []
        for k in range(KP):
            vals = np.concatenate([r['err'][k:k + 1] for r in PER[(s, n)]])
            fin = vals[np.isfinite(vals)]
            per_kp.append(float(fin.mean()) if fin.size else None)
        st['per_kp_mean_px'] = per_kp
        SUM[(s, n)] = st
        log(f"kp [{s} {n}] mean {st['mean']:.3f}  median {st['median']:.3f}  "
            f"p75 {st['p75']:.3f}  p90 {st['p90']:.3f}  p95 {st['p95']:.3f}  "
            f"max {st['max']:.1f}  cat_imgs {st['cat_images']}/{st['n_images']}  "
            f"cat_rates {st['cat_rate_kp']}")

# ---------------- paired per-image analysis ----------------
PAIRED = {}
PERIMG_ROWS = []
for s in ('linemod', 'occ'):
    rows = []
    for i in range(len(PER[(s, 'D0')])):
        r0, r1 = PER[(s, 'D0')][i], PER[(s, 'D1')][i]
        assert r0['image_id'] == r1['image_id']
        e0, e1 = r0['err'], r1['err']
        f0, f1 = e0[np.isfinite(e0)], e1[np.isfinite(e1)]
        m0 = float(f0.mean()) if f0.size else float('nan')
        m1 = float(f1.mean()) if f1.size else float('nan')
        md0 = float(np.median(f0)) if f0.size else float('nan')
        md1 = float(np.median(f1)) if f1.size else float('nan')
        rows.append(dict(split=s, image_id=r0['image_id'], rgb=r0['rgb'],
                         D0_mean=m0, D1_mean=m1, D0_median=md0, D1_median=md1,
                         D0_max=img_max_finite(e0), D1_max=img_max_finite(e1),
                         D0_cat=img_cat(e0), D1_cat=img_cat(e1)))
    for r in rows:
        r['delta_mean'] = r['D1_mean'] - r['D0_mean']
        r['delta_median'] = r['D1_median'] - r['D0_median']
    dm = np.array([r['delta_mean'] for r in rows], np.float64)
    dd = np.array([r['delta_median'] for r in rows], np.float64)
    def bwt(d):
        b = int((d < -TIE_EPS).sum()); w = int((d > TIE_EPS).sum())
        t = int((np.abs(d) <= TIE_EPS).sum())
        return b, w, t
    PAIRED[s] = dict(
        mean_based=dict(better=bwt(dm)[0], worse=bwt(dm)[1], tie=bwt(dm)[2],
                        mean_delta=float(np.nanmean(dm)) if np.isfinite(dm).any() else None,
                        median_delta=float(np.nanmedian(dm)) if np.isfinite(dm).any() else None),
        median_based=dict(better=bwt(dd)[0], worse=bwt(dd)[1], tie=bwt(dd)[2]))
    PERIMG_ROWS.extend(rows)
    log(f"paired [{s}] mean-based better/worse/tie = {bwt(dm)}  "
        f"mean_delta {PAIRED[s]['mean_based']['mean_delta']:+.3f}  "
        f"median_delta {PAIRED[s]['mean_based']['median_delta']:+.3f}")
    log(f"paired [{s}] median-based better/worse/tie = {bwt(dd)}")

# ---------------- non-catastrophic subset analysis ----------------
NONCAT = {}
for s in ('linemod', 'occ'):
    idx0 = [i for i, r in enumerate(PER[(s, 'D0')]) if not img_cat(r['err'])]
    idx1 = [i for i, r in enumerate(PER[(s, 'D1')]) if not img_cat(r['err'])]
    com = sorted(set(idx0) & set(idx1))
    def pooled(idxs, n):
        if not idxs:
            return dict(count=0)
        return stats_ext(np.concatenate([PER[(s, n)][i]['err'] for i in idxs]))
    NONCAT[s] = dict(
        D0_branch_specific=dict(images=idx0, stats=pooled(idx0, 'D0')),
        D1_branch_specific=dict(images=idx1, stats=pooled(idx1, 'D1')),
        common=dict(images=com, D0=pooled(com, 'D0'), D1=pooled(com, 'D1')),
        n_cat_D0=len(PER[(s, 'D0')]) - len(idx0), n_cat_D1=len(PER[(s, 'D1')]) - len(idx1))
    c0, c1 = NONCAT[s]['common']['D0'], NONCAT[s]['common']['D1']
    m0 = c0.get('median'); m1 = c1.get('median')
    log(f"non-cat common [{s}]: {len(com)} images  "
        f"D0 median {m0 if m0 is not None else float('nan'):.3f}  "
        f"D1 median {m1 if m1 is not None else float('nan'):.3f}")

# ---------------- optional: PnP flip diagnosis from EXP014 (no PnP run here) ----------------
PNP_DIAG = None
if os.path.exists(EXP14_PERIMG):
    rows14 = list(csv.DictReader(open(EXP14_PERIMG)))
    PNP_DIAG = {}
    for s in ('linemod', 'occ'):
        for n in ('D0', 'D1'):
            rr = [r for r in rows14 if r['split'] == s and r['decoder'] == n]
            flips = [r for r in rr if float(r['rot_err_deg']) > 90.0]
            cat_ids = {PER[(s, n)][i]['image_id']
                       for i in range(len(PER[(s, n)])) if img_cat(PER[(s, n)][i]['err'])}
            PNP_DIAG[f'{s}_{n}'] = dict(
                n_images=len(rr),
                pnp_flip_gt90deg=len(flips),
                flip_image_ids=[int(r['image_id']) for r in flips],
                cat_images_this_branch=sorted(cat_ids),
                flip_and_keypoint_catastrophic=len(
                    {int(r['image_id']) for r in flips} & cat_ids))
    for k, v in PNP_DIAG.items():
        log(f"PnP diag (EXP014 reuse) [{k}]: flips(>90deg) {v['pnp_flip_gt90deg']}/"
            f"{v['n_images']}  flip&cat overlap {v['flip_and_keypoint_catastrophic']}")

# ---------------- decision (pre-registered operationalization) ----------------
d0m = SUM[('occ', 'D0')]['median']; d1m = SUM[('occ', 'D1')]['median']
med_ok = d0m is not None and d1m is not None and d1m < d0m
pb = PAIRED['occ']['mean_based']
paired_ok = pb['better'] > pb['worse']
nc = NONCAT['occ']['common']
m0n, m1n = nc['D0'].get('median'), nc['D1'].get('median')
noncat_ok = m0n is not None and m1n is not None and m1n < m0n
if med_ok and paired_ok and noncat_ok:
    decision = 'GO'
elif (not med_ok) and (not paired_ok):
    decision = 'STOP'
else:
    decision = 'WEAK'
if d0m is not None and d1m is not None:
    log(f"OCC median: D0 {d0m:.3f} -> D1 {d1m:.3f}  ({(d0m-d1m)/d0m:+.2%} relative change)")
    log(f"OCC mean:   D0 {SUM[('occ','D0')]['mean']:.3f} -> D1 {SUM[('occ','D1')]['mean']:.3f}")
log(f"DECISION: {decision}  (median_ok {med_ok}, paired_ok {paired_ok}, noncat_ok {noncat_ok})")


# ---------------- outputs ----------------
with open(os.path.join(RES_DIR, 'per_keypoint.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['dataset', 'image_id', 'method', 'keypoint_id', 'error_px'])
    for s, iid, n, k, e in ROWS_KP:
        wr.writerow([s, iid, n, k, 'nan' if not np.isfinite(e) else f'{e:.4f}'])

with open(os.path.join(RES_DIR, 'per_image.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['dataset', 'image_id', 'rgb', 'D0_mean_kp_error', 'D1_mean_kp_error',
                 'D0_median_kp_error', 'D1_median_kp_error', 'D0_max_kp_error',
                 'D1_max_kp_error', 'D0_catastrophic', 'D1_catastrophic',
                 'delta_mean', 'delta_median'])
    for r in PERIMG_ROWS:
        wr.writerow([r['split'], r['image_id'], r['rgb'],
                     f"{r['D0_mean']:.4f}", f"{r['D1_mean']:.4f}",
                     f"{r['D0_median']:.4f}", f"{r['D1_median']:.4f}",
                     f"{r['D0_max']:.2f}", f"{r['D1_max']:.2f}",
                     int(r['D0_cat']), int(r['D1_cat']),
                     f"{r['delta_mean']:.4f}", f"{r['delta_median']:.4f}"])

summary = dict(
    identity_check=dict(max_seg_diff=max_seg, max_vertex_diff=max_ver,
                        mask_identical=mask_all_eq, max_kp_diff=max_kp,
                        max_feat_cache_diff=max_feat,
                        d0_vs_d1_seg_bitwise_and_mask=SEG_ID_OK),
    direction_gate=dict(computed=DIR, reference_exp013=DIR_REF,
                        tol_deg=0.01, passed=True),
    image_ids=dict(lin=HELD_RGB['linemod'], occ=HELD_RGB['occ'],
                   source='EXP013 results/image_ids.json (heldout), EXP014-identical'),
    vote_params=dict(round_hyp_num=ROUND_HYP, inlier_thresh=INLIER_THRESH,
                     max_num=MAX_NUM, vote_type=str(VOTE_TYPE), seed=SEED),
    keypoint_stats={f'{s}_{n}': SUM[(s, n)] for s in ('linemod', 'occ') for n in ('D0', 'D1')},
    paired={s: PAIRED[s] for s in ('linemod', 'occ')},
    noncatastrophic_subset=NONCAT,
    pnp_diagnosis_exp014_reuse=PNP_DIAG,
    decision_rules='GO iff OCC median D1<D0 AND OCC paired mean-based better>worse '
                   'AND OCC common non-cat subset median D1<D0; STOP iff median '
                   'D1>=D0 AND paired better<=worse; WEAK otherwise',
    decision=dict(median_ok=bool(med_ok), paired_ok=bool(paired_ok),
                  noncat_ok=bool(noncat_ok), decision=decision),
    runtime_s=time.time() - T0)
json.dump(summary, open(os.path.join(RES_DIR, 'summary.json'), 'w'), indent=2)
log(f'DONE in {time.time()-T0:.1f}s')
