"""EXP014: End-to-End Pose Causal Validation.

Research question (EXP012/013 follow-up): EXP012/013 established that a light
cross-channel 3x3 decoder (D1) on the FROZEN PVNet shared feature improves
pixel-to-keypoint direction prediction and that the improvement generalizes
to held-out images (HELD-OUT OCC 6.156 -> 4.936 deg, +19.83%, GO-B). The
only question here: does the D1 direction improvement PROPAGATE through
PVNet's original voting -> keypoint -> PnP pipeline into real 6D pose
improvement? This is a causal validation experiment: the ONLY variable is
the vertex prediction (D0 original 1x1 head vs EXP013 frozen D1 decoder);
every downstream component is the project's original baseline, unmodified.

Strict isolation:
  - 199.pth frozen; segmentation untouched (D1 branch adds channels AFTER
    the original head split, so seg channels are bit-identical to D0).
  - Voting: original lib ransac_voting_layer_v3, original parameters
    (round_hyp_num=128, inlier_thresh=0.99, max_num=100 from
    configs/exp001_reliability_voting_scratch.json, asserted at runtime),
    identical RNG seeding (seed 0 re-applied before EVERY voting call) for
    both branches. No RANSAC/pnp/evaluator modification of any kind.
  - Pose: original Evaluator (lib.utils.evaluation_utils) standard PnP path
    (pnp(), cv2.SOLVEPNP_ITERATIVE, 'linemod' intrinsics, Farthest 3D
    points) -- exactly the project's formal baseline evaluator.
  - D1 parameters: EXP013's final decoder state. EXP013 did not persist the
    decoder weights, so the state is REGENERATED deterministically with the
    EXP013 fitting protocol verbatim (same data, same seed/optimizer/epochs;
    EXP013 demonstrated bit-exact reproducibility: its FIT results equal
    EXP012's to all printed digits). The regenerated state is verified
    against EXP013's recorded HELD-OUT direction numbers BEFORE any pose
    evaluation (abort otherwise), then frozen and saved locally for reuse.
    No re-fitting decisions, no tuning.

Pipeline identity check (before anything else): on every HELD-OUT image the
D0 branch (manual trunk -> convraw[3]) must produce segmentation and vertex
outputs numerically identical to the original net.forward outputs, identical
argmax masks, and identical keypoints through the identical voting call
(max abs diff < 1e-5); abort otherwise.

Data: EXP013's HELD-OUT set only (20 LIN val_real_set[20:40] + 20 OCC
test_real_set first half [20:40]); IDs loaded from EXP013's
results/image_ids.json and asserted; FIT images excluded (leakage assert).

Metrics chain (per split, D0 vs D1):
  L1 direction: EXP006/008 pool (visible fg, |K-p|>=1px, 9 kps pooled)
                mean/median/p90 deg, bad>10deg, endpoint (unit space).
  L2 keypoint:  pixel error of voted keypoints vs GT projected keypoints,
                pooled over all 9 kps x 20 images: mean/median px.
  L3 pose:      original Evaluator recorders: ADD (<0.1*diameter fraction;
                mean 3D distance in mm as secondary), 2D projection
                (<5 px fraction; mean px secondary), 5cm5deg fraction.
No proximity band analysis (EXP013 found inconsistent FIT/HELD-OUT band
patterns; per instruction it is out of scope here).

Pre-registered decision (thresholds fixed in advance; ADD/2D/5cm5deg are
  SUCCESS-RATE fractions, so improvement = relative increase (D1-D0)/D0):
  HELD-OUT OCC ADD improvement
  >= 10%  -> GO-C       (direction gain propagates to pose)
  <  3%   -> STOP-C     (gain does not translate; locate bottleneck next)
  else    -> BORDERLINE (report chain, await supervisor)

Run from repo root:
  python experiments/exp014_end_to_end_pose_validation/run_exp014.py
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
from lib.utils.net_utils import smooth_l1_loss
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v3
from lib.utils.evaluation_utils import Evaluator

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(EXP_DIR, 'results')
os.makedirs(RES_DIR, exist_ok=True)
EXP13_DIR = os.path.join(ROOT, 'experiments', 'exp013_heldout_decoder_generalization')

N_IMG = 20
KP = 9
EPOCHS = 5                     # EXP013 fitting protocol verbatim (state regeneration)
LR = 1e-3
SEED = 0
EPS = 1e-9
VOTE_TYPE = VotingType.Farthest          # train_linemod_exp001.py verbatim
TRAIN_CFG = json.load(open(os.path.join(ROOT, 'configs',
                                        'exp001_reliability_voting_scratch.json')))
ROUND_HYP, MAX_NUM = TRAIN_CFG['vote_round_hyp_num'], TRAIN_CFG['vote_max_num']
INLIER_THRESH = 0.99                     # EvalWrapperBaseline verbatim
assert (ROUND_HYP, MAX_NUM) == (128, 100), 'vote params changed -> STOP'

# EXP013 recorded HELD-OUT direction references (direction sanity gate, Sec.7)
EXP13_SUM = json.load(open(os.path.join(EXP13_DIR, 'results', 'final_summary.json')))
DIR_REF = {s: {'D0': EXP13_SUM['d0_heldout'][s if s != 'linemod' else 'lin'],
               'D1': EXP13_SUM['mean']['heldout'][s]['D1']}
           for s in ('linemod', 'occ')}
GO_C_MIN, STOP_C_MAX = 0.10, 0.03        # pre-registered thresholds

torch.manual_seed(SEED); np.random.seed(SEED)
T0 = time.time()
LOG = open(os.path.join(RES_DIR, 'run.log'), 'w', buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + '\n')

json.dump(dict(cls='cat', n_img=N_IMG, ckpt='data/model/cat_linemod_train/199.pth',
               variable='vertex prediction ONLY: D0 original convraw[3] 1x1 head '
                        'vs EXP013 frozen D1 decoder (zero-init 3x3 cross-channel '
                        'residual, 9842 params); everything downstream identical',
               frozen='199.pth entire PVNet frozen; D1 loaded/regenerated from EXP013 '
                      'final state (EXP013 fitting protocol verbatim, seed 0; verified '
                      'against EXP013 recorded held-out direction numbers before use)',
               data='EXP013 HELD-OUT set only, IDs loaded from EXP013 results/image_ids.json '
                    '(20 LIN val_real_set[20:40], 20 OCC test_real_set first half [20:40]); '
                    'FIT-excluded asserted',
               voting='original ransac_voting_layer_v3, params from '
                      'configs/exp001_reliability_voting_scratch.json '
                      f'(round_hyp_num={ROUND_HYP}, inlier_thresh={INLIER_THRESH}, '
                      f'max_num={MAX_NUM}); seed {SEED} re-applied before every call '
                      'for both branches; NO modification',
               pnp='original Evaluator standard PnP path (pnp, SOLVEPNP_ITERATIVE, '
                   "projector.intrinsic_matrix['linemod'], Farthest 3D points); "
                   'uncertainty PnP NOT used (project baseline is standard PnP)',
               metrics='L1 direction (EXP006/008 pool), L2 keypoint pixel error '
                       '(pooled 9 kps x 20 imgs), L3 ADD/2D-projection/5cm5deg '
                       '(original Evaluator recorders); no proximity analysis (per spec)',
               identity_check='D0 branch seg/vertex/mask/keypoints must equal original '
                              'net.forward pipeline (<1e-5) on all held-out images',
               go_stop=f'HELD-OUT OCC ADD improvement >= {GO_C_MIN:.0%} -> GO-C; '
                       f'< {STOP_C_MAX:.0%} -> STOP-C; else BORDERLINE'),
          open(os.path.join(RES_DIR, 'config.json'), 'w'), indent=2)

device = 'cuda'
net = Resnet18_8s(ver_dim=KP * 2, seg_dim=2)
sd = torch.load(os.path.join(ROOT, 'data/model/cat_linemod_train/199.pth'), map_location='cpu')
net.load_state_dict(sd['net'])
net = net.to(device).eval()
for p in net.parameters():
    p.requires_grad_(False)

AUG_CFG = TRAIN_CFG['aug_cfg']
AMODAL_DIR = {'linemod': os.path.join(cfg.LINEMOD, 'cat', 'amodal_mask'),
              'occ': os.path.join(cfg.OCCLUSION_LINEMOD, 'amodal_masks', 'cat')}
EXP13_IDS = json.load(open(os.path.join(EXP13_DIR, 'results', 'image_ids.json')))
EXP5_IDS = json.load(open(os.path.join(
    ROOT, 'experiments', 'exp005_visibility_diagnosis', 'results', 'image_ids.json')))
FIT_RGB = {s: EXP13_IDS['fit'][s]['rgb'] for s in ('linemod', 'occ')}
HELD_RGB = {s: EXP13_IDS['heldout'][s]['rgb'] for s in ('linemod', 'occ')}
# leakage asserts: EXP013 file itself asserts disjointness; re-assert here
for s in ('linemod', 'occ'):
    assert set(FIT_RGB[s]).isdisjoint(set(HELD_RGB[s])), f'FIT/HELD-OUT overlap in {s}'
    assert EXP13_IDS['disjoint_assert'][s] is True, f'EXP013 disjoint flag false for {s}'


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


def angle_deg(a, b):
    dot = np.clip((a * b).sum(-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def mstats(a):
    a = np.asarray(a, np.float64); a = a[np.isfinite(a)]
    return dict(count=int(a.size), mean=float(a.mean()) if a.size else float('nan'),
                median=float(np.median(a)) if a.size else float('nan'),
                p90=float(np.quantile(a, 0.9)) if a.size else float('nan'))


def d0_outputs(feat):
    """D0 branch: original frozen 1x1 head on the shared feature (frozen F path)."""
    with torch.no_grad():
        out = net.convraw[3](feat)
        return out[:, :net.seg_dim], out[:, net.seg_dim:]


def cache_split(split, ds, loader, id_offset, with_pose):
    """One frozen forward per image: F, D0 seg/vertex, pool geometry, (+pose/kps)."""
    items = []
    for i, data in enumerate(loader):
        image, mask_gt, vertex_gt, vw, pose, gt_kps = [d.to(device) for d in data]
        assert vertex_gt.shape[1] == KP * 2 and vw.shape[1] == 1
        with torch.no_grad():
            x2s, x4s, x8s, x16s, x32s, xfc = net.resnet18_8s(image)
            fm = net.conv8s(torch.cat([xfc, x8s], 1)); fm = net.up8sto4s(fm)
            fm = net.conv4s(torch.cat([fm, x4s], 1)); fm = net.up4sto2s(fm)
            fm = net.conv2s(torch.cat([fm, x2s], 1)); fm = net.up2storaw(fm)
            feat = net.convraw[0:3](torch.cat([fm, image], 1))   # (1,32,H,W)
        item = dict(image_id=id_offset + i, image=image, feat=feat,
                    vertex_gt=vertex_gt, vw=vw)
        if with_pose:
            gt2 = (gt_kps[0, :, :2] if gt_kps.dim() == 3 else gt_kps[0, 0, :, :2]).cpu().numpy()
            vis = (mask_gt[0].cpu().numpy() == 1)
            ys, xs = np.nonzero(vis)
            coords = np.stack([xs, ys], 1).astype(np.float64)
            v_raw = gt2[None, :, :] - coords[:, None, :]
            d_raw = np.linalg.norm(v_raw, axis=2)
            ok = d_raw >= 1.0
            u_gt = v_raw / (d_raw[..., None] + EPS)
            item.update(vis=vis, ys=ys, xs=xs, ok=ok, u_gt=u_gt,
                        pose=pose[0].cpu().numpy(),           # (3,4) GT pose
                        gt_kps=gt2)                           # (9,2) GT projected kps
        items.append(item)
    return items


# ---------------- build FIT (for D1 state regeneration) and HELD-OUT caches ----------------
CACHE = {'fit': {}, 'heldout': {}}
for split in ('linemod', 'occ'):
    ds, loader = build_loader(split, 0, N_IMG, expect=FIT_RGB[split])
    CACHE['fit'][split] = cache_split(split, ds, loader, 0, with_pose=False)
    ds, loader = build_loader(split, N_IMG, 2 * N_IMG, expect=HELD_RGB[split])
    CACHE['heldout'][split] = cache_split(split, ds, loader, N_IMG, with_pose=True)
    log(f'--- {split}: FIT + HELD-OUT cached and ID-asserted '
        f'({time.time()-T0:.1f}s) ---')


# ---------------- Step 1: pipeline identity check (Sec.6, before anything else) ----------------
def vote_keypoints(seg, ver):
    """EvalWrapperBaseline verbatim (permute/view + argmax mask + ransac_voting_layer_v3),
    with seed re-applied so both branches see the identical RNG stream."""
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
    for item in CACHE['heldout'][split]:
        with torch.no_grad():
            seg_f, ver_f = net(item['image'])                 # original PVNet outputs
        seg0, ver0 = d0_outputs(item['feat'])                 # D0 branch outputs
        max_seg = max(max_seg, float((seg_f - seg0).abs().max()))
        max_ver = max(max_ver, float((ver_f - ver0).abs().max()))
        mask_all_eq &= bool((torch.argmax(seg_f, 1) == torch.argmax(seg0, 1)).all())
        kp_f = vote_keypoints(seg_f, ver_f)                   # original pipeline keypoints
        kp_0 = vote_keypoints(seg0, ver0)                     # D0-branch keypoints
        max_kp = max(max_kp, float((kp_f - kp_0).abs().max()))
log(f'identity check: max|seg diff| {max_seg:.2e}  max|vertex diff| {max_ver:.2e}  '
    f'mask identical {mask_all_eq}  max|keypoint diff| {max_kp:.2e}')
assert max_seg < 1e-5 and max_ver < 1e-5 and mask_all_eq and max_kp < 1e-5, \
    'D0 branch does not reproduce the original PVNet pipeline -> STOP'
log('pipeline identity check PASSED (D0 == original baseline, all held-out images)')


# ---------------- Step 2: EXP013 D1 state (deterministic regeneration, verified) ----------------
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


def full_image_loss(dec, item):
    with torch.no_grad():
        out = normalize_vertices(dec(item['feat']))
        return float(smooth_l1_loss(out, item['vertex_gt'], item['vw'],
                                    reduce=False)[0])


def fit_d1(split):
    """EXP013 fit_d1 verbatim (pixel minibatch 4096, 5 epochs, Adam 1e-3, seed 0)."""
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    dec = ResidualDecoder().to(device)
    init_loss = float(np.mean([full_image_loss(dec, it) for it in CACHE['fit'][split]]))
    d0_loss = float(np.mean([full_image_loss(d0_decode_only, it) for it in CACHE['fit'][split]]))
    log(f'regen D1-{split}: params {sum(p.numel() for p in dec.parameters())}  '
        f'init loss {init_loss:.6f}  (D0 reference {d0_loss:.6f})')
    opt = torch.optim.Adam(dec.parameters(), lr=LR)
    dec.train()
    PB = 4096
    for ep in range(EPOCHS):
        tot, nb = 0.0, 0
        for item in CACHE['fit'][split]:
            gt, vw = item['vertex_gt'], item['vw']
            fg = (vw[0, 0] > 0).nonzero(as_tuple=False)
            perm = rng.permutation(fg.shape[0])
            for s0 in range(0, fg.shape[0], PB):
                idx = perm[s0:s0 + PB]
                s = torch.zeros(1, 1, gt.shape[2], gt.shape[3], device=device)
                s[0, 0, fg[idx, 0], fg[idx, 1]] = 1.0
                out = normalize_vertices(dec(item['feat']))
                loss = smooth_l1_loss(out, gt, vw * s, reduce=False)[0]
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach()); nb += 1
        log(f'regen D1-{split} epoch {ep+1}/{EPOCHS} loss {tot/max(nb,1):.6f}  '
            f't={time.time()-T0:.0f}s')
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec


def d0_decode_only(feat):
    with torch.no_grad():
        return net.convraw[3](feat)[:, net.seg_dim:]


MODELS = {s: fit_d1(s) for s in ('linemod', 'occ')}
for s in ('linemod', 'occ'):
    torch.save(MODELS[s].state_dict(),
               os.path.join(RES_DIR, f'd1_decoder_{s}.pth'))   # local reuse only, NOT committed
log('D1 state regenerated (EXP013 protocol verbatim) and saved locally')


# ---------------- Step 3: direction sanity check vs EXP013 (Sec.7) ----------------
def evaluate_direction(caches):
    out = {}
    for split in ('linemod', 'occ'):
        res = {'D0': {'ang': [], 'end': []}, 'D1': {'ang': [], 'end': []}}
        for item in caches[split]:
            feat, ys, xs, ok, u_gt = item['feat'], item['ys'], item['xs'], item['ok'], item['u_gt']
            H, W = item['vis'].shape
            for name in ('D0', 'D1'):
                with torch.no_grad():
                    v = (d0_decode_only(feat) if name == 'D0' else MODELS[split](feat))
                    v = normalize_vertices(v)
                ver = v.permute(0, 2, 3, 1).contiguous().view(1, H, W, KP, 2)[0]
                dirs = ver[ys, xs].cpu().numpy().astype(np.float64)
                u_pred = dirs / (np.linalg.norm(dirs, axis=2)[..., None] + EPS)
                ang = angle_deg(u_pred, u_gt)
                res[name]['ang'].append(ang[ok])
                res[name]['end'].append(np.linalg.norm(u_pred - u_gt, axis=2)[ok])
        out[split] = {n: dict(stats=mstats(np.concatenate(res[n]['ang'])),
                              endpoint=float(np.concatenate(res[n]['end']).mean()),
                              bad=float((np.concatenate(res[n]['ang']) > 10).mean()))
                      for n in ('D0', 'D1')}
    return out


DIR = evaluate_direction(CACHE['heldout'])
for s in ('linemod', 'occ'):
    log(f"direction sanity [{s}]: D0 {DIR[s]['D0']['stats']['mean']:.3f} "
        f"(EXP013 {DIR_REF[s]['D0']:.3f})  D1 {DIR[s]['D1']['stats']['mean']:.3f} "
        f"(EXP013 {DIR_REF[s]['D1']:.3f})")
    assert abs(DIR[s]['D0']['stats']['mean'] - DIR_REF[s]['D0']) < 0.01 and \
           abs(DIR[s]['D1']['stats']['mean'] - DIR_REF[s]['D1']) < 0.01, \
        'direction does not reproduce EXP013 held-out -> STOP'
log('direction sanity check PASSED (reproduces EXP013 held-out)')


# ---------------- Step 4: end-to-end pose pipeline (identical downstream, Sec.8) ----------------
EVALUATORS = {(s, n): Evaluator() for s in ('linemod', 'occ') for n in ('D0', 'D1')}
KP_ERR = {(s, n): [] for s in ('linemod', 'occ') for n in ('D0', 'D1')}
PER_IMAGE = {(s, n): [] for s in ('linemod', 'occ') for n in ('D0', 'D1')}

for split in ('linemod', 'occ'):
    for item in CACHE['heldout'][split]:
        seg0, ver0 = d0_outputs(item['feat'])
        with torch.no_grad():
            ver1 = MODELS[split](item['feat'])
            seg1 = net.convraw[3](item['feat'])[:, :net.seg_dim]
        assert (seg1 == seg0).all(), 'seg channel changed by D1 branch -> STOP'
        kp = {'D0': vote_keypoints(seg0, ver0), 'D1': vote_keypoints(seg1, ver1)}
        for n in ('D0', 'D1'):
            kpn = kp[n][0].cpu().numpy().astype(np.float64)
            err = np.linalg.norm(kpn - item['gt_kps'], axis=1)       # (9,) px
            KP_ERR[(split, n)].append(err)
            ev = EVALUATORS[(split, n)]
            pose_pred = ev.evaluate(kpn, item['pose'], 'cat',
                                    'linemod', VOTE_TYPE, intri_matrix=None)
            PER_IMAGE[(split, n)].append(dict(
                image_id=item['image_id'], rgb=HELD_RGB[split][item['image_id'] - N_IMG],
                kp_mean_px=float(err.mean()), kp_max_px=float(err.max()),
                add_dist_m=float(ev.add_dists[-1]), add_pass=bool(ev.add_recorder[-1]),
                proj_px=float(ev.proj_mean_diffs[-1]), proj_pass=bool(ev.projection_2d_recorder[-1]),
                cm_pass=bool(ev.cm_degree_5_recorder[-1]),
                rot_deg=float(np.rad2deg(np.arccos(np.clip((np.trace(
                    pose_pred[:, :3] @ item['pose'][:, :3].T) - 1) / 2, -1, 1)))),
                trans_m=float(np.linalg.norm(pose_pred[:, 3] - item['pose'][:, 3]))))

POSE = {}
for s in ('linemod', 'occ'):
    for n in ('D0', 'D1'):
        ev = EVALUATORS[(s, n)]
        # identical arithmetic to Evaluator.average_precision (avoids its tmp.npy write)
        proj = float(np.mean(ev.projection_2d_recorder))
        add = float(np.mean(ev.add_recorder))
        cm = float(np.mean(ev.cm_degree_5_recorder))
        kp_all = np.concatenate(KP_ERR[(s, n)])
        POSE[(s, n)] = dict(add_frac=add, proj_frac=proj, cm_frac=cm,
                            add_dist_mm=float(np.mean(ev.add_dists)) * 1000.0,
                            proj_mean_px=float(np.mean(ev.proj_mean_diffs)),
                            kp_mean_px=float(kp_all.mean()),
                            kp_median_px=float(np.median(kp_all)),
                            per_kp_mean_px=[float(x) for x in
                                            np.stack(KP_ERR[(s, n)]).mean(0)],
                            n_images=len(ev.add_recorder))
        log(f"pose [{s} {n}] ADD {add:.4f} ({POSE[(s,n)]['add_dist_mm']:.1f}mm)  "
            f"2D {proj:.4f} ({POSE[(s,n)]['proj_mean_px']:.2f}px)  "
            f"5cm5deg {cm:.4f}  KP {POSE[(s,n)]['kp_mean_px']:.3f}px "
            f"(median {POSE[(s,n)]['kp_median_px']:.3f}px)")

with open(os.path.join(RES_DIR, 'per_image_pose.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['split', 'decoder', 'image_id', 'rgb', 'kp_mean_px', 'kp_max_px',
                 'add_dist_mm', 'add_pass', 'proj_px', 'proj_pass', '5cm5deg_pass',
                 'rot_err_deg', 'trans_err_mm'])
    for s in ('linemod', 'occ'):
        for n in ('D0', 'D1'):
            for r in PER_IMAGE[(s, n)]:
                wr.writerow([s, n, r['image_id'], r['rgb'], f"{r['kp_mean_px']:.2f}",
                             f"{r['kp_max_px']:.1f}", f"{r['add_dist_m']*1000:.1f}",
                             int(r['add_pass']), f"{r['proj_px']:.2f}", int(r['proj_pass']),
                             int(r['cm_pass']), f"{r['rot_deg']:.2f}",
                             f"{r['trans_m']*1000:.1f}"])

# ---------------- outputs ----------------
with open(os.path.join(RES_DIR, 'pose_summary.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['split', 'decoder', 'n_images', 'ADD_frac(<0.1d)', 'ADD_mean_dist_mm',
                 '2D_proj_frac(<5px)', '2D_proj_mean_px', '5cm5deg_frac',
                 'keypoint_mean_px', 'keypoint_median_px'])
    for s in ('linemod', 'occ'):
        for n in ('D0', 'D1'):
            p = POSE[(s, n)]
            wr.writerow([s, n, p['n_images'], f"{p['add_frac']:.4f}",
                         f"{p['add_dist_mm']:.2f}", f"{p['proj_frac']:.4f}",
                         f"{p['proj_mean_px']:.3f}", f"{p['cm_frac']:.4f}",
                         f"{p['kp_mean_px']:.3f}", f"{p['kp_median_px']:.3f}"])

with open(os.path.join(RES_DIR, 'direction_summary.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['split', 'decoder', 'count', 'mean_deg', 'median_deg', 'p90_deg',
                 'bad_rate_gt10deg', 'endpoint_unit_mean'])
    for s in ('linemod', 'occ'):
        for n in ('D0', 'D1'):
            st, d = DIR[s][n]['stats'], DIR[s][n]
            wr.writerow([s, n, st['count'], f"{st['mean']:.3f}", f"{st['median']:.3f}",
                         f"{st['p90']:.3f}", f"{d['bad']:.4f}", f"{d['endpoint']:.4f}"])

with open(os.path.join(RES_DIR, 'keypoint_summary.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['split', 'decoder'] + [f'kp{ i }_mean_px' for i in range(KP)])
    for s in ('linemod', 'occ'):
        for n in ('D0', 'D1'):
            wr.writerow([s, n] + [f'{x:.3f}' for x in POSE[(s, n)]['per_kp_mean_px']])

# decision (pre-registered). ADD/2D/5cm5deg are SUCCESS-RATE fractions
# (fraction of images correct, higher = better), so improvement is defined
# as the relative INCREASE of the success rate: (D1 - D0) / D0.
imp = {}
for s in ('linemod', 'occ'):
    d0f, d1f = POSE[(s, 'D0')]['add_frac'], POSE[(s, 'D1')]['add_frac']
    imp[s] = (d1f - d0f) / d0f if d0f > 0 else (0.0 if d1f == 0 else float('inf'))
if imp['occ'] >= GO_C_MIN:
    decision = 'GO-C'
elif imp['occ'] < STOP_C_MAX:
    decision = 'STOP-C'
else:
    decision = 'BORDERLINE'
log(f"POSE IMPROVEMENT  LIN {imp['linemod']:+.2%}  OCC {imp['occ']:+.2%}  "
    f"(ADD success-rate frac, (D1-D0)/D0; OCC gate >= {GO_C_MIN:.0%} GO-C / "
    f"< {STOP_C_MAX:.0%} STOP-C)")
log(f'DECISION: {decision}')

json.dump(dict(direction={s: {n: DIR[s][n] for n in ('D0', 'D1')} for s in ('linemod', 'occ')},
               direction_ref_exp013=DIR_REF,
               keypoint={s: {n: {k: POSE[(s, n)][k] for k in
                                 ('kp_mean_px', 'kp_median_px', 'per_kp_mean_px')}
                             for n in ('D0', 'D1')} for s in ('linemod', 'occ')},
               pose={f'{s}_{n}': POSE[(s, n)] for s in ('linemod', 'occ')
                     for n in ('D0', 'D1')},
               add_improvement=imp, decision=decision,
               thresholds=dict(go_c_min=GO_C_MIN, stop_c_max=STOP_C_MAX),
               identity_check=dict(max_seg_diff=max_seg, max_vertex_diff=max_ver,
                                   mask_identical=mask_all_eq, max_kp_diff=max_kp),
               vote_params=dict(round_hyp_num=ROUND_HYP, inlier_thresh=INLIER_THRESH,
                                max_num=MAX_NUM, vote_type=str(VOTE_TYPE)),
               runtime_s=time.time() - T0),
          open(os.path.join(RES_DIR, 'final_summary.json'), 'w'), indent=2)
log(f'DONE in {time.time()-T0:.1f}s')
