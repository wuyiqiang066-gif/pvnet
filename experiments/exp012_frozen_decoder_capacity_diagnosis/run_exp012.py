"""EXP012: Frozen Shared-Feature Decoder Capacity Diagnosis (information-locating).

Research question (EXP006/008/009/010/011 follow-up):
  EXP008: GT directions of visible pixels contain near-exact keypoint
          information (global oracle 0.001 deg).
  EXP009/010/011: simple local/global aggregation of the PREDICTED direction
          field cannot recover it (frozen local residual degrades OCC;
          validity-aware selection is worse; predicted-ray global recovery
          gives only 3.5%).
  Remaining question: is the occlusion-robust direction information still
  present in the frozen 32-ch shared feature F (convraw[0:3] output), with
  the original 1x1 vertex head merely a capacity bottleneck (A) -- or is it
  already lost at feature-extraction time (B)?

Decoders (all consume the identical frozen F; only the 1x1 vertex head is
replaced; segmentation path untouched):
  D0: original PVNet vertex head convraw[3] (1x1, 32->18), frozen. Baseline.
  D1: Conv3x3(32->32) -> LeakyReLU(0.1) -> Conv1x1(32->18)          (914 params)
  D2: Conv3x3 -> LeakyReLU(0.1) -> Conv3x3 -> LeakyReLU(0.1)
      -> Conv1x1(32->18)                                          (1234 params)
  LeakyReLU(0.1) is the network's own activation; without any nonlinearity
  D2 would collapse to a linear map equivalent to D1. No BN / SE / attention /
  deformable / multi-scale / transformer components.

Offline fitting (Plan A, preferred by protocol): one frozen forward per image
caches F, the dataset GT vertex targets and fg weights; decoders are then
fitted on the cached features only. Capacity-probe protocol: D1/D2 are fitted
IN-SAMPLE on the fixed 20 images of each split (D1-LIN fitted on the 20 LIN
images and evaluated on the same 20 LIN; D1-OCC fitted on the 20 OCC images
and evaluated on the same 20 OCC; likewise D2). In-sample fitting deliberately
removes generalization and domain-shift confounds (EXP009 lesson): the
experiment asks ONLY whether a better decoder can extract more direction
information from F, not whether it generalizes.

EXP007-trap control (warm start): D1/D2 are defined as D0(F) PLUS a
zero-initialized 3x3(->3x3) -> 1x1 branch (the EXP009 control strategy), so
each decoder is EXACTLY the D0 function at initialization and its in-sample
loss can only decrease from the baseline's; optimization budget therefore
cannot manufacture a spurious degradation, and any in-sample improvement over
D0 is attributable to the added 3x3 receptive-field capacity. Training: 5
epochs over all fg pixels (pixel minibatch 4096, Adam 1e-3, seed 0 re-seeded
per fit). Loss = the EXP009 direction loss aligned with the original vertex
supervision: lib.utils.net_utils.smooth_l1_loss(unit-normalized pred, dataset
GT vertex targets, fg vertex weights, normalize=True; per-fg-element mean).
Backbone / original vertex head / segmentation / GT definitions untouched;
199.pth unchanged.

Evaluation (direction representation only): EXP006/008 pool (visible fg
pixels with |K-p|>=1px, all 9 kps pooled): mean/median/p90 deg, bad rate
>10 deg, endpoint error in unit space; OCC proximity bands 0-5 / 5-20 / 20+
px from the occluded region (EXP005/011 amodal-mask definitions); field
roughness (local neighbor variation of the unit field) as a smoothing check.
No ADD / ADD-S / PnP / RANSAC-pose; no keypoint-specific analysis.

D0 gate: D0 must reproduce EXP006 (LIN 2.969 / OCC 5.121, tol 0.05) BEFORE
any fitting; otherwise the run aborts immediately.

Pre-registered decision:
  GO-A         iff min(OCC mean of D1, OCC mean of D2) <= 0.8 x 5.121 = 4.097
               (>=20% relative improvement)
  BORDERLINE   if best OCC mean in (4.097, 4.609]   (10-20%)
  STOP-A       if best OCC mean > 4.609             (<10% improvement)

Run from repo root:
  python experiments/exp012_frozen_decoder_capacity_diagnosis/run_exp012.py
"""
import os, sys, json, re, csv, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from torch.utils.data import DataLoader, SequentialSampler

from lib.networks.model_repository import Resnet18_8s
from lib.datasets.linemod_dataset import LineModDatasetRealAug, VotingType
from lib.utils.data_utils import LineModImageDB, OcclusionLineModImageDB
from lib.utils.config import cfg
from lib.utils.net_utils import smooth_l1_loss

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(EXP_DIR, 'results')
os.makedirs(RES_DIR, exist_ok=True)

N_IMG = 20
KP = 9
EPOCHS = 5                     # protocol cap
LR = 1e-3
SEED = 0
EPS = 1e-9
LIN_BASE, OCC_BASE = 2.969, 5.121            # EXP006 reference
GO_OCC_MAX = 0.8 * OCC_BASE                  # 4.097 -> GO-A
STOP_OCC_MAX = 0.9 * OCC_BASE                # 4.609 -> below this is at best BORDERLINE
LIN_END_REF, OCC_END_REF = 0.058, 0.092      # EXP006 endpoint (unit space) reference
LIN_COUNT_REF, OCC_COUNT_REF = 440940, 250033  # EXP006/011 pooled entry counts
BANDS = ((0.0, 5.0), (5.0, 20.0), (20.0, np.inf))

torch.manual_seed(SEED); np.random.seed(SEED)
T0 = time.time()
LOG = open(os.path.join(RES_DIR, 'run.log'), 'w', buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + '\n')

json.dump(dict(cls='cat', n_img=N_IMG, ckpt='data/model/cat_linemod_train/199.pth',
               frozen='entire PVNet (backbone, conv8s/4s/2s, convraw incl. original 1x1 '
                      'vertex head, BN stats); no_grad forward; requires_grad=False',
               shared_feature='F = convraw[0:3](cat(fm, x)) (1,32,H,W), identical to EXP009',
               decoders=dict(
                   D0='original convraw[3] 1x1 head (frozen), channels [seg_dim:]',
                   D1='D0 + zero-init [Conv3x3(32->32) -> LeakyReLU(0.1) -> Conv1x1(32->18)]',
                   D2='D0 + zero-init [Conv3x3 -> LeakyReLU(0.1) -> Conv3x3 -> '
                      'LeakyReLU(0.1) -> Conv1x1(32->18)]'),
               fitting='offline on cached frozen features; IN-SAMPLE per split '
                       '(D1/D2 fitted on the fixed 20 images of the split and evaluated '
                       'on the same 20 images; capacity probe, no generalization claim); '
                       'EXP007-trap control: D1/D2 = D0 + zero-init 3x3(->3x3) -> 1x1 '
                       'branch, init = D0 exactly (EXP009 control strategy)',
               loss='lib.utils.net_utils.smooth_l1_loss(unit-normalized pred, dataset GT '
                    'vertex targets, fg vertex weights, normalize=True) - per-fg-element '
                    'mean (EXP009 direction-loss convention; GT supervision unchanged)',
               train='5 epochs over all fg pixels (pixel minibatch 4096, Adam lr=1e-3, '
                     'seed 0 re-seeded per fit, batch=1 image per step)',
               eval='direction error (deg) on EXP006/008 pool (visible fg, |K-p|>=1px, '
                    '9 kps pooled): mean/median/p90, bad rate >10 deg, endpoint error '
                    '(unit space); OCC proximity bands via amodal mask (EXP005/011); '
                    'field roughness (smoothing check)',
               d0_gate=f'D0 must reproduce EXP006: |LIN-{LIN_BASE}|<0.05 and '
                       f'|OCC-{OCC_BASE}|<0.05, else abort before fitting',
               go_stop=f'GO-A iff min(OCC D1, OCC D2) <= {GO_OCC_MAX:.3f} (>=20%); '
                       f'STOP-A if > {STOP_OCC_MAX:.3f} (<10%); else BORDERLINE'),
          open(os.path.join(RES_DIR, 'config.json'), 'w'), indent=2)

device = 'cuda'
net = Resnet18_8s(ver_dim=KP * 2, seg_dim=2)
sd = torch.load(os.path.join(ROOT, 'data/model/cat_linemod_train/199.pth'), map_location='cpu')
net.load_state_dict(sd['net'])
net = net.to(device).eval()
for p in net.parameters():
    p.requires_grad_(False)

AUG_CFG = json.load(open(os.path.join(
    ROOT, 'configs', 'exp001_reliability_voting_scratch.json')))['aug_cfg']
AMODAL_DIR = {'linemod': os.path.join(cfg.LINEMOD, 'cat', 'amodal_mask'),
              'occ': os.path.join(cfg.OCCLUSION_LINEMOD, 'amodal_masks', 'cat')}
EXP5_IDS = json.load(open(os.path.join(
    ROOT, 'experiments', 'exp005_visibility_diagnosis', 'results', 'image_ids.json')))
EXP5_RGB = {s: [r['rgb'] for r in EXP5_IDS['image_ids'][s]] for s in ('linemod', 'occ')}


def normalize_vertices(t):
    b = t.view(t.shape[0], KP, 2, *t.shape[2:])
    n = torch.clamp(b.pow(2).sum(2, keepdim=True).sqrt(), min=EPS)
    return (b / n).view(t.shape)


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


def angle_deg(a, b):
    dot = np.clip((a * b).sum(-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def mstats(a):
    a = np.asarray(a, np.float64); a = a[np.isfinite(a)]
    return dict(count=int(a.size), mean=float(a.mean()) if a.size else float('nan'),
                median=float(np.median(a)) if a.size else float('nan'),
                p90=float(np.quantile(a, 0.9)) if a.size else float('nan'))


# ---------------- offline feature cache (one frozen forward per image) ----------------
CACHE = {'linemod': [], 'occ': []}
for split in ('linemod', 'occ'):
    ds, loader = build_loader(split)
    log(f'--- {split}: caching frozen features for {len(ds)} images (IDs verified) ---')
    for image_id, data in enumerate(loader):
        image, mask_gt, vertex_gt, vw, pose, gt_kps = [d.to(device) for d in data]
        assert vertex_gt.shape[1] == KP * 2 and vw.shape[1] == 1
        with torch.no_grad():
            x2s, x4s, x8s, x16s, x32s, xfc = net.resnet18_8s(image)
            fm = net.conv8s(torch.cat([xfc, x8s], 1)); fm = net.up8sto4s(fm)
            fm = net.conv4s(torch.cat([fm, x4s], 1)); fm = net.up4sto2s(fm)
            fm = net.conv2s(torch.cat([fm, x2s], 1)); fm = net.up2storaw(fm)
            feat = net.convraw[0:3](torch.cat([fm, image], 1))   # (1,32,H,W)
        gt2 = (gt_kps[0, :, :2] if gt_kps.dim() == 3 else gt_kps[0, 0, :, :2]).cpu().numpy()
        vis = (mask_gt[0].cpu().numpy() == 1)
        ys, xs = np.nonzero(vis)
        coords = np.stack([xs, ys], 1).astype(np.float64)
        v_raw = gt2[None, :, :] - coords[:, None, :]
        d_raw = np.linalg.norm(v_raw, axis=2)
        ok = d_raw >= 1.0
        u_gt = v_raw / (d_raw[..., None] + EPS)
        d_occ = None
        if split == 'occ':
            mid = re.findall(r'(\d+)', os.path.basename(ds.imagedb[image_id]['rgb_pth']))[0]
            ap = os.path.join(AMODAL_DIR['occ'], str(int(mid)) + '.png')
            if os.path.exists(ap):
                amodal = np.asarray(Image.open(ap)) > 0
                occ_region = amodal & ~vis
                if occ_region.any():
                    dist = cv2.distanceTransform((~occ_region).astype(np.uint8),
                                                 cv2.DIST_L2, 3)
                    d_occ = dist[ys, xs].astype(np.float64)
        CACHE[split].append(dict(image_id=image_id, feat=feat, vertex_gt=vertex_gt,
                                 vw=vw, ys=ys, xs=xs, ok=ok, u_gt=u_gt,
                                 d_occ=d_occ, vis=vis))
log(f'features cached in {time.time()-T0:.1f}s')


# ---------------- decoders ----------------
def d0_decode(feat):
    """D0: original frozen 1x1 vertex head on the shared feature."""
    with torch.no_grad():
        return net.convraw[3](feat)[:, net.seg_dim:]


class ResidualDecoder(nn.Module):
    """D0(F) + zero-init branch: n3 x [Conv3x3(32->32) -> LeakyReLU(0.1)] -> 1x1.
    At init the branch outputs exactly 0, so the decoder IS D0 (EXP009-style
    control: fitting starts at the baseline and can only improve in-sample)."""

    def __init__(self, n3, cin=32, kp=KP):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(cin, cin, 3, padding=1, bias=True) for _ in range(n3)])
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.c1 = nn.Conv2d(cin, kp * 2, 1, bias=True)
        nn.init.zeros_(self.c1.weight)
        nn.init.zeros_(self.c1.bias)

    def forward(self, f):
        h = f
        for m in self.convs:
            h = self.act(m(h))
        return net.convraw[3](f)[:, net.seg_dim:] + self.c1(h)


def make_decoder(name):
    return ResidualDecoder(1 if name == 'D1' else 2)


def full_image_loss(dec, item):
    """In-sample loss under the EXP009 normalization (per-fg-element mean)."""
    with torch.no_grad():
        out = normalize_vertices(dec(item['feat']))
        return float(smooth_l1_loss(out, item['vertex_gt'], item['vw'],
                                    reduce=False)[0])


def evaluate(fns_by_split):
    """Direction metrics on the EXP006/008 pool; fns_by_split[split] = {name: fn}."""
    out = {}
    for split in ('linemod', 'occ'):
        decode_fns = fns_by_split[split]
        res = {n: {'ang': [], 'end': []} for n in decode_fns}
        bands = {n: {bi: [] for bi in range(len(BANDS))} for n in decode_fns}
        band_img = {n: {bi: set() for bi in range(len(BANDS))} for n in decode_fns}
        rough = {n: [] for n in decode_fns}
        for item in CACHE[split]:
            feat, vis = item['feat'], item['vis']
            ys, xs, ok, u_gt = item['ys'], item['xs'], item['ok'], item['u_gt']
            H, W = vis.shape
            for name, fn in decode_fns.items():
                with torch.no_grad():
                    v = normalize_vertices(fn(feat))
                ver = v.permute(0, 2, 3, 1).contiguous().view(1, H, W, KP, 2)[0]
                dirs = ver[ys, xs].cpu().numpy().astype(np.float64)
                m_pred = np.linalg.norm(dirs, axis=2)
                u_pred = dirs / (m_pred[..., None] + EPS)
                ang = angle_deg(u_pred, u_gt)
                res[name]['ang'].append(ang[ok])
                res[name]['end'].append(np.linalg.norm(u_pred - u_gt, axis=2)[ok])
                # field roughness (smoothing check): mean neighbor variation of unit field
                U = v[0].view(KP, 2, H, W).cpu().numpy()
                dxx = np.linalg.norm(U[:, :, :, 1:] - U[:, :, :, :-1], axis=1)  # (K,H,W-1)
                dyy = np.linalg.norm(U[:, :, 1:, :] - U[:, :, :-1, :], axis=1)  # (K,H-1,W)
                vx = vis[:, 1:] & vis[:, :-1]
                vy = vis[1:, :] & vis[:-1, :]
                vals = [dxx[:, vx].ravel(), dyy[:, vy].ravel()]
                vals = [v2 for v2 in vals if v2.size]
                rough[name].append(float(np.concatenate(vals).mean())
                                   if vals else float('nan'))
                if split == 'occ' and item['d_occ'] is not None:
                    d_occ = item['d_occ']
                    for bi, (lo, hi) in enumerate(BANDS):
                        sel = ok & (d_occ >= lo)[:, None] & (d_occ < hi)[:, None]
                        if sel.any():
                            bands[name][bi].append(ang[sel])
                            band_img[name][bi].add(item['image_id'])
        out[split] = dict(res=res, bands=bands, band_img=band_img, rough=rough)
    return out


# ---------------- D0 gate (must reproduce EXP006 before any fitting) ----------------
d0_only = evaluate({s: {'D0': d0_decode} for s in ('linemod', 'occ')})
d0_lin = mstats(np.concatenate(d0_only['linemod']['res']['D0']['ang']))
d0_occ = mstats(np.concatenate(d0_only['occ']['res']['D0']['ang']))
log(f"D0 gate: LIN {d0_lin['mean']:.3f} (ref {LIN_BASE})  OCC {d0_occ['mean']:.3f} "
    f"(ref {OCC_BASE})  counts {d0_lin['count']}/{d0_occ['count']} "
    f"(ref {LIN_COUNT_REF}/{OCC_COUNT_REF})")
assert abs(d0_lin['mean'] - LIN_BASE) < 0.05 and abs(d0_occ['mean'] - OCC_BASE) < 0.05, \
    'D0 does not reproduce EXP006 -> STOP before fitting'
assert d0_lin['count'] == LIN_COUNT_REF and d0_occ['count'] == OCC_COUNT_REF, \
    'pool counts differ from EXP006/011 -> STOP'
log('D0 gate passed; proceeding to decoder fitting')


# ---------------- offline fitting of D1/D2 (in-sample per split, 5 epochs) ----------------
PB = 4096   # pixel minibatch size (one epoch = one pass over ALL fg pixels)


def fit(name, split):
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    dec = make_decoder(name).to(device)
    n_par = sum(p.numel() for p in dec.parameters())
    init_loss = float(np.mean([full_image_loss(dec, it) for it in CACHE[split]]))
    d0_loss = float(np.mean([full_image_loss(d0_decode, it) for it in CACHE[split]]))
    log(f'fit {name}-{split}: params {n_par}  init loss {init_loss:.6f}  '
        f'(D0 reference {d0_loss:.6f})')
    opt = torch.optim.Adam(dec.parameters(), lr=LR)
    dec.train()
    tl = [[0, init_loss]]
    for ep in range(EPOCHS):
        tot, nb = 0.0, 0
        for item in CACHE[split]:
            gt, vw = item['vertex_gt'], item['vw']
            fg = (vw[0, 0] > 0).nonzero(as_tuple=False)          # (n_fg, 2)
            perm = rng.permutation(fg.shape[0])
            for s0 in range(0, fg.shape[0], PB):
                idx = perm[s0:s0 + PB]
                s = torch.zeros(1, 1, gt.shape[2], gt.shape[3], device=device)
                s[0, 0, fg[idx, 0], fg[idx, 1]] = 1.0
                out = normalize_vertices(dec(item['feat']))
                loss = smooth_l1_loss(out, gt, vw * s, reduce=False)[0]
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach()); nb += 1
        tl.append([ep + 1, tot / max(nb, 1)])
        log(f'fit {name}-{split} epoch {ep+1}/{EPOCHS} '
            f'loss {tot/max(nb,1):.6f}  t={time.time()-T0:.0f}s')
    dec.eval()
    return dec, n_par, tl


MODELS = {}
fit_logs, n_params = {}, {}
for name in ('D1', 'D2'):
    for split in ('linemod', 'occ'):
        key = f'{name}-{split}'
        MODELS[key], n_params[key], fit_logs[key] = fit(name, split)


# ---------------- full evaluation ----------------
def make_fns(split):
    return {'D0': d0_decode,
            'D1': (lambda f, m=MODELS[f'D1-{split}']: m(f)),
            'D2': (lambda f, m=MODELS[f'D2-{split}']: m(f))}


FULL = evaluate({s: make_fns(s) for s in ('linemod', 'occ')})

# ---------------- CSV outputs ----------------
with open(os.path.join(RES_DIR, 'direction_summary.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['split', 'decoder', 'count', 'mean_deg', 'median_deg', 'p90_deg',
                 'bad_rate_gt10deg', 'endpoint_unit_mean'])
    for s in ('linemod', 'occ'):
        for name in ('D0', 'D1', 'D2'):
            ang = np.concatenate(FULL[s]['res'][name]['ang'])
            end = np.concatenate(FULL[s]['res'][name]['end'])
            ms = mstats(ang)
            wr.writerow([s, name, ms['count'], f"{ms['mean']:.3f}", f"{ms['median']:.3f}",
                         f"{ms['p90']:.3f}", f"{float((ang > 10).mean()):.4f}",
                         f"{end.mean():.4f}"])

with open(os.path.join(RES_DIR, 'occ_proximity.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['dist_to_occluded_region_px', 'n_images', 'n_entries',
                 'D0_mean_deg', 'D1_mean_deg', 'D2_mean_deg'])
    for bi, (lo, hi) in enumerate(BANDS):
        hi_s = f'{hi:g}' if np.isfinite(hi) else 'inf'
        if not FULL['occ']['bands']['D0'][bi]:
            wr.writerow([f'{lo:g}-{hi_s}', 0, 0] + [''] * 3); continue
        wr.writerow([f'{lo:g}-{hi_s}', len(FULL['occ']['band_img']['D0'][bi]),
                     len(np.concatenate(FULL['occ']['bands']['D0'][bi]))] +
                    [f"{np.concatenate(FULL['occ']['bands'][n][bi]).mean():.3f}"
                     for n in ('D0', 'D1', 'D2')])

with open(os.path.join(RES_DIR, 'smoothing_check.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['split', 'decoder', 'field_roughness_mean'])
    for s in ('linemod', 'occ'):
        for name in ('D0', 'D1', 'D2'):
            wr.writerow([s, name, f"{float(np.nanmean(FULL[s]['rough'][name])):.5f}"])

with open(os.path.join(RES_DIR, 'train_log.csv'), 'w', newline='') as f:
    wr = csv.writer(f); wr.writerow(['fit', 'epoch', 'train_loss'])
    for key, tl in fit_logs.items():
        for ep, ls in tl:
            wr.writerow([key, ep, f'{ls:.6f}'])

with open(os.path.join(RES_DIR, 'fit_quality.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['split', 'decoder', 'in_sample_loss_full_image'])
    for s in ('linemod', 'occ'):
        for name, mod in (('D0', d0_decode), ('D1', MODELS[f'D1-{s}']),
                          ('D2', MODELS[f'D2-{s}'])):
            ls = float(np.mean([full_image_loss(mod, it) for it in CACHE[s]]))
            wr.writerow([s, name, f'{ls:.6f}'])

# ---------------- decision ----------------
occ_mean = {n: mstats(np.concatenate(FULL['occ']['res'][n]['ang']))['mean']
            for n in ('D0', 'D1', 'D2')}
lin_mean = {n: mstats(np.concatenate(FULL['linemod']['res'][n]['ang']))['mean']
            for n in ('D0', 'D1', 'D2')}
best_name = min(('D1', 'D2'), key=lambda n: occ_mean[n])
best_occ = occ_mean[best_name]
rel_imp = (OCC_BASE - best_occ) / OCC_BASE
if best_occ <= GO_OCC_MAX:
    decision = 'GO-A'
elif best_occ > STOP_OCC_MAX:
    decision = 'STOP-A'
else:
    decision = 'BORDERLINE'
log(f"RESULT  LIN  D0 {lin_mean['D0']:.3f}  D1 {lin_mean['D1']:.3f}  "
    f"D2 {lin_mean['D2']:.3f}")
log(f"RESULT  OCC  D0 {occ_mean['D0']:.3f}  D1 {occ_mean['D1']:.3f}  "
    f"D2 {occ_mean['D2']:.3f}")
log(f'BEST {best_name}: OCC {best_occ:.3f} ({rel_imp:+.2%} vs baseline {OCC_BASE}); '
    f'GO<={GO_OCC_MAX:.3f} < BORDERLINE <= {STOP_OCC_MAX:.3f} < STOP')
log(f'DECISION: {decision}')

json.dump(dict(baseline=dict(lin=d0_lin['mean'], occ=d0_occ['mean'],
                             endpoint_lin=float(np.concatenate(
                                 FULL['linemod']['res']['D0']['end']).mean()),
                             endpoint_occ=float(np.concatenate(
                                 FULL['occ']['res']['D0']['end']).mean())),
                   lin_mean=lin_mean, occ_mean=occ_mean,
                   best=best_name, rel_improvement=rel_imp,
                   decision=decision, go_occ_max=GO_OCC_MAX,
                   stop_occ_max=STOP_OCC_MAX,
                   n_params=n_params,
                   counts={s: mstats(np.concatenate(FULL[s]['res']['D0']['ang']))['count']
                           for s in ('linemod', 'occ')},
                   runtime_s=time.time() - T0),
              open(os.path.join(RES_DIR, 'final_summary.json'), 'w'), indent=2)
log(f'DONE in {time.time()-T0:.1f}s')
