"""EXP013: Held-out Frozen Decoder Generalization Validation.

Research question (EXP012 follow-up): EXP012 showed that on the fixed 20+20
images a frozen 32-ch shared feature F plus a cross-channel 3x3 readout (D1)
improves OCC direction error 5.121 -> 4.040 deg (+21.1%), and judged GO-A.
But EXP012 was an IN-SAMPLE capacity probe: D1 was fitted and evaluated on
the same images. The only question here: does that improvement GENERALIZE to
images never used for decoder fitting?

Models (nothing redesigned, EXP012 final implementation verbatim):
  D0: original PVNet convraw[3] 1x1 vertex head, frozen (baseline).
  D1: D0 + zero-init [Conv3x3(32->32) -> LeakyReLU(0.1) -> Conv1x1(32->18)]
      (EXP012 ResidualDecoder, 9842 params), fitted with the identical
      protocol: offline on cached frozen features, 5 epochs over all fg
      pixels (pixel minibatch 4096, Adam lr=1e-3, seed 0 re-seeded per fit),
      FINAL decoder state used (no checkpoint selection). Fitted separately
      per split: D1-LIN on the 20 LIN FIT images, D1-OCC on the 20 OCC FIT
      images (EXP012 convention).

Data protocol:
  FIT set      = EXP012's exact 20 LIN + 20 OCC images (EXP005 IDs, asserted
                 against experiments/exp005_visibility_diagnosis results).
  HELD-OUT set = the NEXT 20 LIN + 20 OCC images of the same deterministic
                 dataset listings (LIN val_real_set[20:40]; OCC first half of
                 test_real_set[20:40]). Asserted disjoint from FIT; fit and
                 held-out image id lists are saved to results/image_ids.json.
  Held-out images NEVER enter the optimizer, gradients, early stopping,
  loss-based model selection, or any parameter update; they are only
  forwarded under no_grad after D1 is frozen.

D0 gate (before any fitting): D0 on the FIT set must reproduce EXP006/012
  (LIN 2.969 / OCC 5.121, counts 440940/250033); otherwise abort. D0 on the
  held-out set is expected to differ (different images) and is recorded, not
  gated.

Evaluation (direction representation only, EXP006/008 pool): visible fg
  pixels with |K-p|>=1px, all 9 kps pooled: mean/median/p90 deg, bad rate
  >10 deg, endpoint error (unit space); OCC proximity bands 0-5 / 5-20 / 20+
  px from the occluded region (amodal-mask definitions) on BOTH the FIT and
  HELD-OUT OCC sets. No ADD / ADD-S / PnP / RANSAC-pose.

Pre-registered decision (thresholds fixed in advance):
  HELD-OUT OCC improvement = (D0_occ - D1_occ) / D0_occ on held-out images
  >= 15%  -> GO-B      (EXP012 improvement generalizes)
  <  5%   -> STOP-B    (in-sample capacity fitting; not a generalizable
                        mechanism; no further decoder complexity allowed)
  else    -> BORDERLINE (report gap, await supervisor decision)

Run from repo root:
  python experiments/exp013_heldout_decoder_generalization/run_exp013.py
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
EPOCHS = 5                     # protocol cap (EXP012 fitting protocol)
LR = 1e-3
SEED = 0
EPS = 1e-9
LIN_BASE, OCC_BASE = 2.969, 5.121            # EXP006/012 reference (FIT gate)
LIN_COUNT_REF, OCC_COUNT_REF = 440940, 250033
EXP012_D1_FIT = {'lin': 2.841569903056655, 'occ': 4.0399292029470795}  # context only
GO_B_MIN, STOP_B_MAX = 0.15, 0.05           # pre-registered thresholds
BANDS = ((0.0, 5.0), (5.0, 20.0), (20.0, np.inf))

torch.manual_seed(SEED); np.random.seed(SEED)
T0 = time.time()
LOG = open(os.path.join(RES_DIR, 'run.log'), 'w', buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + '\n')

json.dump(dict(cls='cat', n_img_per_set=N_IMG, ckpt='data/model/cat_linemod_train/199.pth',
               frozen='entire PVNet frozen (backbone, convraw incl. original 1x1 vertex '
                      'head, BN stats); no_grad forward; requires_grad=False',
               decoders=dict(
                   D0='original convraw[3] 1x1 head (frozen), channels [seg_dim:]',
                   D1='EXP012 final decoder verbatim: D0 + zero-init [Conv3x3(32->32) -> '
                      'LeakyReLU(0.1) -> Conv1x1(32->18)], 9842 params'),
               data=dict(fit='EXP012 20+20 images (EXP005 IDs, asserted)',
                         heldout='next 20 LIN (val_real_set[20:40]) + 20 OCC '
                                 '(test_real_set first half [20:40]) of the same '
                                 'deterministic listings; disjoint asserted'),
               fitting='D1 fitted ONLY on FIT caches; 5 epochs over all fg pixels '
                       '(pixel minibatch 4096, Adam lr=1e-3, seed 0 re-seeded per fit); '
                       'FINAL state used, no checkpoint/model selection; held-out never '
                       'enters optimizer/gradients/selection',
               loss='lib.utils.net_utils.smooth_l1_loss(unit-normalized pred, dataset GT '
                    'vertex targets, fg vertex weights, normalize=True) - EXP012 verbatim',
               eval='direction error (deg) on EXP006/008 pool (visible fg, |K-p|>=1px, '
                    '9 kps pooled): mean/median/p90, bad rate >10 deg, endpoint (unit '
                    'space); OCC proximity bands 0-5/5-20/20+ px on FIT and HELD-OUT',
               d0_gate=f'D0 on FIT must reproduce EXP006/012: |LIN-{LIN_BASE}|<0.05 and '
                       f'|OCC-{OCC_BASE}|<0.05 and exact counts, else abort',
               go_stop=f'HELD-OUT OCC improvement >= {GO_B_MIN:.0%} -> GO-B; '
                       f'< {STOP_B_MAX:.0%} -> STOP-B; else BORDERLINE'),
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


def build_loader(split, lo, hi, expect=None):
    """expect=None -> held-out slice: assert every rgb NOT in the FIT set."""
    if split == 'linemod':
        db = LineModImageDB('cat', has_render_set=False, has_fuse_set=False)
        db_set, prefix = db.val_real_set[lo:hi], cfg.LINEMOD
    else:
        db = OcclusionLineModImageDB('cat')
        half = db.test_real_set[:len(db.test_real_set) // 2]
        db_set, prefix = half[lo:hi], cfg.OCCLUSION_LINEMOD
    ds = LineModDatasetRealAug(db_set, prefix, VotingType.Farthest,
                               augment=False, cfg=AUG_CFG)
    ids = []
    for i in range(len(ds)):
        rgb = os.path.basename(ds.imagedb[i]['rgb_pth'])
        ids.append(rgb)
        if expect is not None:
            assert rgb == expect[i], f'FIT mismatch {split}[{lo+i}]: {rgb}'
        else:
            assert rgb not in EXP5_RGB[split], f'LEAK: held-out {rgb} is a FIT image'
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'tle', os.path.join(ROOT, 'tools', 'train_linemod_exp001.py'))
    tle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tle)
    loader = DataLoader(ds, batch_sampler=tle.ImageSizeBatchSampler(
        SequentialSampler(ds), 1, False, AUG_CFG), num_workers=6)
    return ds, loader, ids


def angle_deg(a, b):
    dot = np.clip((a * b).sum(-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def mstats(a):
    a = np.asarray(a, np.float64); a = a[np.isfinite(a)]
    return dict(count=int(a.size), mean=float(a.mean()) if a.size else float('nan'),
                median=float(np.median(a)) if a.size else float('nan'),
                p90=float(np.quantile(a, 0.9)) if a.size else float('nan'))


def cache_split(split, ds, loader, id_offset):
    """One frozen forward per image: F, GT vertex targets, fg weights, pool, bands."""
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
            mid = re.findall(r'(\d+)', os.path.basename(ds.imagedb[i]['rgb_pth']))[0]
            ap = os.path.join(AMODAL_DIR['occ'], str(int(mid)) + '.png')
            if os.path.exists(ap):
                amodal = np.asarray(Image.open(ap)) > 0
                occ_region = amodal & ~vis
                if occ_region.any():
                    dist = cv2.distanceTransform((~occ_region).astype(np.uint8),
                                                 cv2.DIST_L2, 3)
                    d_occ = dist[ys, xs].astype(np.float64)
        items.append(dict(image_id=id_offset + i, feat=feat, vertex_gt=vertex_gt,
                          vw=vw, ys=ys, xs=xs, ok=ok, u_gt=u_gt, d_occ=d_occ, vis=vis))
    return items


# ---------------- build FIT and HELD-OUT caches (disjoint asserted) ----------------
IMAGE_IDS = {'fit': {}, 'heldout': {}}
CACHE = {'fit': {}, 'heldout': {}}
for split in ('linemod', 'occ'):
    ds, loader, ids = build_loader(split, 0, N_IMG, expect=EXP5_RGB[split])
    IMAGE_IDS['fit'][split] = dict(img_ids=list(range(N_IMG)), rgb=ids)
    CACHE['fit'][split] = cache_split(split, ds, loader, 0)
    ds, loader, ids = build_loader(split, N_IMG, 2 * N_IMG)
    IMAGE_IDS['heldout'][split] = dict(img_ids=list(range(N_IMG, 2 * N_IMG)), rgb=ids)
    CACHE['heldout'][split] = cache_split(split, ds, loader, N_IMG)
    fids, hids = IMAGE_IDS['fit'][split]['rgb'], IMAGE_IDS['heldout'][split]['rgb']
    log(f'--- {split}: FIT [{fids[0]}..{fids[-1]}] + HELD-OUT [{hids[0]}..{hids[-1]}] cached ---')

for split in ('linemod', 'occ'):
    fit_set, held_set = set(IMAGE_IDS['fit'][split]['rgb']), set(IMAGE_IDS['heldout'][split]['rgb'])
    assert fit_set.isdisjoint(held_set), f'FIT/HELD-OUT overlap in {split}'
IMAGE_IDS['disjoint_assert'] = {s: bool(set(IMAGE_IDS['fit'][s]['rgb']).isdisjoint(
    set(IMAGE_IDS['heldout'][s]['rgb']))) for s in ('linemod', 'occ')}
json.dump(IMAGE_IDS, open(os.path.join(RES_DIR, 'image_ids.json'), 'w'), indent=2)
log(f'FIT/HELD-OUT disjoint asserted; ids saved ({time.time()-T0:.1f}s)')


# ---------------- decoders (EXP012 verbatim) ----------------
def d0_decode(feat):
    """D0: original frozen 1x1 vertex head on the shared feature."""
    with torch.no_grad():
        return net.convraw[3](feat)[:, net.seg_dim:]


class ResidualDecoder(nn.Module):
    """EXP012 final D1 verbatim: D0(F) + zero-init Conv3x3(32->32) ->
    LeakyReLU(0.1) -> Conv1x1(32->18). At init the branch outputs exactly 0,
    so the decoder IS D0 (EXP009-style control)."""

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


def evaluate(caches, fns_by_split):
    """Direction metrics on the EXP006/008 pool; fns_by_split[split] = {name: fn}."""
    out = {}
    for split in ('linemod', 'occ'):
        decode_fns = fns_by_split[split]
        res = {n: {'ang': [], 'end': []} for n in decode_fns}
        bands = {n: {bi: [] for bi in range(len(BANDS))} for n in decode_fns}
        band_img = {n: {bi: set() for bi in range(len(BANDS))} for n in decode_fns}
        for item in caches[split]:
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
                if split == 'occ' and item['d_occ'] is not None:
                    d_occ = item['d_occ']
                    for bi, (lo, hi) in enumerate(BANDS):
                        sel = ok & (d_occ >= lo)[:, None] & (d_occ < hi)[:, None]
                        if sel.any():
                            bands[name][bi].append(ang[sel])
                            band_img[name][bi].add(item['image_id'])
        out[split] = dict(res=res, bands=bands, band_img=band_img)
    return out


# ---------------- D0 gate on FIT (must reproduce EXP006/012 before fitting) ----------------
d0_fit = evaluate(CACHE['fit'], {s: {'D0': d0_decode} for s in ('linemod', 'occ')})
d0_fit_lin = mstats(np.concatenate(d0_fit['linemod']['res']['D0']['ang']))
d0_fit_occ = mstats(np.concatenate(d0_fit['occ']['res']['D0']['ang']))
log(f"D0 gate (FIT): LIN {d0_fit_lin['mean']:.3f} (ref {LIN_BASE})  "
    f"OCC {d0_fit_occ['mean']:.3f} (ref {OCC_BASE})  counts "
    f"{d0_fit_lin['count']}/{d0_fit_occ['count']} (ref {LIN_COUNT_REF}/{OCC_COUNT_REF})")
assert abs(d0_fit_lin['mean'] - LIN_BASE) < 0.05 and abs(d0_fit_occ['mean'] - OCC_BASE) < 0.05, \
    'D0 does not reproduce EXP006/012 on FIT -> STOP before fitting'
assert d0_fit_lin['count'] == LIN_COUNT_REF and d0_fit_occ['count'] == OCC_COUNT_REF, \
    'FIT pool counts differ from EXP006/012 -> STOP'
log('D0 FIT gate passed')

# D0 on HELD-OUT (recorded only; different images, values expected to differ)
d0_held = evaluate(CACHE['heldout'], {s: {'D0': d0_decode} for s in ('linemod', 'occ')})
d0_held_lin = mstats(np.concatenate(d0_held['linemod']['res']['D0']['ang']))
d0_held_occ = mstats(np.concatenate(d0_held['occ']['res']['D0']['ang']))
log(f"D0 (HELD-OUT, recorded): LIN {d0_held_lin['mean']:.3f}  OCC {d0_held_occ['mean']:.3f}  "
    f"counts {d0_held_lin['count']}/{d0_held_occ['count']}")


# ---------------- fit D1 on FIT caches only (EXP012 protocol verbatim) ----------------
PB = 4096   # pixel minibatch size (one epoch = one pass over ALL fg pixels)


def fit_d1(split):
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    dec = ResidualDecoder().to(device)
    n_par = sum(p.numel() for p in dec.parameters())
    init_loss = float(np.mean([full_image_loss(dec, it) for it in CACHE['fit'][split]]))
    d0_loss = float(np.mean([full_image_loss(d0_decode, it) for it in CACHE['fit'][split]]))
    log(f'fit D1-{split}: params {n_par}  init loss {init_loss:.6f}  '
        f'(D0 reference {d0_loss:.6f})')
    opt = torch.optim.Adam(dec.parameters(), lr=LR)
    dec.train()
    tl = [[0, init_loss]]
    for ep in range(EPOCHS):
        tot, nb = 0.0, 0
        for item in CACHE['fit'][split]:
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
        log(f'fit D1-{split} epoch {ep+1}/{EPOCHS} loss {tot/max(nb,1):.6f}  '
            f't={time.time()-T0:.0f}s')
    dec.eval()
    return dec, n_par, tl


MODELS, n_params, fit_logs = {}, {}, {}
for split in ('linemod', 'occ'):
    MODELS[split], n_params[split], fit_logs[split] = fit_d1(split)
# D1 now FROZEN; held-out is only forwarded from here on (no_grad in evaluate).
for split in ('linemod', 'occ'):
    for p in MODELS[split].parameters():
        p.requires_grad_(False)


# ---------------- full evaluation: D0/D1 on FIT and HELD-OUT ----------------
def make_fns(split):
    return {'D0': d0_decode,
            'D1': (lambda f, m=MODELS[split]: m(f))}


FULL = {'fit': evaluate(CACHE['fit'], {s: make_fns(s) for s in ('linemod', 'occ')}),
        'heldout': evaluate(CACHE['heldout'], {s: make_fns(s) for s in ('linemod', 'occ')})}

# ---------------- CSV outputs ----------------
with open(os.path.join(RES_DIR, 'direction_summary.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['set', 'split', 'decoder', 'count', 'mean_deg', 'median_deg', 'p90_deg',
                 'bad_rate_gt10deg', 'endpoint_unit_mean'])
    for set_name in ('fit', 'heldout'):
        for s in ('linemod', 'occ'):
            for name in ('D0', 'D1'):
                ang = np.concatenate(FULL[set_name][s]['res'][name]['ang'])
                end = np.concatenate(FULL[set_name][s]['res'][name]['end'])
                ms = mstats(ang)
                wr.writerow([set_name, s, name, ms['count'], f"{ms['mean']:.3f}",
                             f"{ms['median']:.3f}", f"{ms['p90']:.3f}",
                             f"{float((ang > 10).mean()):.4f}", f"{end.mean():.4f}"])

with open(os.path.join(RES_DIR, 'occ_proximity.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['dist_to_occluded_region_px', 'n_images_fit', 'n_entries_fit',
                 'fit_D0_mean_deg', 'fit_D1_mean_deg',
                 'n_images_heldout', 'n_entries_heldout',
                 'heldout_D0_mean_deg', 'heldout_D1_mean_deg'])
    for bi, (lo, hi) in enumerate(BANDS):
        hi_s = f'{hi:g}' if np.isfinite(hi) else 'inf'
        row = [f'{lo:g}-{hi_s}']
        for set_name in ('fit', 'heldout'):
            b = FULL[set_name]['occ']['bands']
            if not b['D0'][bi]:
                row += [0, 0, '', '']
            else:
                row += [len(FULL[set_name]['occ']['band_img']['D0'][bi]),
                        len(np.concatenate(b['D0'][bi])),
                        f"{np.concatenate(b['D0'][bi]).mean():.3f}",
                        f"{np.concatenate(b['D1'][bi]).mean():.3f}"]
        wr.writerow(row)

with open(os.path.join(RES_DIR, 'train_log.csv'), 'w', newline='') as f:
    wr = csv.writer(f); wr.writerow(['fit', 'epoch', 'train_loss'])
    for split, tl in fit_logs.items():
        for ep, ls in tl:
            wr.writerow([f'D1-{split}', ep, f'{ls:.6f}'])

# ---------------- decision (pre-registered thresholds) ----------------
occ_mean = {set_name: {n: mstats(np.concatenate(FULL[set_name]['occ']['res'][n]['ang']))['mean']
                       for n in ('D0', 'D1')}
            for set_name in ('fit', 'heldout')}
lin_mean = {set_name: {n: mstats(np.concatenate(FULL[set_name]['linemod']['res'][n]['ang']))['mean']
                       for n in ('D0', 'D1')}
            for set_name in ('fit', 'heldout')}
imp_fit = (occ_mean['fit']['D0'] - occ_mean['fit']['D1']) / occ_mean['fit']['D0']
imp_held = (occ_mean['heldout']['D0'] - occ_mean['heldout']['D1']) / occ_mean['heldout']['D0']
gap = imp_fit - imp_held
if imp_held >= GO_B_MIN:
    decision = 'GO-B'
elif imp_held < STOP_B_MAX:
    decision = 'STOP-B'
else:
    decision = 'BORDERLINE'

for set_name in ('fit', 'heldout'):
    log(f"RESULT [{set_name.upper():8s}] LIN  D0 {lin_mean[set_name]['D0']:.3f}  "
        f"D1 {lin_mean[set_name]['D1']:.3f}")
    log(f"RESULT [{set_name.upper():8s}] OCC  D0 {occ_mean[set_name]['D0']:.3f}  "
        f"D1 {occ_mean[set_name]['D1']:.3f}")
log(f'FIT improvement     {imp_fit:+.2%} (EXP012 reference +21.11%)')
log(f'HELD-OUT improvement {imp_held:+.2%}')
log(f'generalization gap   {gap:+.2%}')
log(f'DECISION: {decision} (GO-B >= {GO_B_MIN:.0%} held-out; STOP-B < {STOP_B_MAX:.0%})')

json.dump(dict(d0_gate=dict(lin=d0_fit_lin['mean'], occ=d0_fit_occ['mean'],
                            count_lin=d0_fit_lin['count'], count_occ=d0_fit_occ['count']),
               d0_heldout=dict(lin=d0_held_lin['mean'], occ=d0_held_occ['mean'],
                               count_lin=d0_held_lin['count'],
                               count_occ=d0_held_occ['count']),
               mean={set_name: {s: {n: mstats(np.concatenate(FULL[set_name][s]['res'][n]['ang']))['mean']
                                for n in ('D0', 'D1')}
                           for s in ('linemod', 'occ')}
                    for set_name in ('fit', 'heldout')},
               fit_improvement_occ=imp_fit, heldout_improvement_occ=imp_held,
               generalization_gap=gap, decision=decision,
               thresholds=dict(go_b_min=GO_B_MIN, stop_b_max=STOP_B_MAX),
               exp012_reference=dict(d1_fit_lin=EXP012_D1_FIT['lin'],
                                     d1_fit_occ=EXP012_D1_FIT['occ']),
               n_params=n_params,
               runtime_s=time.time() - T0),
          open(os.path.join(RES_DIR, 'final_summary.json'), 'w'), indent=2)
log(f'DONE in {time.time()-T0:.1f}s')
