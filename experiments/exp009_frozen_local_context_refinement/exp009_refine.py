"""EXP009: Frozen Local Context Direction Refinement (mechanism validation).

Question (EXP008 follow-up): EXP008 showed the GT direction field is locally
highly redundant (r=3 oracle 0.569 deg on OCC vs 5.121 deg baseline). Can a
TINY learnable local-context residual module exploit that redundancy while
the entire PVNet stays frozen?

Design:
  frozen PVNet 199.pth (backbone + heads + BN running stats; no_grad forward,
  requires_grad=False) produces the 32-ch shared feature F (convraw[0:3]
  output, full resolution) and the frozen vertex head output v_base.
  Trainable residual module:
      dv = PW1x1( DW3x3(F) )        # 3x3 depthwise (32 ch) -> 1x1 pointwise (32->18)
      v_refined = normalize(v_base + dv)
  Pointwise conv is ZERO-initialized so step 0 is exactly the baseline.
  Loss: the ORIGINAL PVNet vertex loss (lib.utils.net_utils.smooth_l1_loss,
  sigma=1, fg-mask weighted, normalize=True) applied to v_refined vs the
  dataset's GT vertex targets. No new supervision, no new loss terms.

Training: 20 LINEMOD cat images (EXP005/006/008 IDs, asserted), augment=False,
20 epochs, Adam lr=1e-3, seed=0, batch=1. Evaluation: the same 20 LIN + 20 OCC
images (asserted IDs), direction error vs GT (EXP006/008 pool: visible fg
pixels with |K-p|>=1px, all 9 kps pooled) plus OCC interior / normal_boundary
/ occlusion_boundary regions (EXP005/006 definitions). Control (199.pth) is
evaluated in-run through the identical evaluator.

Pre-registered GO/STOP: GO iff OCC mean <= 4.609 deg (>=10% relative
improvement) AND LIN mean <= 3.118 deg (<=5% relative degradation).

Run from repo root:
  python experiments/exp009_frozen_local_context_refinement/exp009_refine.py
"""
import os, sys, json, re, csv, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
EPOCHS = 20
LR = 1e-3
SEED = 0
EPS = 1e-9
LIN_BASE, OCC_BASE = 2.969, 5.121          # EXP006/008 reference
GO_OCC_MAX, GO_LIN_MAX = 4.609, 3.118      # pre-registered thresholds

torch.manual_seed(SEED); np.random.seed(SEED)
T0 = time.time()
LOG = open(os.path.join(RES_DIR, 'run.log'), 'w', buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + '\n')

json.dump(dict(cls='cat', n_img=N_IMG, ckpt='data/model/cat_linemod_train/199.pth',
               frozen='entire PVNet (backbone, conv8s/4s/2s, convraw incl. vertex head, '
                      'BN running stats); no_grad trunk forward; requires_grad=False',
               module='DW3x3(32,g=32) -> PW1x1(32->18); PW zero-init; '
                      'v_refined = normalize(v_base + dv)',
               n_params=32 * 9 + 32 + 18 * 32 + 18,
               loss='lib.utils.net_utils.smooth_l1_loss(sigma=1, vertex_weights=fg mask), '
                    'applied to v_refined vs dataset GT vertex targets',
               train='20 LIN images (EXP005 IDs), augment=False, 20 epochs, Adam lr=1e-3, seed=0',
               eval='direction error (deg), EXP006/008 pool (visible fg, |K-p|>=1px, 9 kps pooled); '
                    'OCC regions via amodal mask (EXP005/006 definitions); Control evaluated in-run',
               go_stop=f'GO iff OCC <= {GO_OCC_MAX} AND LIN <= {GO_LIN_MAX}'),
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


class LocalContextResidual(nn.Module):
    """3x3 depthwise -> 1x1 pointwise residual on the frozen 32-ch feature."""
    def __init__(self, cin=32, kp=KP):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=True)
        self.pw = nn.Conv2d(cin, kp * 2, 1, bias=True)
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

    def forward(self, feat):
        return self.pw(self.dw(feat))


@torch.no_grad()
def trunk_forward(x):
    """Frozen PVNet forward; returns shared 32-ch feature and frozen vertex head output."""
    x2s, x4s, x8s, x16s, x32s, xfc = net.resnet18_8s(x)
    fm = net.conv8s(torch.cat([xfc, x8s], 1)); fm = net.up8sto4s(fm)
    fm = net.conv4s(torch.cat([fm, x4s], 1)); fm = net.up4sto2s(fm)
    fm = net.conv2s(torch.cat([fm, x2s], 1)); fm = net.up2storaw(fm)
    feat = net.convraw[0:3](torch.cat([fm, x], 1))       # (1,32,H,W), LeakyReLU out
    v_base = net.convraw[3](feat)[:, net.seg_dim:]       # (1,18,H,W) frozen vertex head
    return feat, v_base


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


def region_labels(vis_mask, amodal_mask):
    """Same definitions as EXP005/006/008."""
    m = (vis_mask > 0)
    interior = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8),
                         iterations=5).astype(bool)
    out = {'interior': interior, 'boundary': m & ~interior}
    if amodal_mask is not None:
        occ_region = amodal_mask & ~m
        dist_occ = cv2.distanceTransform((~occ_region).astype(np.uint8), cv2.DIST_L2, 3)
        occl_b = m & (dist_occ <= 5)
        out['occl_boundary'] = occl_b
        out['interior'] = out['interior'] & ~occl_b
        out['boundary'] = out['boundary'] & ~occl_b
    else:
        out['occl_boundary'] = np.zeros_like(interior)
    return out


def angle_deg(a, b):
    dot = np.clip((a * b).sum(-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def evaluate(module=None):
    """Direction error with the EXP006/008 evaluator; module=None -> Control."""
    res = {'linemod': {'all': []},
           'occ': {'all': [], 'interior': [], 'normal_boundary': [], 'occlusion_boundary': []}}
    for split in ('linemod', 'occ'):
        ds, loader = build_loader(split)
        for image_id, data in enumerate(loader):
            image, mask_gt, vertex_gt, vw, pose, gt_kps = [d.to(device) for d in data]
            feat, v_base = trunk_forward(image)
            v = v_base if module is None else v_base + module(feat)
            v = normalize_vertices(v)
            b_, _, h, w = v.shape
            ver = v.permute(0, 2, 3, 1).contiguous().view(b_, h, w, -1, 2)[0].detach()

            gt2 = (gt_kps[0, :, :2] if gt_kps.dim() == 3 else gt_kps[0, 0, :, :2]).cpu().numpy()
            vis = (mask_gt[0].cpu().numpy() == 1)
            ys, xs = np.nonzero(vis)
            coords = np.stack([xs, ys], 1).astype(np.float64)
            dirs = ver[ys, xs].cpu().numpy().astype(np.float64)
            m_pred = np.linalg.norm(dirs, axis=2)
            u_pred = dirs / (m_pred[..., None] + EPS)

            v_raw = gt2[None, :, :] - coords[:, None, :]
            d_raw = np.linalg.norm(v_raw, axis=2)
            ok = d_raw >= 1.0
            u_gt = v_raw / (d_raw[..., None] + EPS)
            ang = angle_deg(u_pred, u_gt)

            res[split]['all'].append(ang[ok])
            if split == 'occ':
                mid = re.findall(r'(\d+)', os.path.basename(ds.imagedb[image_id]['rgb_pth']))[0]
                ap = os.path.join(AMODAL_DIR['occ'], str(int(mid)) + '.png')
                amodal = (np.asarray(Image.open(ap)) > 0) if os.path.exists(ap) else None
                regions = region_labels(vis, amodal)
                rmap = {'interior': regions['interior'], 'normal_boundary': regions['boundary'],
                        'occlusion_boundary': regions['occl_boundary']}
                for r, sel in rmap.items():
                    s = sel[ys, xs][:, None] & ok
                    if s.any():
                        res[split][r].append(ang[s])
    return {s: {r: np.concatenate(a) for r, a in regs.items()} for s, regs in res.items()}


def mstats(a):
    a = np.asarray(a, np.float64)
    return float(a.mean()), float(np.median(a)), float(np.quantile(a, 0.9))


def main():
    # ---------------- Control (199.pth, identical evaluator) ----------------
    control = evaluate(module=None)
    for s in ('linemod', 'occ'):
        m, md, p90 = mstats(control[s]['all'])
        ref = LIN_BASE if s == 'linemod' else OCC_BASE
        log(f'control {s}: mean {m:.3f} (ref {ref:.3f}, d={abs(m-ref):.4f}) '
            f'median {md:.3f} p90 {p90:.3f} count {control[s]["all"].size}')
        assert abs(m - ref) < 0.05, f'control mismatch vs EXP006/008 on {s}'

    # ---------------- train tiny residual module ----------------
    module = LocalContextResidual(cin=32, kp=KP).to(device)
    n_par = sum(p.numel() for p in module.parameters())
    log(f'trainable params: {n_par}')
    opt = torch.optim.Adam(module.parameters(), lr=LR)
    ds_train, loader_train = build_loader('linemod')

    train_log = []
    module.train()
    for epoch in range(EPOCHS):
        ep = 0.0; nb = 0
        for data in loader_train:
            image, mask_gt, vertex_gt, vw, pose, gt_kps = [d.to(device) for d in data]
            assert vertex_gt.shape[1] == KP * 2 and vw.shape[1] == 1
            feat, v_base = trunk_forward(image)
            v_ref = normalize_vertices(v_base + module(feat))
            loss = smooth_l1_loss(v_ref, vertex_gt, vw, reduce=False).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep += float(loss); nb += 1
        train_log.append([epoch + 1, ep / max(nb, 1)])
        log(f'epoch {epoch+1}/{EPOCHS}  loss {ep/max(nb,1):.6f}  t={time.time()-T0:.0f}s')
    module.eval()

    torch.save(module.state_dict(), os.path.join(RES_DIR, 'refine_module.pth'))

    # ---------------- evaluate EXP009 (same evaluator) ----------------
    exp9 = evaluate(module=module)

    with open(os.path.join(RES_DIR, 'direction_error_summary.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['split', 'method', 'count', 'mean_deg', 'median_deg', 'p90_deg'])
        for s in ('linemod', 'occ'):
            for label, store_ in (('199.pth_baseline', control[s]), ('exp009', exp9[s])):
                for region, a in store_.items():
                    m, md, p90 = mstats(a)
                    wr.writerow([s, f'{label}_{region}' if region != 'all' else label,
                                 a.size, f'{m:.3f}', f'{md:.3f}', f'{p90:.3f}'])

    with open(os.path.join(RES_DIR, 'occ_region_direction_error.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['region', 'control_mean_deg', 'exp009_mean_deg', 'rel_change'])
        for r in ('all', 'interior', 'normal_boundary', 'occlusion_boundary'):
            cm = mstats(control['occ'][r])[0]; em = mstats(exp9['occ'][r])[0]
            wr.writerow([r, f'{cm:.3f}', f'{em:.3f}', f'{(em - cm) / cm:+.4f}'])

    lin_m = mstats(exp9['linemod']['all'])[0]
    occ_m = mstats(exp9['occ']['all'])[0]
    lin_rel = (lin_m - LIN_BASE) / LIN_BASE
    occ_rel = (occ_m - OCC_BASE) / OCC_BASE
    go = (occ_m <= GO_OCC_MAX) and (lin_m <= GO_LIN_MAX)
    decision = 'GO' if go else 'STOP'
    log(f'RESULT  LIN {lin_m:.3f} (baseline {LIN_BASE}, rel {lin_rel:+.2%})  '
        f'OCC {occ_m:.3f} (baseline {OCC_BASE}, rel {occ_rel:+.2%})')
    log(f'DECISION: {decision}  (GO iff OCC <= {GO_OCC_MAX} and LIN <= {GO_LIN_MAX})')

    with open(os.path.join(RES_DIR, 'train_log.csv'), 'w', newline='') as f:
        wr = csv.writer(f); wr.writerow(['epoch', 'train_loss'])
        wr.writerows(train_log)
    json.dump(dict(lin_mean=lin_m, occ_mean=occ_m, lin_rel=lin_rel, occ_rel=occ_rel,
                   decision=decision, go_occ_max=GO_OCC_MAX, go_lin_max=GO_LIN_MAX,
                   runtime_s=time.time() - T0, n_params=n_par),
              open(os.path.join(RES_DIR, 'final_summary.json'), 'w'), indent=2)
    log(f'DONE in {time.time()-T0:.1f}s')


if __name__ == '__main__':
    main()
