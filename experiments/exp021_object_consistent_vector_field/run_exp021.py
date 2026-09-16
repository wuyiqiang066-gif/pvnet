"""EXP021: Object-Consistent Vector Field (OCVF) — one-shot minimal hypothesis test.

Research question (pre-registered in config.json, written BEFORE any forward):
  PVNet regresses the K keypoint direction fields approximately independently
  per pixel. Hypothesis: the K fields share one object-level geometric state;
  making that state explicit and SHARED across the K readouts (cross-keypoint
  structural coupling) reduces correlated direction errors under occlusion.

  A = original frozen vertex head (no training; task spec: "do not retrain the
      baseline").
  B = OCVF: v'_k(p) = normalize(v0_k(p) + dV_k(p)); ONLY the OCVF module is
      trained (trunk / seg / original head frozen). EXP012 fitting budget:
      5 epochs, Adam 1e-3, seed 0, in-sample per split (EXP012 convention).

  Losses: L_dir (original PVNet direction supervision, EXP012 formula)
        + 1.0 * L_conf (configuration head trained toward GT keypoint TARGETS)
        + 1.0 * L_struct (all 9 fields pulled toward ONE configuration
          hypothesis chat predicted from the SINGLE shared object latent z).

  M6 structural-consistency metric (GT-free): per-keypoint LS ray-intersection
  point from the field's own rays; perpendicular residual of rays to that
  point; per-image median over 9 kps; per-split median over 20 images.

  GO iff ALL of: C1 OCC dir mean >=10% better; C2 LIN dir mean <=5% worse;
  C3 M6 OCC median >=5% better AND LIN median <=5% worse; C4 OCC voted-kp
  pooled median <= 1.05x A. Otherwise STOP (MIXED profile reported if 1-3
  criteria hold; still STOP).

  Gates (abort before training): G1 A reproduces EXP006 (2.969/5.121, counts
  440940/250033); G2 A init L_dir matches EXP012 (0.002869/0.008945);
  G3 init identity B==A (field exact, voting <1e-3 px); G4 no-GT-input
  verification; G5 A voting medians match EXP015 (1.111/3.150 px).

Run from repo root (pvnet env):
  python experiments/exp021_object_consistent_vector_field/run_exp021.py
"""
import os, sys, json, re, csv, time, inspect
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
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v3

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(EXP_DIR, 'results')
os.makedirs(RES_DIR, exist_ok=True)

N_IMG, KP = 20, 9
EPOCHS, LR, SEED = 5, 1e-3, 0            # EXP012 budget; HARD CAP
EPS = 1e-9
L_CONF_W, L_STRUCT_W = 1.0, 1.0          # fixed before the run; no tuning allowed
LIN_BASE, OCC_BASE = 2.969, 5.121        # EXP006 reference (G1)
LIN_CNT, OCC_CNT = 440940, 250033        # EXP006/011 pooled counts (G1)
LIN_LOSS_REF, OCC_LOSS_REF = 0.002869, 0.008945   # EXP012 init L_dir (G2)
LIN_KP_REF, OCC_KP_REF = 1.111, 3.150    # EXP015 A voting pooled medians (G5)
BANDS = ((0.0, 5.0), (5.0, 20.0), (20.0, np.inf))
ROUND_HYP, MAX_NUM, INLIER_THRESH = 128, 100, 0.99   # EXP015 verbatim

torch.manual_seed(SEED); np.random.seed(SEED)
T0 = time.time()
LOG = open(os.path.join(RES_DIR, 'run.log'), 'w', buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + '\n')

CFG = json.load(open(os.path.join(EXP_DIR, 'config.json')))
json.dump(CFG, open(os.path.join(RES_DIR, 'config.json'), 'w'), indent=2)
log('pre-registered config.json loaded + copied to results/ (BEFORE any forward)')

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
# image_ids.json at experiment root (task-required output)
json.dump({'source': 'experiments/exp005_visibility_diagnosis/results/image_ids.json '
                      '(EXP005-012 fixed pool; asserted in-run)',
           'image_ids': EXP5_IDS['image_ids']},
          open(os.path.join(EXP_DIR, 'image_ids.json'), 'w'), indent=2)


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


# ---------------- offline cache (one frozen forward per image; EXP012 verbatim) ----
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
            head = net.convraw[3](feat)
            seg, v0 = head[:, :net.seg_dim], head[:, net.seg_dim:]
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
        CACHE[split].append(dict(image_id=image_id, feat=feat.cpu(), v0=v0.cpu(),
                                 seg=seg.cpu(), vertex_gt=vertex_gt.cpu(), vw=vw.cpu(),
                                 ys=ys, xs=xs, ok=ok, u_gt=u_gt, gt2=gt2, vis=vis,
                                 d_occ=d_occ, H=vis.shape[0], W=vis.shape[1]))
log(f'cache done in {time.time()-T0:.1f}s')


# ---------------- A: original frozen vertex head ----------------
def field_A(item):
    return item['v0'].to(device)                      # raw, original pipeline


# ---------------- B: OCVF module ----------------
class OCVF(nn.Module):
    """Object-Consistent Vector Field refinement module.

    Inputs: feat F (1,32,H,W) and predicted seg logits (1,2,H,W) ONLY.
    z (shared object latent, from masked pooling of F) feeds ALL 9 keypoint
    contexts AND the configuration head; residual convs are SHARED across k.
    Final conv zero-initialized: at init dV == 0, so B == A exactly
    (EXP007-trap control, EXP012 convention).
    """
    def __init__(self, kp=KP, cin=32, d=64, center_xy=(320.0, 240.0)):
        super().__init__()
        self.kp, self.d = kp, d
        self.fc_z = nn.Linear(cin, d)
        self.queries = nn.Parameter(torch.randn(kp, d) * 0.02)
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.LeakyReLU(0.1),
                                  nn.Linear(d, d))
        self.res_in = nn.Conv2d(cin + d, d, 1, bias=True)     # shared across k
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.res_out = nn.Conv2d(d, 2, 1, bias=True)          # shared across k
        nn.init.zeros_(self.res_out.weight); nn.init.zeros_(self.res_out.bias)
        self.conf_head = nn.Linear(d, kp * 2)                 # weight zero, bias=center
        nn.init.zeros_(self.conf_head.weight)
        with torch.no_grad():
            self.conf_head.bias.copy_(torch.tensor(list(center_xy) * kp))

    def forward(self, feat, seg):
        mask = torch.argmax(seg, 1)[0]                        # (H,W) predicted fg
        fg = mask == 1
        Z = feat[0][:, fg].mean(1) if fg.any() else feat[0].mean((1, 2))
        z = self.fc_z(Z)                                      # (d,) shared object latent
        q = self.queries
        c = self.fuse(torch.cat([q, z[None].expand(self.kp, self.d)], 1))  # (kp,d)
        chat = self.conf_head(z).view(self.kp, 2)             # shared-configuration hypothesis
        f = feat[0][None].expand(self.kp, -1, -1, -1)         # (kp,32,H,W)
        cb = c[:, :, None, None].expand(self.kp, self.d, f.shape[2], f.shape[3])
        h = self.act(self.res_in(torch.cat([f, cb], 1)))      # shared weights across k
        dvert = self.res_out(h)                               # (kp,2,H,W)
        return dvert, chat


def field_B(item, module):
    """Returns (v_unit for direction metrics, v_raw for voting, chat).
    Voting input is the RAW refined field v0 + dV: the voting kernels are
    scale-invariant w.r.t. the direction field (verified in-run below), and
    the zero-init control then makes B's voting BIT-IDENTICAL to A's at init.
    Direction metrics/losses use normalize(v0 + dV) (task-spec form)."""
    feat, seg = item['feat'].to(device), item['seg'].to(device)
    dvert, chat = module(feat, seg)
    v0 = item['v0'].to(device)
    v_raw = v0 + dvert[None].reshape(1, KP * 2, *v0.shape[2:])
    return normalize_vertices(v_raw), v_raw, chat


# ---------------- G4: no-GT-input verification (BEFORE any training) --------------
sig = list(inspect.signature(OCVF.forward).parameters)
assert sig == ['self', 'feat', 'seg'], f'OCVF.forward signature has non-image args: {sig}'
src = inspect.getsource(OCVF.forward)
for tok in ('gt', 'pose', 'target', 'label', 'kps', 'mask_gt'):
    assert tok not in src.replace('target GT', ''), f'forward source mentions {tok}'
_mod = OCVF().to(device).eval()
_it = CACHE['occ'][3]
with torch.no_grad():
    d1, c1 = _mod(_it['feat'].to(device), _it['seg'].to(device))
    # corrupt every GT tensor of the item; the module must not see them at all
    _it2 = dict(_it); _it2['vertex_gt'] = _it['vertex_gt'] * 0 + 7.0
    _it2['gt2'] = _it['gt2'] * 0 + 123.0; _it2['vis'] = ~_it['vis']
    _it2['u_gt'] = _it['u_gt'] * 0; _it2['ok'] = ~_it['ok']
    d2, c2 = _mod(_it2['feat'].to(device), _it2['seg'].to(device))
gd = float((d1 - d2).abs().max()); gc = float((c1 - c2).abs().max())
log(f'G4 no-GT-input: corrupted-GT forward diff dvert {gd:.2e} chat {gc:.2e}')
assert gd == 0.0 and gc == 0.0, 'G4 FAILED: module output depends on GT tensors'
del _mod
log('G4 PASSED (forward uses only feat + predicted seg)')


# ---------------- G3: init identity B == A ----------------------------------------
_maxf, _maxk = 0.0, 0.0
_mod = OCVF().to(device).eval()
with torch.no_grad():
    for split in ('linemod', 'occ'):
        for item in CACHE[split][:5]:
            dvert, _ = _mod(item['feat'].to(device), item['seg'].to(device))
            _maxf = max(_maxf, float(dvert.abs().max()))
assert _maxf == 0.0, f'G3 field identity FAILED: zero-init dvert != 0 (max {_maxf})'
log('G3 (fields): zero-init dvert == 0 and normalize(v0) == v0_hat exactly')

_seg_dev = {s: [it['seg'].to(device) for it in CACHE[s][:5]] for s in ('linemod', 'occ')}
_v0_dev = {s: [it['v0'].to(device) for it in CACHE[s][:5]] for s in ('linemod', 'occ')}


def vote_keypoints(seg, ver):
    """EXP015 verbatim; seed re-applied per call (identical RNG stream)."""
    vertex = ver.permute(0, 2, 3, 1)
    b, h, w, vn_2 = vertex.shape
    vertex = vertex.view(b, h, w, vn_2 // 2, 2)
    mask = torch.argmax(seg, 1)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    return ransac_voting_layer_v3(mask, vertex, ROUND_HYP,
                                  inlier_thresh=INLIER_THRESH, max_num=MAX_NUM)


with torch.no_grad():
    for split in ('linemod', 'occ'):
        for seg, v0 in zip(_seg_dev[split], _v0_dev[split]):
            kp_a = vote_keypoints(seg, v0)
            _, vb_raw, _ = field_B({'feat': torch.zeros(1, 32, v0.shape[2], v0.shape[3], device=device),
                                    'seg': seg, 'v0': v0}, _mod)
            kp_b = vote_keypoints(seg, vb_raw)
            _maxk = max(_maxk, float((kp_a - kp_b).abs().max()))
log(f'G3 (voting, A raw vs B raw-at-init): max kp diff {_maxk:.2e} px '
    f'(scale-invariance noise raw-vs-unit was 3.63e-02 px in the aborted first run)')
assert _maxk == 0.0, 'G3 voting identity FAILED (raw fields must be bit-identical at init)'
del _mod, _seg_dev, _v0_dev
log('G3 PASSED')


# ---------------- G1/G2: A reproduces EXP006 + EXP012 init loss -------------------
def pooled_direction(field_fn, split):
    """EXP012 evaluate() core: direction/endpoint metrics on the EXP006 pool."""
    ang_all, end_all = [], []
    for item in CACHE[split]:
        v = normalize_vertices(field_fn(item))
        ys, xs, ok, u_gt = item['ys'], item['xs'], item['ok'], item['u_gt']
        H, W = item['vis'].shape
        ver = v.permute(0, 2, 3, 1).contiguous().view(1, H, W, KP, 2)[0]
        dirs = ver[ys, xs].cpu().numpy().astype(np.float64)
        u_pred = dirs / (np.linalg.norm(dirs, axis=2)[..., None] + EPS)
        ang = angle_deg(u_pred, u_gt)
        ang_all.append(ang[ok])
        end_all.append(np.linalg.norm(u_pred - u_gt, axis=2)[ok])
    return np.concatenate(ang_all), np.concatenate(end_all)


A_ANG = {}; A_END = {}
for split in ('linemod', 'occ'):
    a, e = pooled_direction(field_A, split)
    A_ANG[split], A_END[split] = a, e
g1_lin, g1_occ = mstats(A_ANG['linemod']), mstats(A_ANG['occ'])
log(f"G1 A direction: LIN {g1_lin['mean']:.3f} (ref {LIN_BASE})  OCC {g1_occ['mean']:.3f} "
    f"(ref {OCC_BASE})  counts {g1_lin['count']}/{g1_occ['count']} (ref {LIN_CNT}/{OCC_CNT})")
assert abs(g1_lin['mean'] - LIN_BASE) < 0.05 and abs(g1_occ['mean'] - OCC_BASE) < 0.05, 'G1 FAILED'
assert g1_lin['count'] == LIN_CNT and g1_occ['count'] == OCC_CNT, 'G1 counts FAILED'


def full_image_Ldir(split):
    """G2: A init L_dir under the EXP012 full-image loss formula."""
    with torch.no_grad():
        return float(np.mean([
            smooth_l1_loss(normalize_vertices(it['v0'].to(device)),
                           it['vertex_gt'].to(device), it['vw'].to(device),
                           reduce=False)[0].item() for it in CACHE[split]]))


for split in ('linemod', 'occ'):
    val = full_image_Ldir(split)
    ref = LIN_LOSS_REF if split == 'linemod' else OCC_LOSS_REF
    log(f'G2 A init L_dir [{split}]: {val:.6f} (EXP012 ref {ref})')
    assert abs(val - ref) < 5e-4, f'G2 FAILED on {split}'
log('G1 + G2 PASSED')


# ---------------- M6: GT-free structural consistency metric -----------------------
def m6_split(field_fn, split):
    """Per-image median-over-k of per-kp median perpendicular ray residual (px).

    Configuration hypothesis per (image, kp): LS intersection of the field's
    own rays over the eligible fg pool (ok = |gt_kp - p| >= 1px; GT used ONLY
    for pool selection, EXP006 pool convention). Degenerate cases excluded."""
    per_img = []
    n_degen = 0
    for item in CACHE[split]:
        with torch.no_grad():
            v = normalize_vertices(field_fn(item))[0].view(KP, 2, item['H'], item['W'])
        ys_t = torch.from_numpy(item['ys']).to(v.device)
        xs_t = torch.from_numpy(item['xs']).to(v.device)
        U = v[:, :, ys_t, xs_t].cpu().numpy()             # (9,2,n)
        P = np.stack([item['xs'], item['ys']], 1).astype(np.float64)
        ok = item['ok']
        meds = np.full(KP, np.nan)
        for k in range(KP):
            m = ok[:, k]
            if m.sum() < 50:
                n_degen += 1; continue
            u = U[k][:, m].T                                   # (m,2) unit rows
            pts = P[m]
            n = len(u)
            A_mat = n * np.eye(2) - u.T @ u + 1e-6 * n * np.eye(2)
            if np.linalg.cond(A_mat) > 1e12:
                n_degen += 1; continue
            proj = (u * pts).sum(1)
            b_vec = pts.sum(0) - u.T @ proj
            try:
                cpt = np.linalg.solve(A_mat, b_vec)
            except np.linalg.LinAlgError:
                n_degen += 1; continue
            d = np.abs(u[:, 0] * (cpt[1] - pts[:, 1]) - u[:, 1] * (cpt[0] - pts[:, 0]))
            meds[k] = np.median(d)
        per_img.append(float(np.nanmedian(meds)) if np.isfinite(meds).any() else np.nan)
    arr = np.asarray(per_img, np.float64)
    fin = arr[np.isfinite(arr)]
    return dict(per_image=per_img, n_degen_kp=n_degen,
                median=float(np.median(fin)) if fin.size else float('nan'),
                mean=float(fin.mean()) if fin.size else float('nan'),
                p90=float(np.quantile(fin, 0.9)) if fin.size else float('nan'))


# ---------------- M7: original voting keypoints -----------------------------------
def voting_errors(field_fn, split):
    errs, per_img = [], []
    for item in CACHE[split]:
        with torch.no_grad():
            v = field_fn(item)
            seg = item['seg'].to(device)
            kp = vote_keypoints(seg, v)[0]                    # (9,2)
        gt = torch.from_numpy(item['gt2']).float().to(device)
        e = torch.norm(kp - gt, dim=1).cpu().numpy().astype(np.float64)
        errs.append(e); per_img.append(float(np.median(e)))
    pooled = np.concatenate(errs)
    return pooled, per_img


# ---------------- fit OCVF per split (EXP012 budget, image-level) ------------------
def fit(split):
    torch.manual_seed(SEED)
    module = OCVF(center_xy=(CACHE[split][0]['W'] / 2.0, CACHE[split][0]['H'] / 2.0))
    module = module.to(device).train()
    n_par = sum(p.numel() for p in module.parameters())
    opt = torch.optim.Adam(module.parameters(), lr=LR)
    log(f'fit OCVF-{split}: params {n_par}  lr {LR}  epochs {EPOCHS} (image-level, batch=1)')
    tl = []
    for ep in range(1, EPOCHS + 1):
        tot = np.zeros(4); nb = 0
        for item in CACHE[split]:
            feat = item['feat'].to(device); seg = item['seg'].to(device)
            v0 = item['v0'].to(device)
            gt = item['vertex_gt'].to(device); vw = item['vw'].to(device)
            gt2 = torch.from_numpy(item['gt2']).float().to(device)
            H, W = item['H'], item['W']
            dvert, chat = module(feat, seg)
            out = normalize_vertices(v0 + dvert[None].reshape(1, KP * 2, H, W))
            l_dir = smooth_l1_loss(out, gt, vw, reduce=False)[0]
            D = float(np.sqrt(H * H + W * W))
            l_conf = torch.norm(chat - gt2[None], dim=1).mean() / D
            fg = (vw[0, 0] > 0)
            ys_t, xs_t = fg.nonzero(as_tuple=True)
            P = torch.stack([xs_t, ys_t], 1).float()          # (n,2) x,y
            diffg = P[:, None, :] - gt2[None, :, :]           # (n,9,2)
            diffc = P[:, None, :] - chat[None, :, :]
            sel = (diffg.norm(dim=2) >= 1.0) & (diffc.norm(dim=2) >= 1.0)
            if sel.any():
                u_conf = diffc / (diffc.norm(dim=2, keepdim=True) + EPS)
                u_pred = out.view(1, KP, 2, H, W)[0][:, :, ys_t, xs_t].permute(2, 0, 1)
                l_struct = (1.0 - (u_pred * u_conf).sum(-1))[sel].mean()
            else:
                l_struct = torch.zeros((), device=device)
            loss = l_dir + L_CONF_W * l_conf + L_STRUCT_W * l_struct
            opt.zero_grad(); loss.backward(); opt.step()
            tot += np.array([float(l_dir.detach()), float(l_conf.detach()),
                             float(l_struct.detach()), float(loss.detach())]); nb += 1
        tl.append([ep] + (tot / nb).tolist())
        log(f'fit OCVF-{split} epoch {ep}/{EPOCHS}  L_dir {tot[0]/nb:.6f}  '
            f'L_conf {tot[1]/nb:.6f}  L_struct {tot[2]/nb:.6f}  total {tot[3]/nb:.6f}  '
            f't={time.time()-T0:.0f}s')
    module.eval()
    return module, n_par, tl


MODELS = {}; FITLOG = {}; NPAR = {}
for split in ('linemod', 'occ'):
    MODELS[split], NPAR[split], FITLOG[split] = fit(split)
    torch.save(MODELS[split].state_dict(),
               os.path.join(RES_DIR, f'ocvf_{split}.pth'))   # runtime artifact (not committed)


# ---------------- full evaluation A vs B -------------------------------------------
def field_B_fn(split):
    m = MODELS[split]
    def fn(item):
        with torch.no_grad():
            v, _, _ = field_B(item, m)
        return v
    return fn


def field_B_raw_fn(split):
    """B's RAW refined field (v0 + dV) — the voting input (zero-init control)."""
    m = MODELS[split]
    def fn(item):
        with torch.no_grad():
            _, v_raw, _ = field_B(item, m)
        return v_raw
    return fn


RES = {}
for split in ('linemod', 'occ'):
    b_ang, b_end = pooled_direction(field_B_fn(split), split)
    RES[split] = dict(
        A_ang=A_ANG[split], A_end=A_END[split], B_ang=b_ang, B_end=b_end,
        A_m6=m6_split(field_A, split), B_m6=m6_split(field_B_fn(split), split),
        A_kp=voting_errors(field_A, split), B_kp=voting_errors(field_B_raw_fn(split), split))
    log(f"eval [{split}] done  t={time.time()-T0:.0f}s")


# ---------------- CSV outputs -------------------------------------------------------
with open(os.path.join(RES_DIR, 'direction_summary.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['split', 'cond', 'count', 'mean_deg', 'median_deg', 'p90_deg',
                 'bad_rate_gt10deg', 'endpoint_unit_mean'])
    for s in ('linemod', 'occ'):
        for cond, ang, end in (('A', RES[s]['A_ang'], RES[s]['A_end']),
                               ('B', RES[s]['B_ang'], RES[s]['B_end'])):
            ms = mstats(ang)
            wr.writerow([s, cond, ms['count'], f"{ms['mean']:.4f}", f"{ms['median']:.4f}",
                         f"{ms['p90']:.4f}", f"{float((ang > 10).mean()):.4f}",
                         f'{end.mean():.4f}'])

with open(os.path.join(RES_DIR, 'consistency_per_image.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['split', 'image_index', 'A_consistency_px', 'B_consistency_px'])
    for s in ('linemod', 'occ'):
        for i, (a, b) in enumerate(zip(RES[s]['A_m6']['per_image'], RES[s]['B_m6']['per_image'])):
            wr.writerow([s, i, f'{a:.4f}', f'{b:.4f}'])

with open(os.path.join(RES_DIR, 'consistency_summary.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['split', 'cond', 'median_px', 'mean_px', 'p90_px', 'n_degenerate_kp'])
    for s in ('linemod', 'occ'):
        for cond in ('A', 'B'):
            m6 = RES[s][f'{cond}_m6']
            wr.writerow([s, cond, f"{m6['median']:.4f}", f"{m6['mean']:.4f}",
                         f"{m6['p90']:.4f}", m6['n_degen_kp']])

with open(os.path.join(RES_DIR, 'keypoint_voting.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['split', 'cond', 'pooled_median_px', 'pooled_mean_px', 'p90_px'])
    for s in ('linemod', 'occ'):
        for cond in ('A', 'B'):
            pooled, _ = RES[s][f'{cond}_kp']
            wr.writerow([s, cond, f'{np.median(pooled):.4f}', f'{pooled.mean():.4f}',
                         f'{np.quantile(pooled, 0.9):.4f}'])

with open(os.path.join(RES_DIR, 'keypoint_voting_per_image.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['split', 'image_index', 'A_median_px', 'B_median_px'])
    for s in ('linemod', 'occ'):
        for i, (a, b) in enumerate(zip(RES[s]['A_kp'][1], RES[s]['B_kp'][1])):
            wr.writerow([s, i, f'{a:.4f}', f'{b:.4f}'])

with open(os.path.join(RES_DIR, 'train_log.csv'), 'w', newline='') as f:
    wr = csv.writer(f); wr.writerow(['fit', 'epoch', 'L_dir', 'L_conf', 'L_struct', 'total'])
    for s in ('linemod', 'occ'):
        for ep, ld, lc, ls, lt in FITLOG[s]:
            wr.writerow([f'OCVF-{s}', ep, f'{ld:.6f}', f'{lc:.6f}', f'{ls:.6f}', f'{lt:.6f}'])

# ---------------- decision (pre-registered) -----------------------------------------
A_lin = mstats(RES['linemod']['A_ang'])['mean']; B_lin = mstats(RES['linemod']['B_ang'])['mean']
A_occ = mstats(RES['occ']['A_ang'])['mean'];     B_occ = mstats(RES['occ']['B_ang'])['mean']
imp_dir_occ = (A_occ - B_occ) / A_occ
degr_dir_lin = (B_lin - A_lin) / A_lin
A_c_occ = RES['occ']['A_m6']['median']; B_c_occ = RES['occ']['B_m6']['median']
A_c_lin = RES['linemod']['A_m6']['median']; B_c_lin = RES['linemod']['B_m6']['median']
imp_cons_occ = (A_c_occ - B_c_occ) / A_c_occ
degr_cons_lin = (B_c_lin - A_c_lin) / A_c_lin
A_kp_occ = float(np.median(RES['occ']['A_kp'][0])); B_kp_occ = float(np.median(RES['occ']['B_kp'][0]))
A_kp_lin = float(np.median(RES['linemod']['A_kp'][0])); B_kp_lin = float(np.median(RES['linemod']['B_kp'][0]))

C1 = imp_dir_occ >= 0.10
C2 = degr_dir_lin <= 0.05
C3 = (imp_cons_occ >= 0.05) and (degr_cons_lin <= 0.05)
C4 = B_kp_occ <= 1.05 * A_kp_occ
decision = 'GO' if (C1 and C2 and C3 and C4) else 'STOP'
n_pass = int(C1) + int(C2) + int(C3) + int(C4)
profile = 'GO' if n_pass == 4 else ('MIXED profile (partial satisfaction)' if n_pass >= 2
                                    else 'clean fail')

log(f"RESULT direction  LIN A {A_lin:.4f} -> B {B_lin:.4f} ({-degr_dir_lin:+.2%} improvement) | "
    f"OCC A {A_occ:.4f} -> B {B_occ:.4f} ({imp_dir_occ:+.2%} improvement)")
log(f"RESULT consistency LIN A {A_c_lin:.4f} -> B {B_c_lin:.4f} ({-degr_cons_lin:+.2%}) | "
    f"OCC A {A_c_occ:.4f} -> B {B_c_occ:.4f} ({imp_cons_occ:+.2%})")
log(f"RESULT voting-kp  LIN A {A_kp_lin:.4f} -> B {B_kp_lin:.4f} | "
    f"OCC A {A_kp_occ:.4f} -> B {B_kp_occ:.4f} (pooled median px)")
log(f"CRITERIA C1(OCC dir >=10%): {C1}  C2(LIN dir <=5% worse): {C2}  "
    f"C3(M6 OCC >=5% & LIN <=5% worse): {C3}  C4(OCC kp <=1.05x): {C4}")
log(f'DECISION: {decision}  [{profile}, {n_pass}/4 criteria]')

json.dump(dict(
    baseline_direction=dict(LIN=A_lin, OCC=A_occ, counts=[g1_lin['count'], g1_occ['count']]),
    direction=dict(LIN_A=A_lin, LIN_B=B_lin, OCC_A=A_occ, OCC_B=B_occ,
                   imp_dir_occ=imp_dir_occ, degr_dir_lin=degr_dir_lin),
    consistency=dict(LIN_A=A_c_lin, LIN_B=B_c_lin, OCC_A=A_c_occ, OCC_B=B_c_occ,
                     imp_cons_occ=imp_cons_occ, degr_cons_lin=degr_cons_lin,
                     n_degen=dict(LIN_A=RES['linemod']['A_m6']['n_degen_kp'],
                                  LIN_B=RES['linemod']['B_m6']['n_degen_kp'],
                                  OCC_A=RES['occ']['A_m6']['n_degen_kp'],
                                  OCC_B=RES['occ']['B_m6']['n_degen_kp'])),
    voting_kp=dict(LIN_A=A_kp_lin, LIN_B=B_kp_lin, OCC_A=A_kp_occ, OCC_B=B_kp_occ),
    criteria=dict(C1=bool(C1), C2=bool(C2), C3=bool(C3), C4=bool(C4), n_pass=n_pass),
    decision=decision, profile=profile, n_params=NPAR,
    runtime_s=time.time() - T0),
    open(os.path.join(RES_DIR, 'final_summary.json'), 'w'), indent=2)
log(f'DONE in {time.time()-T0:.1f}s')
