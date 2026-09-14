"""EXP004: cheap, non-self-referential vote-quality signals from baseline PVNet.

20 LINEMOD + 20 OCC val images (deterministic first-20), 199.pth, no training.
Signals (per foreground pixel i, keypoint k):
  A seg_confidence   = P(fg | i)                (higher = expected better)
  B boundary_dist    = dist(i, mask boundary)    (higher = expected better)
  C local_consistency= mean cos(v_i, neighbours) within 5x5 fg neighbourhood
  E vertex_magnitude = ||v_raw||                 (orientation via pooled Pearson)
  F seg_x_boundary   = seg_conf * boundary_dist / p90(boundary_dist)
Target: e_ik = point-to-line distance(GT kp_k, vote line of i).
Signal D skipped by design (cost). Cap ~3000 sampled fg pixels per image
(deterministic row-major stride).

Run from repo root: python experiments/exp004_vote_quality_signal/exp004_signals.py
"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import json
import numpy as np
import torch
import torch.nn.functional as tF
from scipy.ndimage import distance_transform_edt
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, SequentialSampler

from lib.networks.model_repository import Resnet18_8s
from lib.datasets.linemod_dataset import LineModDatasetRealAug, VotingType
from lib.utils.data_utils import LineModImageDB, OcclusionLineModImageDB
from lib.utils.config import cfg

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
N_IMG = int(os.environ.get('EXP004_NIMG', '20'))
PX_CAP = 3000
KP = 9

aug_cfg = json.load(open(os.path.join(
    ROOT, 'configs', 'exp001_reliability_voting_scratch.json')))['aug_cfg']

device = 'cuda'
net = Resnet18_8s(ver_dim=18, seg_dim=2)
sd = torch.load(os.path.join(ROOT, 'data/model/cat_linemod_train/199.pth'), map_location='cpu')
net.load_state_dict(sd['net'])
net = net.to(device).eval()


def local_consistency(ver, fg):
    """ver [h,w,9,2] cuda, fg [h,w] bool. Returns [h,w,9] mean cos to 5x5 fg neighbours."""
    m = fg.float()[None, None]
    cnt = tF.conv2d(m, torch.ones(1, 1, 5, 5, device=device), padding=2)[0, 0] - 1.0
    out = torch.zeros(ver.shape[0], ver.shape[1], KP, device=device)
    for k in range(KP):
        d = ver[:, :, k, :].permute(2, 0, 1)[None]                 # [1,2,h,w]
        d = d / (d.norm(dim=1, keepdim=True) + 1e-9)
        s = tF.conv2d(d, torch.ones(1, 2, 5, 5, device=device), padding=2)[0]  # [2,h,w]
        dot = (d[0] * s).sum(0) - 1.0                              # minus self cos
        out[:, :, k] = dot / cnt.clamp(min=1)
    return out


def main():
    from torch.utils.data import DataLoader as DL
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'tle', os.path.join(ROOT, 'tools', 'train_linemod_exp001.py'))
    tle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tle)
    ImageSizeBatchSampler = tle.ImageSizeBatchSampler

    rows = {s: {k: [] for k in ('img', 'kp', 'e', 'seg_conf', 'boundary', 'localcons',
                                'mag', 'segbnd')} for s in ('linemod', 'occ')}
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
        loader = DL(ds, batch_sampler=ImageSizeBatchSampler(
            SequentialSampler(ds), 1, False, aug_cfg), num_workers=6)
        for image_id, data in enumerate(loader):
            image, mask_gt, vertex_gt, vw, pose, gt_kps = [d.to(device) for d in data]
            with torch.no_grad():
                seg_pred, vertex_pred = net(image)
                prob = tF.softmax(seg_pred, 1)[0, 1]                       # [h,w]
                b, _, h, w = vertex_pred.shape
                ver = vertex_pred.permute(0, 2, 3, 1).contiguous().view(b, h, w, -1, 2)[0]
                fg = (torch.argmax(seg_pred, 1)[0] == 1)
                gt2 = gt_kps[0, :, :2] if gt_kps.dim() == 3 else gt_kps[0, 0, :, :2]

                fg_np = fg.cpu().numpy()
                dist_map = torch.from_numpy(distance_transform_edt(fg_np)).float().to(device)
                lc = local_consistency(ver, fg)
                mag = ver.norm(dim=-1)                                     # [h,w,9]

                ys, xs = torch.nonzero(fg, as_tuple=True)
                n = len(xs)
                if n == 0:
                    continue
                stride = max(1, int(n) // PX_CAP)
                sel = torch.arange(0, n, stride, device=device)
                ys, xs = ys[sel], xs[sel]

                v = ver[ys, xs]                                            # [S,9,2]
                v = v / (v.norm(dim=-1, keepdim=True) + 1e-9)
                diff = gt2.unsqueeze(0) - torch.stack([xs, ys], 1).float().unsqueeze(1)
                e = torch.abs(diff[..., 0] * v[..., 1] - diff[..., 1] * v[..., 0])  # [S,9]

                s_conf = prob[ys, xs].unsqueeze(1).expand(-1, KP)
                bdist = dist_map[ys, xs].unsqueeze(1).expand(-1, KP)
                lcons = lc[ys, xs]
                mgn = mag[ys, xs]
                bnorm = torch.quantile(dist_map[fg], 0.9).clamp(min=1.0)
                sxbnd = (s_conf * (bdist / bnorm)).clamp(max=10.0)

                r = rows[split]
                r['img'].append(torch.full((len(sel) * KP,), image_id, device=device))
                r['kp'].append(torch.arange(KP, device=device).repeat(len(sel)).float())
                r['e'].append(e.flatten())
                r['seg_conf'].append(s_conf.flatten())
                r['boundary'].append(bdist.flatten())
                r['localcons'].append(lcons.flatten())
                r['mag'].append(mgn.flatten())
                r['segbnd'].append(sxbnd.flatten())
            print('DBG {} img {} n_px {} rows_e {}'.format(
                split, image_id, int(n), len(rows[split]['e'])), flush=True)
            print('{} {}/{}'.format(split, image_id, len(loader)), flush=True)

    summary_rows = []
    rank_rows = []
    band_rows = []
    for split in ('linemod', 'occ'):
        r = {k: torch.cat(v).cpu().numpy() for k, v in rows[split].items()}
        n = len(r['e'])
        print('{}: {} vote rows'.format(split, n))
        rng = np.random.RandomState(0)
        random_mask = rng.rand(n) < 0.5
        for sig in ('seg_conf', 'boundary', 'localcons', 'mag', 'segbnd'):
            x = r[sig]
            pr, sr = pearsonr(x, r['e'])[0], spearmanr(x, r['e'])[0]
            summary_rows.append((sig, split, pr, sr))
            # orientation: a-priori higher=better except magnitude (pooled Pearson sign)
            orient = 1.0
            if sig == 'mag':
                orient = 1.0 if pr >= 0 else -1.0
            score = orient * x
            order = np.argsort(-score)
            keep = order[:n // 2]
            e_sig = float(r['e'][keep].mean())
            e_rnd = float(r['e'][random_mask].mean())
            e_orc = float(np.sort(r['e'])[:n // 2].mean())
            rank_rows.append((sig, split, e_sig, e_rnd, e_orc))
            # error-band analysis: mean signal in top/bottom error bands
            eo = np.argsort(r['e'])
            for name, sl in (('top10', eo[:n // 10]), ('top30', eo[:3 * n // 10]),
                             ('top50', eo[:n // 2]), ('bot50', eo[n // 2:]),
                             ('bot30', eo[3 * n // 10:]), ('bot10', eo[n // 10:])):
                band_rows.append((sig, split, name, float(r['e'][sl].mean()),
                                  float(x[sl].mean())))

    os.makedirs(os.path.join(EXP_DIR, 'results'), exist_ok=True)
    with open(os.path.join(EXP_DIR, 'results', 'signal_summary.csv'), 'w') as f:
        f.write('signal,split,pearson,spearman\n')
        for row in summary_rows:
            f.write('{},{},{:.4f},{:.4f}\n'.format(*row))
    with open(os.path.join(EXP_DIR, 'results', 'ranking_keep50.csv'), 'w') as f:
        f.write('signal,split,e_selected,e_random,e_oracle\n')
        for row in rank_rows:
            f.write('{},{},{:.4f},{:.4f},{:.4f}\n'.format(*row))
    with open(os.path.join(EXP_DIR, 'results', 'error_bands.csv'), 'w') as f:
        f.write('signal,split,band,mean_e,mean_signal\n')
        for row in band_rows:
            f.write('{},{},{},{:.4f},{:.4f}\n'.format(*row))
    np.savez(os.path.join(EXP_DIR, 'results', 'rows.npz'),
             **{'{}_{}'.format(s, k): torch.cat(v).cpu().numpy()
                for s in rows for k, v in rows[s].items()})
    print('summary:')
    for row in summary_rows:
        print('  {:<10} {:<8} pearson {:+.4f} spearman {:+.4f}'.format(*row))
    print('ranking keep50 (e_selected / e_random / e_oracle):')
    for row in rank_rows:
        print('  {:<10} {:<8} {:.3f} / {:.3f} / {:.3f}'.format(*row))
    print('DONE')


if __name__ == '__main__':
    main()
