"""Final analysis: Exp001-S top-k oracle (true/learned/random x 100/70/50/30)
+ reliability weight distribution + Pearson/Spearman. Deterministic LS oracle.
Usage: python /tmp/final_topk_rel.py <epochs...>
Writes results/topk_results.csv and rel_stats.csv in exp001_final_analysis.
"""
import sys, os, json
sys.path.append('.')
sys.path.append('..')

import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, SequentialSampler

import importlib.util
spec = importlib.util.spec_from_file_location('e', 'tools/train_linemod_exp001.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

from lib.networks.model_repository_exp001 import Resnet18_8sReliability
from lib.datasets.linemod_dataset import LineModDatasetRealAug, VotingType
from lib.utils.data_utils import LineModImageDB, OcclusionLineModImageDB
from lib.utils.config import cfg
from tools.train_linemod_exp001 import pseudo_reliability

EPS = [int(x) for x in sys.argv[1:]]
train_cfg = json.load(open('configs/exp001_reliability_voting_scratch.json'))
train_cfg['linemod_cls'] = 'cat'
RES = 'experiments/exp001_final_analysis/results'
os.makedirs(RES, exist_ok=True)
MODEL_DIR = 'experiments/exp001_reliability_voting_scratch/model/cat_exp001s_scratch'
KEEPS = [1.0, 0.7, 0.5, 0.3]
RANKS = ['learned', 'random', 'true']
MIN_PX = 10

db = LineModImageDB('cat', has_render_set=False, has_fuse_set=False)
occ_db = OcclusionLineModImageDB('cat')
half = occ_db.test_real_set[:len(occ_db.test_real_set) // 2]

topk_csv = os.path.join(RES, 'topk_results.csv')
rel_csv = os.path.join(RES, 'rel_stats.csv')
for p, hdr in ((topk_csv, 'epoch,split,ranking,keep,num,projection_error,add,cm_degree_5\n'),
               (rel_csv, 'epoch,split,n_images,n_px,r_mean,r_std,r_p10,r_p50,r_p90,frac_gt09,pearson_r_vs_e,spearman_r_vs_e\n')):
    if not os.path.exists(p):
        with open(p, 'w') as f:
            f.write(hdr)


def make_loader(db_set, data_prefix):
    ds = LineModDatasetRealAug(db_set, data_prefix, VotingType.Farthest, augment=False,
                               cfg=train_cfg['aug_cfg'])
    return DataLoader(ds, batch_sampler=m.ImageSizeBatchSampler(
        SequentialSampler(ds), 1, False, cfg=train_cfg['aug_cfg']), num_workers=6)


def ls_intersect(pts, dirs):
    if len(pts) == 0:
        return None
    d = dirs
    a11 = 1.0 - d[:, 0] * d[:, 0]
    a12 = -d[:, 0] * d[:, 1]
    a22 = 1.0 - d[:, 1] * d[:, 1]
    A = np.array([[a11.sum(), a12.sum()], [a12.sum(), a22.sum()]])
    b = np.array([(a11 * pts[:, 0] + a12 * pts[:, 1]).sum(),
                  (a12 * pts[:, 0] + a22 * pts[:, 1]).sum()])
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


for ep in EPS:
    sd = torch.load(os.path.join(MODEL_DIR, '{}.pth'.format(ep)), map_location='cpu')
    net = Resnet18_8sReliability(ver_dim=18, seg_dim=2)
    net.load_state_dict(sd['net'])
    wrapper = torch.nn.DataParallel(m.NetWrapperRel(net, True, train_cfg['reliability_sigma_px'],
                                                    train_cfg['reliability_tau1_px'],
                                                    train_cfg['reliability_tau2_px'])).cuda()
    wrapper.eval()
    for split, db_set, prefix in (('linemod_val', db.val_real_set, cfg.LINEMOD),
                                  ('occ_val', half, cfg.OCCLUSION_LINEMOD)):
        loader = make_loader(db_set, prefix)
        results = {(r, k): m.Evaluator() for r in RANKS for k in KEEPS}
        rs = []
        with torch.no_grad():
            for data in loader:
                image, mask, vertex, vertex_weights, pose, gt_kps = [d.cuda() for d in data]
                gt2 = gt_kps[:, :, :2]
                out = wrapper(image, mask, vertex, vertex_weights, gt2)
                r_pred = torch.sigmoid(out[2])
                vw4 = vertex_weights[0]
                if vw4.dim() == 3:
                    vw4 = vw4[0]
                vw = vw4.bool().cpu().numpy()
                r_flat = r_pred[0].reshape(9, -1)[:, torch.from_numpy(vw.flatten()).cuda()].cpu().numpy()
                rs.append(r_flat.T)
                ver_np = out[1][0].permute(1, 2, 0).cpu().numpy().reshape(vw.shape[0], vw.shape[1], 9, 2)
                rel_np = r_pred[0].cpu().numpy()
                _, e_all = pseudo_reliability(out[1].detach(), gt2, vertex_weights, 3.0)
                e_np = e_all[0].cpu().numpy()
                corners = {}
                for k in range(9):
                    ys, xs = np.nonzero(vw)
                    if len(xs) < MIN_PX:
                        corners[('fail', k)] = True
                        continue
                    pts = np.stack([xs, ys], 1).astype(np.float64)
                    dirs = ver_np[ys, xs, k, :]
                    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
                    r_k = rel_np[k][ys, xs]
                    e_k = e_np[k][ys, xs]
                    order = np.argsort(-r_k)
                    perm = np.random.RandomState(0).permutation(len(pts))
                    for rank, o in (('learned', order), ('random', perm), ('true', np.argsort(e_k))):
                        for keep in KEEPS:
                            nk = max(int(len(pts) * keep), MIN_PX)
                            sel = o[:nk]
                            c = ls_intersect(pts[sel], dirs[sel])
                            corners[(rank, keep, k)] = c if c is not None else (1e6, 1e6)
                pose_np = pose.cpu().numpy()
                for bi in range(pose_np.shape[0]):
                    for rank in RANKS:
                        for keep in KEEPS:
                            kp = np.zeros((9, 2))
                            for k in range(9):
                                kp[k] = (1e6, 1e6) if ('fail', k) in corners else corners[(rank, keep, k)]
                            results[(rank, keep)].evaluate(kp, pose_np[bi], 'cat', 'linemod',
                                                           VotingType.Farthest, intri_matrix=None)
        r_all = np.concatenate([r.reshape(-1) for r in rs])
        # correlation on first 100 images' pixels
        r_c = np.concatenate([r.reshape(-1) for r in rs[:100]])
        e_c = []
        sub = LineModDatasetRealAug(db_set[:100], prefix, VotingType.Farthest, augment=False,
                                    cfg=train_cfg['aug_cfg'])
        ldr = DataLoader(sub, batch_sampler=m.ImageSizeBatchSampler(
            SequentialSampler(sub), 1, False, cfg=train_cfg['aug_cfg']), num_workers=6)
        with torch.no_grad():
            for data in ldr:
                image, mask, vertex, vertex_weights, pose, gt_kps = [d.cuda() for d in data]
                gt2 = gt_kps[:, :, :2]
                out = wrapper(image, mask, vertex, vertex_weights, gt2)
                _, e_all = pseudo_reliability(out[1].detach(), gt2, vertex_weights, 3.0)
                fg = vertex_weights[:, 0].bool().unsqueeze(1).expand_as(e_all) \
                    if vertex_weights.dim() == 4 else vertex_weights.bool().expand_as(e_all)
                e_c.append(e_all[fg].cpu().numpy())
        e_c = np.concatenate(e_c)
        pr, sr = pearsonr(r_c, e_c)[0], spearmanr(r_c, e_c)[0]
        with open(rel_csv, 'a') as f:
            f.write('{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(
                ep, split, 100, len(r_c), r_all.mean(), r_all.std(),
                np.percentile(r_all, 10), np.percentile(r_all, 50), np.percentile(r_all, 90),
                (r_all > 0.9).mean(), pr, sr))
        print('e{} {} rel: mean {:.3f} std {:.3f} p10/p50/p90 {:.2f}/{:.2f}/{:.2f} '
              'pearson {:.3f} spearman {:.3f}'.format(ep, split, r_all.mean(), r_all.std(),
                                                      np.percentile(r_all, 10),
                                                      np.percentile(r_all, 50),
                                                      np.percentile(r_all, 90), pr, sr),
              flush=True)
        for rank in RANKS:
            for keep in KEEPS:
                proj, add, cm = results[(rank, keep)].average_precision(False)
                with open(topk_csv, 'a') as f:
                    f.write('{},{},{},{},{},{:.4f},{:.4f},{:.4f}\n'.format(
                        ep, split, rank, keep, len(db_set), proj, add, cm))
                print('e{} {} {} {}: add {:.4f}'.format(ep, split, rank, keep, add), flush=True)
    del wrapper
    torch.cuda.empty_cache()
print('done')
