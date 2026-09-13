"""EXP003 mechanism diagnostics (the most important check):

  distance-to-coarse-keypoint d_ik  vs  true vote error e_ik
  (Pearson / Spearman), reliability distributions, filter ratios.

Usage (from repo root):
  python experiments/exp003_consensus_weighting/analyze_exp003.py \
      --split linemod --n_images 100 --seed 0
"""
import os, sys, json, argparse, importlib.util
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, SequentialSampler

from lib.networks.model_repository import Resnet18_8s
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v2
from lib.datasets.linemod_dataset import LineModDatasetRealAug, VotingType
from lib.utils.data_utils import LineModImageDB, OcclusionLineModImageDB
from lib.utils.config import cfg
from consensus_weighting import get_voting_problem, consensus_reliability
from run_exp003 import ImageSizeBatchSampler, RC, X3, EXP_DIR, build_split


def line_dist(ref_kp, coords, direct_k):
    v = direct_k / (torch.norm(direct_k, dim=1, keepdim=True) + 1e-9)
    diff = ref_kp.unsqueeze(0) - coords
    return torch.abs(diff[:, 0] * v[:, 1] - diff[:, 1] * v[:, 0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['linemod', 'occ'], required=True)
    p.add_argument('--n_images', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    device = 'cuda'
    net = Resnet18_8s(ver_dim=18, seg_dim=2)
    sd = torch.load(os.path.join(ROOT, CONF['checkpoint']), map_location='cpu')
    net.load_state_dict(sd['net'])
    net = net.to(device).eval()

    db_set, loader = build_split(a.split, a.n_images)
    ds_d, ds_e = [], []
    kept = {'A4': [], 'A6': []}
    r_soft = []
    with torch.no_grad():
        for image_id, data in enumerate(loader):
            image, mask_gt, vertex_gt, vertex_weights, pose, gt_kps = [d.to(device) for d in data]
            seg_pred, vertex_pred = net(image)
            mask = torch.argmax(seg_pred, 1)
            b, _, h, w = vertex_pred.shape
            ver = vertex_pred.permute(0, 2, 3, 1).contiguous().view(b, h, w, -1, 2)
            torch.manual_seed(a.seed)
            corners = ransac_voting_layer_v2(
                mask, ver, RC['class_num'], RC['round_hyp_num'],
                inlier_thresh=RC['inlier_thresh'], confidence=RC['confidence'],
                max_iter=RC['max_iter'], min_num=RC['min_num'],
                max_num=RC['max_num'], refine_iter_num=RC['refine_iter_num'])
            torch.manual_seed(a.seed)
            problem = get_voting_problem(mask, ver, RC['class_num'], RC['round_hyp_num'],
                                         max_num=RC['max_num'], min_num=RC['min_num'])
            coarse = corners[0, 0]
            gt2 = gt_kps[:, 0, :, :2][0] if gt_kps.dim() == 4 else gt_kps[0, :, :2]
            if problem is None:
                continue
            for k in range(9):
                if not torch.isfinite(coarse[k]).all():
                    continue
                coords = problem['coords']
                dk = problem['direct'][:, k, :]
                d = line_dist(coarse[k], coords, dk)
                e = line_dist(gt2[k], coords, dk)
                ds_d.append(d.cpu().numpy())
                ds_e.append(e.cpu().numpy())
                r_s, _ = consensus_reliability(coords, dk, coarse[k], sigma=X3['sigma_soft'],
                                               mode='soft', weight_clip=X3['weight_clip'])
                r_soft.append(r_s.cpu().numpy())
                for tau_key in ('A4', 'A6'):
                    tau = X3['tau_hard'][0] if tau_key == 'A4' else X3['tau_hard'][1]
                    r_h, _ = consensus_reliability(coords, dk, coarse[k], tau=tau, mode='hard')
                    kept[tau_key].append(float(r_h.mean().item()))
            if image_id % 25 == 0:
                print('{} {}/{}'.format(a.split, image_id, len(loader)), flush=True)

    d_all = np.concatenate(ds_d)
    e_all = np.concatenate(ds_e)
    r_all = np.concatenate(r_soft)
    pr, sr = pearsonr(d_all, e_all)[0], spearmanr(d_all, e_all)[0]
    out = dict(
        split=a.split, n_images=a.n_images, seed=a.seed, n_pixels=int(len(d_all)),
        pearson_d_vs_e=float(pr), spearman_d_vs_e=float(sr),
        d_stats=dict(mean=float(d_all.mean()), median=float(np.median(d_all)),
                     p10=float(np.percentile(d_all, 10)), p90=float(np.percentile(d_all, 90))),
        e_stats=dict(mean=float(e_all.mean()), median=float(np.median(e_all)),
                     p10=float(np.percentile(e_all, 10)), p90=float(np.percentile(e_all, 90))),
        soft_rel_stats=dict(mean=float(r_all.mean()), median=float(np.median(r_all)),
                            p10=float(np.percentile(r_all, 10)),
                            p50=float(np.percentile(r_all, 50)),
                            p90=float(np.percentile(r_all, 90))),
        filter_keep_ratio=dict(A4=float(np.mean(kept['A4'])),
                               A6=float(np.mean(kept['A6']))),
    )
    res_dir = os.path.join(EXP_DIR, 'results')
    fig_dir = os.path.join(EXP_DIR, 'figures')
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(res_dir, 'mechanism_{}.json'.format(a.split)), 'w') as f:
        json.dump(out, f, indent=2)
    np.savez(os.path.join(res_dir, 'mechanism_{}.npz'.format(a.split)), d=d_all, e=e_all, r=r_all)
    print(json.dumps({k: v for k, v in out.items() if k != 'n_pixels'}, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(r_all, bins=50, log=True)
    axes[0].set_title('soft reliability r (sigma=4)')
    axes[1].hist(d_all, bins=100, log=True)
    axes[1].set_title('coarse-consensus distance d (px)')
    axes[2].hist(e_all, bins=100, log=True)
    axes[2].set_title('true vote error e (px)')
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'mechanism_{}.png'.format(a.split)), dpi=150)
    fig, ax = plt.subplots(figsize=(5, 5))
    idx = np.random.RandomState(0).choice(len(d_all), min(50000, len(d_all)), replace=False)
    ax.scatter(d_all[idx], e_all[idx], s=1, alpha=0.2)
    ax.set_xlabel('coarse-consensus distance d (px)')
    ax.set_ylabel('true vote error e (px)')
    ax.set_title('P(earson)={:.3f} S(pearman)={:.3f}'.format(pr, sr))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'd_vs_e_scatter_{}.png'.format(a.split)), dpi=150)
    print('figures saved')


CONF = json.load(open(os.path.join(EXP_DIR, 'config.json')))

if __name__ == '__main__':
    main()
