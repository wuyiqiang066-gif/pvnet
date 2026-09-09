"""exp001 Part 7: visualization & analysis.

Figures (saved under experiments/exp001_reliability_voting/results/figures/):
  1. reliability_heatmap_<split>_<idx>.png  image | GT mask+keypoints | reliability heatmap
                                            (+ per-keypoint reliability maps)
  2. kp_error_comparison.png / .csv         baseline vs weighted voting, per-image mean kp error
  3. weight_distribution.png / .json        predicted weights in interior / visible boundary /
                                            occluded region / occlusion boundary

Vote-level analysis only (PnP and evaluation are untouched, per design).
Usage:
  python tools/visualize_exp001.py --ckpt <path.pth> --num 8
  (omit --ckpt to validate the pipeline with a freshly initialized rel_head)
"""
import sys, os, json, argparse

sys.path.append('.')
sys.path.append('..')

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.networks.model_repository_exp001 import Resnet18_8sReliability
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v3
from lib.ransac_voting_gpu_layer.ransac_voting_weighted import ransac_voting_layer_v3_weighted
from lib.datasets.linemod_dataset import LineModDatasetRealAug, VotingType
from lib.utils.data_utils import LineModImageDB, OcclusionLineModImageDB
from lib.utils.config import cfg

EXP_DIR = 'experiments/exp001_reliability_voting'
VOTE_NUM = 9


def load_net(ckpt_path):
    net = Resnet18_8sReliability(ver_dim=VOTE_NUM * 2, seg_dim=2).cuda().eval()
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location='cpu')
        net.load_state_dict(sd['net'], strict=False)
        print('loaded ckpt {} (missing keys are the fresh rel_head)'.format(ckpt_path))
    else:
        print('WARNING: no ckpt given -> rel_head is randomly initialized '
              '(pipeline validation only, numbers are meaningless)')
    return net


def erode(mask, k):
    return cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


def amodal_mask_path(cls_name, item):
    import re
    rgb_pth = item.get('rgb_pth', '') if isinstance(item, dict) else ''
    stem = os.path.splitext(os.path.basename(rgb_pth))[0]
    m = re.search(r'(\d+)', stem)          # e.g. color_00000 -> 0 -> amodal_masks/cat/0.png
    if not m:
        return None
    sid = int(m.group(1))
    for name in ('{}.png'.format(sid), '{:05d}.png'.format(sid), stem + '.png'):
        p = os.path.join(cfg.OCCLUSION_LINEMOD, 'amodal_masks', cls_name, name)
        if os.path.exists(p):
            return p
    return None


def predict(net, rgb, mask, ver, vw, pose, hcoords, args):
    img = rgb.unsqueeze(0).cuda()
    mask_t = mask.unsqueeze(0).cuda()
    kp = hcoords.unsqueeze(0).cuda()
    with torch.no_grad():
        seg, verv, rel = net(img)
        h, w = seg.shape[2:]
        vertex = verv.permute(0, 2, 3, 1).view(1, h, w, VOTE_NUM, 2)
        w_map = torch.sigmoid(rel).permute(0, 2, 3, 1)                       # (1,h,w,9)
        pts_b = ransac_voting_layer_v3(mask_t, vertex, args.round_hyp_num,
                                       inlier_thresh=0.99, max_num=args.vote_max_num)
        pts_w = ransac_voting_layer_v3_weighted(mask_t, vertex, w_map, args.round_hyp_num,
                                                inlier_thresh=0.99, max_num=args.vote_max_num,
                                                min_weight=args.min_weight)
    gt = kp
    if gt.dim() == 3 and gt.shape[-1] == 3:
        gt = gt[..., :2] / torch.clamp(gt[..., 2:], min=1e-6)
    err_b = (pts_b[0].cpu() - gt[0].cpu()).norm(dim=1).numpy()               # (9,)
    err_w = (pts_w[0].cpu() - gt[0].cpu()).norm(dim=1).numpy()
    return (seg[0].cpu(), w_map[0].cpu(), mask.numpy(), rgb.numpy(),
            gt[0].cpu().numpy(), err_b, err_w)


IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denorm(img_chw):
    """CHW normalized image -> HWC uint8 for display."""
    img = np.transpose(img_chw, (1, 2, 0)) * IMG_STD + IMG_MEAN
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def figure_heatmap(split, idx, rgb, mask_np, gt_kps, rel_map, out_dir):
    """image | GT mask+keypoints | mean reliability overlay | per-kp reliability maps."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    img = denorm(rgb)
    mean_rel = rel_map.mean(axis=2)                                          # (h,w) mean over kps

    axes[0, 0].imshow(img); axes[0, 0].set_title('image')
    axes[0, 1].imshow(img); axes[0, 1].imshow(mask_np, alpha=0.4, cmap='viridis')
    axes[0, 1].plot(gt_kps[:, 0], gt_kps[:, 1], 'r+', ms=12, mew=2)
    axes[0, 1].set_title('GT mask + 9 keypoints')
    axes[0, 2].imshow(img); axes[0, 2].imshow(mean_rel, alpha=0.55, cmap='inferno',
                                               vmin=0, vmax=1)
    axes[0, 2].plot(gt_kps[:, 0], gt_kps[:, 1], 'c+', ms=12, mew=2)
    axes[0, 2].set_title('mean predicted reliability')
    im = axes[0, 3].imshow(mean_rel, cmap='inferno', vmin=0, vmax=1)
    axes[0, 3].set_title('reliability map'); fig.colorbar(im, ax=axes[0, 3], fraction=0.046)

    for j, k in enumerate((0, 1, 2)):
        ax = axes[1, j]
        ax.imshow(rel_map[:, :, k], cmap='inferno', vmin=0, vmax=1)
        ax.plot(gt_kps[k, 0], gt_kps[k, 1], 'c+', ms=10, mew=2)
        ax.set_title('rel kp{}'.format(k))
    axes[1, 3].hist(mean_rel[mask_np > 0].ravel(), bins=32, range=(0, 1), color='steelblue')
    axes[1, 3].set_title('reliability histogram (fg)')
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('{} #{} reliability'.format(split, idx))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'reliability_heatmap_{}_{}.png'.format(split, idx)),
                dpi=120)
    plt.close(fig)


def collect_set(net, split, num, args):
    """run predictions on a few images; returns per-image records (rgb kept for figures)."""
    if split == 'LINEMOD':
        db = LineModImageDB('cat', has_render_set=False, has_fuse_set=False)
        ds = LineModDatasetRealAug(db.test_real_set[:num], cfg.LINEMOD, VotingType.Farthest,
                                   augment=False, cfg=args.aug_cfg)
        metas = db.test_real_set[:num]
    else:
        db = OcclusionLineModImageDB('cat')
        ds = LineModDatasetRealAug(db.test_real_set[:num], cfg.OCCLUSION_LINEMOD,
                                   VotingType.Farthest, augment=False, cfg=args.aug_cfg)
        metas = db.test_real_set[:num]

    records = []
    for i, meta in enumerate(metas):
        item = ds[(i, 480, 640)]
        rgb, mask, ver, vw, pose, hcoords = item
        seg, w_map, mask_np, rgb_np, gt_kps, err_b, err_w = predict(
            net, rgb, mask, ver, vw, pose, hcoords, args)
        rec = {'split': split, 'idx': i,
               'rgb_pth': meta.get('rgb_pth', '') if isinstance(meta, dict) else '',
               'rgb': rgb_np, 'gt_kps': gt_kps,
               'err_baseline': err_b.tolist(), 'err_weighted': err_w.tolist(),
               'mean_err_baseline': float(err_b.mean()), 'mean_err_weighted': float(err_w.mean()),
               'w_map': w_map.numpy(), 'mask': mask_np}
        if split == 'OCC':
            rec['amodal_pth'] = amodal_mask_path('cat', meta if isinstance(meta, dict) else {})
        records.append(rec)
        print('{} #{}: mean kp err baseline {:.2f}px weighted {:.2f}px'.format(
            split, i, err_b.mean(), err_w.mean()), flush=True)
    return records


def figure_kp_error(all_records, out_dir):
    import csv
    csv_path = os.path.join(out_dir, 'kp_error_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(['split', 'idx', 'rgb', 'mean_err_baseline_px', 'mean_err_weighted_px',
                       'kp_errs_baseline_px', 'kp_errs_weighted_px'])
        for r in all_records:
            wcsv.writerow([r['split'], r['idx'], r['rgb_pth'],
                           '{:.3f}'.format(r['mean_err_baseline']),
                           '{:.3f}'.format(r['mean_err_weighted']),
                           ' '.join('{:.3f}'.format(v) for v in r['err_baseline']),
                           ' '.join('{:.3f}'.format(v) for v in r['err_weighted'])])

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for split, marker in (('LINEMOD', 'o'), ('OCC', 's')):
        xs = [r['mean_err_baseline'] for r in all_records if r['split'] == split]
        ys = [r['mean_err_weighted'] for r in all_records if r['split'] == split]
        ax.scatter(xs, ys, marker=marker, label=split, s=42)
    lim = max([max(r['mean_err_baseline'], r['mean_err_weighted']) for r in all_records]) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='y = x')
    ax.set_xlabel('baseline (equal-weight) mean kp err (px)')
    ax.set_ylabel('exp001 (weighted) mean kp err (px)')
    ax.set_title('per-image mean keypoint error: voting-level comparison')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'kp_error_comparison.png'), dpi=120)
    plt.close(fig)
    print('saved kp_error_comparison (csv + png)')


def figure_weight_distribution(all_records, out_dir):
    """weights in interior / visible boundary / occluded region / occlusion boundary."""
    def region_weights(rec, region):
        vis = rec['mask'] > 0
        if region == 'interior':
            sel = erode(vis, 5)
        elif region == 'visible_boundary':
            sel = vis ^ erode(vis, 3)          # 3x3 erosion -> 1px boundary ring
        elif region == 'occluded_region':
            am = cv2.imread(rec.get('amodal_pth', ''), 0)
            if am is None:
                return None
            sel = (am > 0) & (~vis)
        elif region == 'occlusion_boundary':
            am = cv2.imread(rec.get('amodal_pth', ''), 0)
            if am is None:
                return None
            amd = (am > 0)
            sel = (amd ^ erode(amd, 3)) & (~vis)   # amodal silhouette edge that is occluded
        if sel.sum() == 0:
            return None
        return rec['w_map'][sel].mean(axis=1)                                # mean over kps

    regions = ['interior', 'visible_boundary', 'occluded_region', 'occlusion_boundary']
    data = {r: [] for r in regions}
    for rec in all_records:
        for rg in regions:
            wv = region_weights(rec, rg)
            if wv is not None and len(wv) > 0:
                data[rg].append(wv)

    stats = {}
    for rg in regions:
        if data[rg]:
            allv = np.concatenate(data[rg])
            stats[rg] = {'n_pixels': int(allv.size),
                         'mean': float(allv.mean()), 'std': float(allv.std()),
                         'p25': float(np.percentile(allv, 25)),
                         'p50': float(np.percentile(allv, 50)),
                         'p75': float(np.percentile(allv, 75))}
        else:
            stats[rg] = None
    with open(os.path.join(out_dir, 'weight_distribution.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    boxes, labels = [], []
    for rg in regions:
        if data[rg]:
            boxes.append(np.concatenate(data[rg]))
            labels.append(rg)
    if boxes:
        bp = ax.boxplot(boxes, tick_labels=labels, showfliers=False)
    ax.set_ylabel('predicted vote weight (mean over kps)')
    ax.set_title('weight distribution by pixel region')
    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'weight_distribution.png'), dpi=120)
    plt.close(fig)
    print('saved weight_distribution (json + png):',
          {k: (v['mean'] if v else None) for k, v in stats.items()})


def main():
    parser = argparse.ArgumentParser(description='exp001 visualization')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--num', type=int, default=8, help='images per split')
    parser.add_argument('--min_weight', type=float, default=0.001)
    parser.add_argument('--round_hyp_num', type=int, default=128)
    parser.add_argument('--vote_max_num', type=int, default=100)
    parser.add_argument('--out_dir', type=str, default=os.path.join(EXP_DIR, 'results', 'figures'))
    args = parser.parse_args()
    args.aug_cfg = json.load(open('configs/exp001_reliability_voting.json'))['aug_cfg']

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    net = load_net(args.ckpt)

    all_records = []
    for split in ('LINEMOD', 'OCC'):
        records = collect_set(net, split, args.num, args)
        all_records += records
        for r in records[:3]:
            figure_heatmap(split, r['idx'], r['rgb'], r['mask'], r['gt_kps'],
                           r['w_map'], out_dir)

    figure_kp_error(all_records, out_dir)
    figure_weight_distribution(all_records, out_dir)
    print('ALL FIGURES SAVED TO', out_dir)


if __name__ == '__main__':
    main()
