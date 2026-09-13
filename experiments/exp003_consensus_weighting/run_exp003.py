"""EXP003 driver: pass-1 baseline voting (verified) -> consensus reliability ->
weighted round-2 voting (A4/A6/B4) -> per-image CSV + metric JSON.

Usage (from repo root):
  python experiments/exp003_consensus_weighting/run_exp003.py --split linemod \
      --n_images 10 --seeds 0 --uniform_check --tag smoke
"""
import os, sys, json, argparse, time, importlib.util
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler

from lib.networks.model_repository import Resnet18_8s
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v2
from lib.datasets.linemod_dataset import LineModDatasetRealAug, VotingType
from lib.utils.data_utils import LineModImageDB, OcclusionLineModImageDB
from lib.utils.config import cfg
from lib.utils.evaluation_utils import Evaluator
from consensus_weighting import (get_voting_problem, consensus_reliability,
                                 weighted_ransac_all)

_spec = importlib.util.spec_from_file_location(
    'tle', os.path.join(ROOT, 'tools', 'train_linemod_exp001.py'))
_tle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tle)
ImageSizeBatchSampler = _tle.ImageSizeBatchSampler

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CONF = json.load(open(os.path.join(EXP_DIR, 'config.json')))
RC = CONF['ransac']
X3 = CONF['exp003']


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['linemod', 'occ'], required=True)
    p.add_argument('--n_images', type=int, default=None)
    p.add_argument('--seeds', type=str, default='0')
    p.add_argument('--methods', type=str, default='baseline,' + ','.join(X3['methods']))
    p.add_argument('--uniform_check', action='store_true',
                   help='smoke: verify round-2 with w=1 reproduces baseline (<1e-5)')
    p.add_argument('--tag', type=str, default='run')
    return p.parse_args()


def build_split(split, n_images):
    aug_cfg = json.load(open(os.path.join(
        ROOT, 'configs', 'exp001_reliability_voting_scratch.json')))['aug_cfg']
    if split == 'linemod':
        db = LineModImageDB('cat', has_render_set=False, has_fuse_set=False)
        db_set, prefix = db.val_real_set, cfg.LINEMOD
    else:
        db = OcclusionLineModImageDB('cat')
        db_set = db.test_real_set[:len(db.test_real_set) // 2]
        prefix = cfg.OCCLUSION_LINEMOD
    if n_images:
        db_set = db_set[:n_images]
    ds = LineModDatasetRealAug(db_set, prefix, VotingType.Farthest, augment=False, cfg=aug_cfg)
    loader = DataLoader(ds, batch_sampler=ImageSizeBatchSampler(
        SequentialSampler(ds), 1, False, aug_cfg), num_workers=6)
    return db_set, loader


def reliability_for(m_key, coords, direct_k, coarse_k):
    if m_key == 'U1':
        return torch.ones([coords.shape[0]], device=coords.device)
    if m_key == 'A4':
        r, _ = consensus_reliability(coords, direct_k, coarse_k,
                                     tau=X3['tau_hard'][0], mode='hard')
    elif m_key == 'A6':
        r, _ = consensus_reliability(coords, direct_k, coarse_k,
                                     tau=X3['tau_hard'][1], mode='hard')
    else:  # B4
        r, _ = consensus_reliability(coords, direct_k, coarse_k,
                                     sigma=X3['sigma_soft'], mode='soft',
                                     weight_clip=X3['weight_clip'])
    return r


def main():
    args = get_args()
    device = 'cuda'
    net = Resnet18_8s(ver_dim=18, seg_dim=2)
    sd = torch.load(os.path.join(ROOT, CONF['checkpoint']), map_location='cpu')
    net.load_state_dict(sd['net'])
    net = net.to(device).eval()

    db_set, loader = build_split(args.split, args.n_images)
    methods = args.methods.split(',')
    if args.uniform_check and 'U1' not in methods:
        methods = ['U1'] + methods
    seeds = [int(s) for s in args.seeds.split(',')]

    res_dir = os.path.join(EXP_DIR, 'results')
    os.makedirs(res_dir, exist_ok=True)
    csv_path = os.path.join(res_dir, 'exp003_{}_{}.csv'.format(args.split, args.tag))
    json_path = os.path.join(res_dir, 'metrics_{}_{}.json'.format(args.split, args.tag))
    csv_f = open(csv_path, 'w')
    csv_f.write('method,seed,image_id,keypoint_id,coarse_x,coarse_y,final_x,final_y,'
                'num_votes,num_reliable,mean_vote_distance,median_vote_distance,fallback\n')
    summary = {}

    for seed in seeds:
        evaluators = {m: Evaluator() for m in methods}
        fallback_counts = {m: {} for m in methods}
        t0 = time.time()
        maxdiff_reported = 0.0
        for image_id, data in enumerate(loader):
            image, mask_gt, vertex_gt, vertex_weights, pose, gt_kps = [d.to(device) for d in data]
            pose_np = pose.cpu().numpy()
            with torch.no_grad():
                seg_pred, vertex_pred = net(image)
                mask = torch.argmax(seg_pred, 1)
                b, _, h, w = vertex_pred.shape
                ver = vertex_pred.permute(0, 2, 3, 1).contiguous().view(b, h, w, -1, 2)

                def base_call():
                    return ransac_voting_layer_v2(
                        mask, ver, RC['class_num'], RC['round_hyp_num'],
                        inlier_thresh=RC['inlier_thresh'], confidence=RC['confidence'],
                        max_iter=RC['max_iter'], min_num=RC['min_num'],
                        max_num=RC['max_num'], refine_iter_num=RC['refine_iter_num'])

                # pass-1: original baseline voting (twice, same seed -> equivalence check)
                torch.manual_seed(seed)
                corners_base = base_call()
                torch.manual_seed(seed)
                corners_p1 = base_call()
                maxdiff = float((corners_base - corners_p1).abs().max().item())
                maxdiff_reported = max(maxdiff_reported, maxdiff)
                if maxdiff >= X3['pass1_tol']:
                    csv_f.close()
                    raise RuntimeError('pass1 != baseline (diff {:.2e} >= {:.2e}) at image '
                                       '{}: EXPERIMENT ABORTED'.format(maxdiff, X3['pass1_tol'],
                                                                       image_id))

                # replicate sampling for round-2 under the same RNG stream
                torch.manual_seed(seed)
                problem = get_voting_problem(mask, ver, RC['class_num'],
                                             RC['round_hyp_num'], max_num=RC['max_num'],
                                             min_num=RC['min_num'])
                coarse = corners_p1[0, 0]                     # [9,2] cuda

                def weights_for(m_key):
                    ws = []
                    for k in range(9):
                        if problem is None:
                            ws.append(torch.zeros([1], device=device))
                        elif m_key == 'U1':
                            ws.append(torch.ones([problem['tn']], device=device))
                        else:
                            r_k, _ = consensus_reliability(
                                problem['coords'], problem['direct'][:, k, :], coarse[k],
                                tau=X3['tau_hard'][0] if m_key == 'A4' else
                                    (X3['tau_hard'][1] if m_key == 'A6' else None),
                                sigma=X3['sigma_soft'], mode='hard' if m_key in ('A4', 'A6')
                                else 'soft', weight_clip=X3['weight_clip'])
                            ws.append(r_k)
                    return torch.stack(ws, 0) if problem is not None else None

                # U1 control: weighted pipeline with w=1 must equal baseline
                if 'U1' in methods:
                    if problem is None:
                        u_diff = 0.0
                    else:
                        finals_u, _ = weighted_ransac_all(
                            problem, weights_for('U1'), coarse, RC,
                            min_votes=X3['min_votes'])
                        u_diff = float((finals_u - coarse).abs().max().item())
                    if u_diff >= X3['pass1_tol']:
                        csv_f.close()
                        raise RuntimeError('U1(w=1) != baseline (diff {:.2e}): sampling '
                                           'replication FAILED, aborting'.format(u_diff))

                for m_key in methods:
                    if m_key == 'baseline':
                        finals = coarse.clone()
                        for k in range(9):
                            csv_f.write('{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{},{},{:.4f},{:.4f},{}\n'.format(
                                m_key, seed, image_id, k,
                                float(coarse[k, 0]), float(coarse[k, 1]),
                                float(coarse[k, 0]), float(coarse[k, 1]), 0, 0, -1.0, -1.0, ''))
                    else:
                        if problem is None:
                            finals = coarse.clone()
                            for k in range(9):
                                fallback_counts[m_key]['no_fg'] = \
                                    fallback_counts[m_key].get('no_fg', 0) + 1
                                csv_f.write('{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{},{},{:.4f},{:.4f},{}\n'.format(
                                    m_key, seed, image_id, k,
                                    float(coarse[k, 0]), float(coarse[k, 1]),
                                    float(coarse[k, 0]), float(coarse[k, 1]),
                                    0, 0, -1.0, -1.0, 'no_fg'))
                            final_np = finals.cpu().numpy()
                            evaluators[m_key].evaluate(final_np, pose_np[0],
                                                       CONF['eval']['linemod_cls'],
                                                       CONF['eval']['dataset_type'],
                                                       VotingType.Farthest, intri_matrix=None)
                            continue
                        weights_all = weights_for(m_key)      # [9,tn]
                        finals, stats_all = weighted_ransac_all(
                            problem, weights_all, coarse, RC, min_votes=X3['min_votes'])
                        for k in range(9):
                            st = stats_all[k]
                            if st['fallback']:
                                fallback_counts[m_key][st['fallback']] = \
                                    fallback_counts[m_key].get(st['fallback'], 0) + 1
                            csv_f.write('{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{},{},{:.4f},{:.4f},{}\n'.format(
                                m_key, seed, image_id, k,
                                float(coarse[k, 0]), float(coarse[k, 1]),
                                float(finals[k, 0]), float(finals[k, 1]),
                                st['num_votes'], st['num_reliable'],
                                st['mean_d'], st['median_d'], st['fallback'] or ''))
                    final_np = finals.cpu().numpy()          # [1,9,2]
                    evaluators[m_key].evaluate(final_np, pose_np[0],
                                               CONF['eval']['linemod_cls'],
                                               CONF['eval']['dataset_type'],
                                               VotingType.Farthest, intri_matrix=None)
                csv_f.flush()
            if image_id % 50 == 0:
                print('split {} seed {} image {}/{} ({:.1f}s) pass1_diff {:.2e}'.format(
                    args.split, seed, image_id, len(loader), time.time() - t0,
                    maxdiff_reported), flush=True)

        summary[str(seed)] = {}
        for m_key in methods:
            proj, add, cm = evaluators[m_key].average_precision(False)
            n_img = len(loader)
            fb_total = sum(fallback_counts[m_key].values())
            summary[str(seed)][m_key] = dict(add=add, projection_error=proj,
                                             cm_degree_5=cm, n_images=n_img,
                                             fallback_counts=fallback_counts[m_key],
                                             fallback_rate=fb_total / max(n_img, 1))
            print('seed {} {}: ADD(-S) {:.4f} proj {:.4f} 5cm5deg {:.4f} fallback_rate {:.4f}'.format(
                seed, m_key, add, proj, cm, fb_total / max(n_img, 1)), flush=True)
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        csv_f.flush()
    csv_f.close()
    print('DONE -> {}'.format(json_path))


if __name__ == '__main__':
    main()
