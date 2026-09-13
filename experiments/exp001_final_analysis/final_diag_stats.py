"""Final analysis: vote-error distribution stats per (model, epoch, split).
Usage: python /tmp/final_diag_stats.py <model:baseline|exp001s> <epochs...>
Writes experiments/exp001_final_analysis/results/diag_results.csv
"""
import sys, os, json
sys.path.append('.')
sys.path.append('..')

import torch
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler

import importlib.util
spec = importlib.util.spec_from_file_location('e', 'tools/train_linemod_exp001.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

from lib.networks.model_repository import Resnet18_8s
from lib.networks.model_repository_exp001 import Resnet18_8sReliability
from lib.datasets.linemod_dataset import LineModDatasetRealAug, VotingType
from lib.utils.data_utils import LineModImageDB, OcclusionLineModImageDB
from lib.utils.config import cfg
from tools.train_linemod_exp001 import pseudo_reliability

MODEL = sys.argv[1]
EPS = [int(x) for x in sys.argv[2:]]
N_IMG = 200
train_cfg = json.load(open('configs/exp001_reliability_voting_scratch.json'))
train_cfg['linemod_cls'] = 'cat'
RES = 'experiments/exp001_final_analysis/results'
os.makedirs(RES, exist_ok=True)
CSV = os.path.join(RES, 'diag_results.csv')
if not os.path.exists(CSV):
    with open(CSV, 'w') as f:
        f.write('model,epoch,split,n_images,n_px,e_mean,e_median,e_p90,bad_ratio_gt10px\n')

db = LineModImageDB('cat', has_render_set=False, has_fuse_set=False)
occ_db = OcclusionLineModImageDB('cat')
half = occ_db.test_real_set[:len(occ_db.test_real_set) // 2]

for ep in EPS:
    if MODEL == 'baseline':
        net = Resnet18_8s(ver_dim=18, seg_dim=2)
        sd = torch.load('data/model/cat_linemod_train/{}.pth'.format(ep), map_location='cpu')
        wrapper = torch.nn.DataParallel(m.NetWrapperRel(net, False, 1.0)).cuda()
        wrapper.module.net.load_state_dict(sd['net'])
    else:
        net = Resnet18_8sReliability(ver_dim=18, seg_dim=2)
        sd = torch.load('experiments/exp001_reliability_voting_scratch/model/cat_exp001s_scratch/{}.pth'.format(ep),
                        map_location='cpu')
        wrapper = torch.nn.DataParallel(m.NetWrapperRel(net, True, train_cfg['reliability_sigma_px'],
                                                        train_cfg['reliability_tau1_px'],
                                                        train_cfg['reliability_tau2_px'])).cuda()
        wrapper.module.net.load_state_dict(sd['net'])
    wrapper.eval()
    for split, db_set, prefix in (('linemod_val', db.val_real_set[:N_IMG], cfg.LINEMOD),
                                  ('occ_val', half[:N_IMG], cfg.OCCLUSION_LINEMOD)):
        ds = LineModDatasetRealAug(db_set, prefix, VotingType.Farthest, augment=False,
                                   cfg=train_cfg['aug_cfg'])
        loader = DataLoader(ds, batch_sampler=m.ImageSizeBatchSampler(
            SequentialSampler(ds), 1, False, cfg=train_cfg['aug_cfg']), num_workers=6)
        es = []
        with torch.no_grad():
            for data in loader:
                image, mask, vertex, vertex_weights, pose, gt_kps = [d.cuda() for d in data]
                gt2 = gt_kps[:, :, :2]
                out = wrapper(image, mask, vertex, vertex_weights, gt2)
                _, e_all = pseudo_reliability(out[1].detach(), gt2, vertex_weights, 3.0)
                fg = vertex_weights.bool()
                if fg.dim() == 4 and fg.shape[1] == 1:
                    fg = fg[:, 0].unsqueeze(1).expand_as(e_all)
                else:
                    fg = fg.expand_as(e_all)
                es.append(e_all[fg].cpu().numpy())
        e = np.concatenate(es)
        row = (MODEL, ep, split, len(db_set), len(e), float(e.mean()),
               float(np.median(e)), float(np.percentile(e, 90)),
               float((e > 10).mean()))
        with open(CSV, 'a') as f:
            f.write('{},{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(*row))
        print('{} e{} {}: mean {:.3f} med {:.3f} p90 {:.3f} bad>10px {:.4f} (n={})'.format(
            MODEL, ep, split, row[5], row[6], row[7], row[8], row[4]), flush=True)
    del wrapper
    torch.cuda.empty_cache()
print('done')
