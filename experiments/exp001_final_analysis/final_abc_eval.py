"""Final analysis: 3-seed ABC eval for baseline / Exp001-S.
Usage: python /tmp/final_abc_eval.py <model:baseline|exp001s> <epochs...>
Writes experiments/exp001_final_analysis/results/abc_results.csv
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

MODEL = sys.argv[1]
EPS = [int(x) for x in sys.argv[2:]]
train_cfg = json.load(open('configs/exp001_reliability_voting_scratch.json'))
train_cfg['linemod_cls'] = 'cat'
EXP_DIR = 'experiments/exp001_final_analysis'
RES = os.path.join(EXP_DIR, 'results')
os.makedirs(RES, exist_ok=True)
CSV = os.path.join(RES, 'abc_results.csv')
if not os.path.exists(CSV):
    with open(CSV, 'w') as f:
        f.write('model,epoch,split,mode,n_repeats,add_mean,add_std,proj_mean,proj_std,cm_mean,cm_std\n')

db = LineModImageDB('cat', has_render_set=False, has_fuse_set=False)
occ_db = OcclusionLineModImageDB('cat')
half = occ_db.test_real_set[:len(occ_db.test_real_set) // 2]


def make_loader(db_set, data_prefix):
    ds = LineModDatasetRealAug(db_set, data_prefix, VotingType.Farthest, augment=False,
                               cfg=train_cfg['aug_cfg'])
    return DataLoader(ds, batch_sampler=m.ImageSizeBatchSampler(
        SequentialSampler(ds), 1, False, cfg=train_cfg['aug_cfg']), num_workers=6)


def eval_repeat(net_w, eval_wv, eval_base, loader, mode, n_repeats=3):
    am, asd, pm, psd, cm, csd = [], [], [], [], [], []
    for seed in range(n_repeats):
        torch.manual_seed(seed)
        np.random.seed(seed)
        ev = m.Evaluator()
        with torch.no_grad():
            for data in loader:
                image, mask, vertex, vertex_weights, pose, gt_kps = [d.cuda() for d in data]
                gt2 = gt_kps[:, :, :2]
                out = net_w(image, mask, vertex, vertex_weights, gt2)
                if mode == 'weighted':
                    corner = eval_wv(out[0], out[1], out[2]).cpu().numpy()
                else:
                    corner = eval_base(out[0], out[1]).cpu().numpy()
                pose_np = pose.cpu().numpy()
                for bi in range(pose_np.shape[0]):
                    ev.evaluate(corner[bi], pose_np[bi], 'cat', 'linemod',
                                VotingType.Farthest, intri_matrix=None)
        proj, add, cm_ = ev.average_precision(False)
        am.append(add); asd.append(0); pm.append(proj); psd.append(0); cm.append(cm_); csd.append(0)
    return (float(np.mean(am)), float(np.std(am)), float(np.mean(pm)), float(np.std(pm)),
            float(np.mean(cm)), float(np.std(cm)))


for ep in EPS:
    if MODEL == 'baseline':
        net = Resnet18_8s(ver_dim=18, seg_dim=2)
        sd = torch.load('data/model/cat_linemod_train/{}.pth'.format(ep), map_location='cpu')
        wrapper = torch.nn.DataParallel(m.NetWrapperRel(net, False, 1.0)).cuda()
        wrapper.module.net.load_state_dict(sd['net'])
        modes = ('uniform',)
    else:
        net = Resnet18_8sReliability(ver_dim=18, seg_dim=2)
        sd = torch.load('experiments/exp001_reliability_voting_scratch/model/cat_exp001s_scratch/{}.pth'.format(ep),
                        map_location='cpu')
        wrapper = torch.nn.DataParallel(m.NetWrapperRel(net, True, train_cfg['reliability_sigma_px'],
                                                        train_cfg['reliability_tau1_px'],
                                                        train_cfg['reliability_tau2_px'])).cuda()
        wrapper.module.net.load_state_dict(sd['net'])
        modes = ('uniform', 'weighted')
    wrapper.eval()
    eval_wv = torch.nn.DataParallel(m.EvalWrapperRel(train_cfg).cuda())
    eval_base = torch.nn.DataParallel(m.EvalWrapperBaseline(train_cfg).cuda())
    for split, db_set, prefix in (('linemod_val', db.val_real_set, cfg.LINEMOD),
                                  ('occ_val', half, cfg.OCCLUSION_LINEMOD)):
        loader = make_loader(db_set, prefix)
        for mode in modes:
            am, asd, pm, psd, cm, csd = eval_repeat(wrapper, eval_wv, eval_base, loader, mode)
            with open(CSV, 'a') as f:
                f.write('{},{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(
                    MODEL, ep, split, mode, 3, am, asd, pm, psd, cm, csd))
            print('{} e{} {} {}: add {:.4f}+-{:.4f}'.format(MODEL, ep, split, mode, am, asd),
                  flush=True)
    del wrapper, eval_wv, eval_base
    torch.cuda.empty_cache()
print('done')
