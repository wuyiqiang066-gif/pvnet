"""EXP007 training: Control A (baseline 20ep) vs Context (EXP007 20ep), seed=0.

Protocol identical to baseline tools/train_linemod.py (configs/linemod_train.json):
Adam lr=1e-3, decay 0.5/20ep, batch 32 (ImageSizeBatchSampler), Farthest votes,
augment=True with baseline aug_cfg, vertex_loss_ratio=1.0, CrossEntropy(seg) +
smooth_l1(vertex). Differences mandated by the spec: 20 real-train images only,
20 epochs, init from baseline 199.pth, seed 0, num_workers=0 (determinism).

Run from repo root:
  python experiments/exp007_context_vertex/train_exp007.py --model baseline --tag controlA
  python experiments/exp007_context_vertex/train_exp007.py --model context  --tag exp007
"""
import os, sys, json, csv, time, argparse, random
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, RandomSampler

from lib.networks.model_repository import Resnet18_8s
from lib.datasets.linemod_dataset import LineModDatasetRealAug, ImageSizeBatchSampler, VotingType
from lib.utils.data_utils import LineModImageDB
from lib.utils.config import cfg
from lib.utils.net_utils import smooth_l1_loss, adjust_learning_rate

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(EXP_DIR, 'checkpoints')
SEED = 0
EPOCHS = 20
N_TRAIN_IMG = 20
CLS = 'cat'
VOTE_NUM = 9

TRAIN_CFG = json.load(open(os.path.join(ROOT, 'configs', 'linemod_train.json')))

class NetLoss(nn.Module):
    """Verbatim replica of baseline train_linemod.NetWrapper loss computation."""
    def __init__(self, net):
        super(NetLoss, self).__init__()
        self.net = net
        self.criterion = nn.CrossEntropyLoss(reduce=False)

    def forward(self, image, mask, vertex, vertex_weights):
        seg_pred, vertex_pred = self.net(image)
        loss_seg = self.criterion(seg_pred, mask)
        loss_seg = torch.mean(loss_seg.view(loss_seg.shape[0], -1), 1)
        loss_vertex = smooth_l1_loss(vertex_pred, vertex, vertex_weights, reduce=False)
        loss = torch.mean(loss_seg) + torch.mean(loss_vertex) * TRAIN_CFG['vertex_loss_ratio']
        return seg_pred, vertex_pred, loss_seg, loss_vertex, loss


def build_model(model_type):
    torch.manual_seed(SEED)  # identical init RNG stream for both arms
    if model_type == 'baseline':
        net = Resnet18_8s(ver_dim=VOTE_NUM * 2, seg_dim=2)
        sd = torch.load(os.path.join(ROOT, 'data/model/cat_linemod_train/199.pth'),
                        map_location='cpu')['net']
        net.load_state_dict(sd)
        info = dict(model_type='baseline', params_total=sum(p.numel() for p in net.parameters()),
                    newly_added=0)
    else:
        from context_model import Resnet18_8sContext, load_from_baseline, count_params
        net = Resnet18_8sContext(ver_dim=VOTE_NUM * 2, seg_dim=2)
        sd = torch.load(os.path.join(ROOT, 'data/model/cat_linemod_train/199.pth'),
                        map_location='cpu')['net']
        load_from_baseline(net, sd)
        info = dict(model_type='context', **count_params(net))
    return net.cuda(), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['baseline', 'context'], required=True)
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    net, info = build_model(args.model)
    print('model init:', json.dumps(info))

    out_dir = os.path.join(CKPT_DIR, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(EXP_DIR, 'results'), exist_ok=True)
    json.dump(info, open(os.path.join(out_dir, 'init_info.json'), 'w'), indent=2)

    # --- data: exactly the same stream for both arms (re-seed before loader) ---
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    image_db = LineModImageDB(CLS, has_render_set=False, has_fuse_set=False)
    train_db = image_db.train_real_set[:N_TRAIN_IMG]
    train_set = LineModDatasetRealAug(train_db, cfg.LINEMOD, VotingType.Farthest,
                                      augment=True, cfg=TRAIN_CFG['aug_cfg'])
    train_loader = DataLoader(train_set,
                              batch_sampler=ImageSizeBatchSampler(
                                  RandomSampler(train_set),
                                  TRAIN_CFG['train_batch_size'], False,
                                  cfg=TRAIN_CFG['aug_cfg']),
                              num_workers=0)
    print(f'train images: {len(train_set)}, batches/epoch: {len(train_loader)}')

    optimizer = optim.Adam(net.parameters(), lr=TRAIN_CFG['lr'])
    wrapper = NetLoss(net).cuda()
    net.train()
    log_path = os.path.join(EXP_DIR, 'results', f'train_log_{args.tag}.csv')
    with open(log_path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['epoch', 'lr', 'loss_seg', 'loss_vertex', 'loss_total', 'seconds'])
        t_start = time.time()
        for epoch in range(EPOCHS):
            adjust_learning_rate(optimizer, epoch, TRAIN_CFG['lr_decay_rate'],
                                 TRAIN_CFG['lr_decay_epoch'])
            ep_seg, ep_ver, ep_tot, ep_t0, nb = 0., 0., 0., time.time(), 0
            for data in train_loader:
                image, mask, vertex, vertex_weights, pose, _ = [d.cuda() for d in data]
                seg_pred, vertex_pred, loss_seg, loss_vertex, loss = wrapper(
                    image, mask, vertex, vertex_weights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_seg += float(loss_seg.mean()); ep_ver += float(loss_vertex.mean())
                ep_tot += float(loss); nb += 1
            wr.writerow([epoch, optimizer.param_groups[0]['lr'],
                         f'{ep_seg/max(nb,1):.5f}', f'{ep_ver/max(nb,1):.5f}',
                         f'{ep_tot/max(nb,1):.5f}', f'{time.time()-ep_t0:.1f}'])
            f.flush()
            print(f'[{args.tag}] epoch {epoch}: seg {ep_seg/max(nb,1):.4f} '
                  f'ver {ep_ver/max(nb,1):.4f} lr {optimizer.param_groups[0]["lr"]:.1e}', flush=True)
            torch.save({'net': net.state_dict(), 'epoch': epoch},
                       os.path.join(out_dir, 'last.pth'))
        print(f'[{args.tag}] done in {time.time()-t_start:.0f}s')


if __name__ == '__main__':
    main()
