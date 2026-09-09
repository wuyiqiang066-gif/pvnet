"""
exp001_reliability_voting: Reliability-aware Pixel Voting (standalone entry).

The original `tools/train_linemod.py` and the baseline network/dataset/PnP/eval
are NOT modified. Set `use_reliability_vote: false` in the config to recover the
exact baseline behaviour (same model class, losses, equal-weight RANSAC voting)
from this same entry.

Innovation (additive only):
  model   : Resnet18_8sReliability -> extra head, output [B,9,H,W] reliability
            logits (one channel per keypoint). Seg/vertex branches untouched.
  training: pseudo reliability r* = exp(-e^2/sigma^2), where e = perpendicular
            distance from the GT keypoint to the pixel's PREDICTED vote line
            (computed on the fly from the dataset's 6th output `hcoords`, i.e.
            the augmented GT keypoints; the vertex prediction is detached so the
            baseline branch receives no gradient from L_rel).
            L_total = L_seg + w_ver * L_vertex + lambda * L_reliability
  inference: same RANSAC framework (same CUDA kernels, same confidence loop),
            weighted hypothesis scoring score(h,k) = sum_i r_i^k * inlier(i,h,k)
            and weighted least-squares refinement.

Unchanged: dataset classes, PnP, evaluation metrics.
Outputs under experiments/exp001_reliability_voting/:
    config.json                      resolved config snapshot
    record/<model_name>.log          text log (+ tensorboard events)
    model/<model_name>/<epoch>.pth   checkpoints
    result_<split>_e<epoch>.txt      evaluation metrics
    result_summary.txt               appended evaluation history
"""
import sys, os, json, time, argparse

sys.path.append('.')
sys.path.append('..')

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from collections import OrderedDict

from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v3
from lib.ransac_voting_gpu_layer.ransac_voting_weighted import ransac_voting_layer_v3_weighted
from lib.networks.model_repository import Resnet18_8s
from lib.networks.model_repository_exp001 import Resnet18_8sReliability
from lib.datasets.linemod_dataset import LineModDatasetRealAug, ImageSizeBatchSampler, VotingType
from lib.utils.data_utils import LineModImageDB, OcclusionLineModImageDB
from lib.utils.evaluation_utils import Evaluator
from lib.utils.net_utils import (AverageMeter, Recorder, smooth_l1_loss,
                                 load_model, save_model, adjust_learning_rate,
                                 compute_precision_recall)
from lib.utils.config import cfg

EXP_DIR = 'experiments/exp001_reliability_voting'
VOTE_TYPE = VotingType.Farthest   # exp001 fixes Farthest voting (9 keypoints)
VOTE_NUM = 9


# ---------------- pseudo reliability label ----------------
def to_kp_px(gt_kps):
    """dataset `hcoords` are homogeneous (x, y, w) -> pixel (x, y)."""
    if gt_kps.dim() == 3 and gt_kps.shape[-1] == 3:
        return gt_kps[..., :2] / torch.clamp(gt_kps[..., 2:], min=1e-6)
    return gt_kps


def pseudo_reliability(vertex_pred, gt_kps, fg_mask, sigma_px, tau1_px=0.0, tau2_px=0.0):
    """r* = exp(-e^2/sigma^2); e = perp distance from GT kp to the PREDICTED vote line.

    vertex_pred: (B,2*vn,H,W) network output (call with .detach())
    gt_kps:      (B,vn,2) GT keypoint pixels (x, y)
    fg_mask:     (B,1,H,W) foreground indicator == dataset `vertex_weights`
    tau1_px:     e <= tau1 -> r* forced to 1 (high confidence, avoids saturation)
    tau2_px:     e >= tau2 -> r* forced to 0 (zero confidence, avoids float underflow)
    returns (r_star, e_all): each (B,vn,H,W)
    """
    B, _, H, W = vertex_pred.shape
    vn = gt_kps.shape[1]
    ver = vertex_pred.permute(0, 2, 3, 1).view(B, H, W, vn, 2)
    ys, xs = torch.meshgrid(torch.arange(H, device=vertex_pred.device),
                            torch.arange(W, device=vertex_pred.device), indexing='ij')
    p = torch.stack([xs, ys], -1).float()                                    # (H,W,2) as (x,y)
    e_all = torch.zeros(B, vn, H, W, device=vertex_pred.device)
    for k in range(vn):   # loop over keypoints to keep peak memory low
        d = ver[:, :, :, k, :]                                               # (B,H,W,2)
        dn = d / (d.norm(dim=-1, keepdim=True) + 1e-8)                       # unit vote line
        v = gt_kps[:, k].view(B, 1, 1, 2) - p                                # (B,H,W,2)
        cross = dn[..., 0] * v[..., 1] - dn[..., 1] * v[..., 0]              # perp distance (px)
        e_all[:, k] = cross.abs()
    r_star = torch.exp(-e_all * e_all / (sigma_px * sigma_px))
    if tau1_px and tau1_px > 0:
        r_star = torch.where(e_all <= tau1_px, torch.ones_like(r_star), r_star)
    if tau2_px and tau2_px > 0:
        r_star = torch.where(e_all >= tau2_px, torch.zeros_like(r_star), r_star)
    return r_star, e_all


def reliability_label_stats(r_star, fg_mask, bins=32):
    """per-batch statistics of the pseudo labels over foreground pixels."""
    m = fg_mask.bool().expand_as(r_star)
    v = r_star[m].float()
    if v.numel() == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'num': 0,
                'hist': [0] * bins, 'bin_edges': [i / bins for i in range(bins + 1)]}
    hist = torch.histc(v, bins=bins, min=0.0, max=1.0)
    return {'mean': v.mean().item(), 'std': v.std().item(),
            'min': v.min().item(), 'max': v.max().item(), 'num': int(v.numel()),
            'hist': [int(x) for x in hist.tolist()],
            'bin_edges': [i / bins for i in range(bins + 1)]}


# ---------------- train wrapper ----------------
class NetWrapperRel(nn.Module):
    """Train/test wrapper. use_rel=False reproduces the baseline exactly."""

    def __init__(self, net, use_rel, sigma_px, tau1_px=0.0, tau2_px=0.0):
        super(NetWrapperRel, self).__init__()
        self.net = net
        self.use_rel = use_rel
        self.sigma_px = sigma_px
        self.tau1_px = tau1_px
        self.tau2_px = tau2_px
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, image, mask, vertex, vertex_weights, gt_kps):
        gt_kps = to_kp_px(gt_kps)

        if not self.use_rel:
            seg_pred, vertex_pred = self.net(image)
            loss_seg = self.criterion(seg_pred, mask)
            loss_seg = torch.mean(loss_seg.view(loss_seg.shape[0], -1), 1)
            loss_vertex = smooth_l1_loss(vertex_pred, vertex, vertex_weights, reduce=False)
            precision, recall = compute_precision_recall(seg_pred, mask)
            zero = torch.zeros(1, device=image.device).squeeze()
            return seg_pred, vertex_pred, zero, loss_seg, loss_vertex, zero, precision, recall

        seg_pred, vertex_pred, rel_logits = self.net(image)
        loss_seg = self.criterion(seg_pred, mask)
        loss_seg = torch.mean(loss_seg.view(loss_seg.shape[0], -1), 1)
        loss_vertex = smooth_l1_loss(vertex_pred, vertex, vertex_weights, reduce=False)
        precision, recall = compute_precision_recall(seg_pred, mask)

        with torch.no_grad():   # detached vertex -> baseline branch gets no gradient from L_rel
            r_star, _ = pseudo_reliability(vertex_pred.detach(), gt_kps, vertex_weights,
                                           self.sigma_px, self.tau1_px, self.tau2_px)
        loss_rel = F.binary_cross_entropy_with_logits(rel_logits, r_star, reduction='none')
        m = vertex_weights.float()                                # (B,1,H,W)
        vn = r_star.shape[1]
        loss_rel = (loss_rel * m).sum() / (m.sum() * vn + 1e-8)   # mean over fg pixels x kps
        return seg_pred, vertex_pred, rel_logits, loss_seg, loss_vertex, loss_rel, precision, recall


# ---------------- eval wrappers ----------------
class EvalWrapperBaseline(nn.Module):
    """Verbatim copy of the baseline EvalWrapper (plain pnp path)."""

    def __init__(self, train_cfg):
        super(EvalWrapperBaseline, self).__init__()
        self.round_hyp_num = train_cfg['vote_round_hyp_num']
        self.max_num = train_cfg['vote_max_num']

    def forward(self, seg_pred, vertex_pred, use_argmax=True):
        vertex_pred = vertex_pred.permute(0, 2, 3, 1)
        b, h, w, vn_2 = vertex_pred.shape
        vertex_pred = vertex_pred.view(b, h, w, vn_2 // 2, 2)
        mask = torch.argmax(seg_pred, 1) if use_argmax else seg_pred
        return ransac_voting_layer_v3(mask, vertex_pred, self.round_hyp_num,
                                      inlier_thresh=0.99, max_num=self.max_num)


class EvalWrapperRel(nn.Module):
    """Weighted-voting eval: reliability per pixel per keypoint."""

    def __init__(self, train_cfg):
        super(EvalWrapperRel, self).__init__()
        self.round_hyp_num = train_cfg['vote_round_hyp_num']
        self.max_num = train_cfg['vote_max_num']
        self.min_weight = train_cfg['reliability_min_weight']

    def forward(self, seg_pred, vertex_pred, rel_logits, use_argmax=True):
        b, _, h, w = seg_pred.shape
        vn = rel_logits.shape[1]
        vertex_pred = vertex_pred.permute(0, 2, 3, 1).view(b, h, w, vn, 2)
        mask = torch.argmax(seg_pred, 1) if use_argmax else seg_pred
        weights = torch.sigmoid(rel_logits).permute(0, 2, 3, 1)   # (b,h,w,vn)
        return ransac_voting_layer_v3_weighted(
            mask, vertex_pred, weights, self.round_hyp_num,
            inlier_thresh=0.99, max_num=self.max_num, min_weight=self.min_weight)


# ---------------- train / val ----------------
def set_stage1_mode(dp_net, freeze=True):
    """stage1: freeze backbone+vertex head (params AND BN running stats), train rel_head only."""
    net_mod = dp_net.module if hasattr(dp_net, 'module') else dp_net
    for name, p in net_mod.named_parameters():
        p.requires_grad = (not freeze) or ('rel_head' in name)
    if freeze:
        for name, m in net_mod.named_modules():
            if 'rel_head' not in name:
                m.eval()   # stop BN running-stat updates in the frozen branch


def save_label_stats(epoch, stage, stats_list, train_cfg):
    """persist per-epoch pseudo-label statistics (Part 4)."""
    if len(stats_list) == 0:
        return
    bins = len(stats_list[0]['hist'])
    hist = np.mean([s['hist'] for s in stats_list], axis=0).astype(np.int64)
    out = {'epoch': epoch, 'stage': stage,
           'mean': float(np.mean([s['mean'] for s in stats_list])),
           'std': float(np.mean([s['std'] for s in stats_list])),
           'min': float(np.min([s['min'] for s in stats_list])),
           'max': float(np.max([s['max'] for s in stats_list])),
           'num_pixels': int(np.mean([s['num'] for s in stats_list])),
           'hist': hist.tolist(),
           'bin_edges': stats_list[0]['bin_edges']}
    results_dir = os.path.join(EXP_DIR, 'results')
    os.makedirs(os.path.join(results_dir, 'label_hist'), exist_ok=True)
    with open(os.path.join(results_dir, 'reliability_label_stats.jsonl'), 'a') as f:
        f.write(json.dumps(out) + '\n')
    np.savez(os.path.join(results_dir, 'label_hist', 'epoch_{:03d}.npz'.format(epoch)),
             hist=hist, bin_edges=np.array(out['bin_edges']))
    print('label stats: epoch {} stage {} mean {:.4f} std {:.4f} min {:.4f} max {:.4f}'.format(
        epoch, stage, out['mean'], out['std'], out['min'], out['max']), flush=True)


def train(net, optimizer, dataloader, epoch, train_cfg, recorder, stage=2):
    seg_loss_rec, ver_loss_rec, rel_loss_rec = AverageMeter(), AverageMeter(), AverageMeter()
    precision_rec, recall_rec = AverageMeter(), AverageMeter()
    recs = [seg_loss_rec, ver_loss_rec, rel_loss_rec, precision_rec, recall_rec]
    recs_names = ['scalar/seg', 'scalar/ver', 'scalar/rel', 'scalar/precision', 'scalar/recall']
    for rec in recs: rec.reset()

    rel_ratio = train_cfg['reliability_loss_ratio']
    if epoch < train_cfg.get('reliability_warmup_epoch', 0):
        rel_ratio = 0.0

    stats_interval = train_cfg['loss_rec_step']
    stats_list = []

    train_begin = time.time()
    net.train()
    if stage == 1:   # re-apply frozen-BN eval mode after net.train()
        set_stage1_mode(net, True)
    size = len(dataloader)
    end = time.time()
    step = 0
    for idx, data in enumerate(dataloader):
        image, mask, vertex, vertex_weights, pose, gt_kps = [d.cuda() for d in data]
        seg_pred, vertex_pred, rel_logits, loss_seg, loss_vertex, loss_rel, precision, recall = \
            net(image, mask, vertex, vertex_weights, gt_kps)
        loss_seg, loss_vertex, loss_rel, precision, recall = \
            [torch.mean(v) for v in (loss_seg, loss_vertex, loss_rel, precision, recall)]
        loss = loss_seg + loss_vertex * train_cfg['vertex_loss_ratio'] + loss_rel * rel_ratio
        vals = (loss_seg, loss_vertex, loss_rel, precision, recall)
        for rec, val_ in zip(recs, vals): rec.update(val_)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()

        if idx % stats_interval == 0 and stage >= 1:
            with torch.no_grad():
                r_now, _ = pseudo_reliability(vertex_pred.detach(), to_kp_px(gt_kps),
                                              vertex_weights, train_cfg['reliability_sigma_px'],
                                              train_cfg['reliability_tau1_px'],
                                              train_cfg['reliability_tau2_px'])
                stats_list.append(reliability_label_stats(r_now, vertex_weights))

        if idx % train_cfg['loss_rec_step'] == 0:
            step = epoch * size + idx
            losses_batch = OrderedDict()
            for name, rec in zip(recs_names, recs):
                losses_batch['train/' + name] = rec.avg
            recorder.rec_loss_batch(losses_batch, step, epoch)
            for rec in recs: rec.reset()

        if idx % train_cfg['img_rec_step'] == 0:
            batch_size = image.shape[0]
            nrow = 5 if batch_size > 5 else batch_size
            recorder.rec_segmentation(F.softmax(seg_pred, dim=1), num_classes=2, nrow=nrow,
                                      step=step, name='train/image/seg')
            recorder.rec_vertex(vertex_pred, vertex_weights, nrow=4, step=step, name='train/image/ver')

    save_label_stats(epoch, stage, stats_list, train_cfg)
    print('epoch {} training cost {} s'.format(epoch, time.time() - train_begin), flush=True)


def val(net, dataloader, epoch, train_cfg, recorder, val_prefix='val', force_eval=False):
    seg_loss_rec, ver_loss_rec, rel_loss_rec = AverageMeter(), AverageMeter(), AverageMeter()
    precision_rec, recall_rec = AverageMeter(), AverageMeter()
    recs = [seg_loss_rec, ver_loss_rec, rel_loss_rec, precision_rec, recall_rec]
    recs_names = ['scalar/seg', 'scalar/ver', 'scalar/rel', 'scalar/precision', 'scalar/recall']
    for rec in recs: rec.reset()

    test_begin = time.time()
    evaluator = Evaluator()
    use_rel = train_cfg['use_reliability_vote']
    if recorder is None:
        recorder = Recorder(False, os.path.join(EXP_DIR, 'record'),
                            os.path.join(EXP_DIR, 'record', 'test_{}.log'.format(val_prefix)))

    if use_rel:
        eval_net = DataParallel(EvalWrapperRel(train_cfg).cuda())
    else:
        eval_net = DataParallel(EvalWrapperBaseline(train_cfg).cuda())
    net.eval()
    do_eval = force_eval or (train_cfg['eval_epoch'] and epoch % train_cfg['eval_inter'] == 0
                             and epoch >= train_cfg['eval_epoch_begin'])
    for idx, data in enumerate(dataloader):
        image, mask, vertex, vertex_weights, pose, gt_kps = [d.cuda() for d in data]
        with torch.no_grad():
            seg_pred, vertex_pred, rel_logits, loss_seg, loss_vertex, loss_rel, precision, recall = \
                net(image, mask, vertex, vertex_weights, gt_kps)
            loss_seg, loss_vertex, loss_rel, precision, recall = \
                [torch.mean(v) for v in (loss_seg, loss_vertex, loss_rel, precision, recall)]

            if do_eval:
                if use_rel:
                    corner_pred = eval_net(seg_pred, vertex_pred, rel_logits).cpu().detach().numpy()
                else:
                    corner_pred = eval_net(seg_pred, vertex_pred).cpu().detach().numpy()
                pose_np = pose.cpu().numpy()
                b = pose_np.shape[0]
                for bi in range(b):
                    evaluator.evaluate(corner_pred[bi], pose_np[bi], train_cfg['linemod_cls'],
                                       'linemod', VOTE_TYPE, intri_matrix=None)

            vals = [loss_seg, loss_vertex, loss_rel, precision, recall]
            for rec, v in zip(recs, vals): rec.update(v)

    losses_batch = OrderedDict()
    for name, rec in zip(recs_names, recs):
        losses_batch['{}/'.format(val_prefix) + name] = rec.avg
    if do_eval:
        proj_err, add, cm = evaluator.average_precision(False)
        losses_batch['{}/scalar/projection_error'.format(val_prefix)] = proj_err
        losses_batch['{}/scalar/add'.format(val_prefix)] = add
        losses_batch['{}/scalar/cm'.format(val_prefix)] = cm
        # persist machine-readable result
        with open(os.path.join(EXP_DIR, 'result_{}_e{}.txt'.format(val_prefix, epoch)), 'w') as f:
            f.write('projection_error\t{}\nadd\t{}\ncm_degree_5\t{}\n'.format(proj_err, add, cm))
        with open(os.path.join(EXP_DIR, 'result_summary.txt'), 'a') as f:
            f.write('{}\tepoch {}\tproj_err {:.4f}\tadd {:.4f}\t5cm5deg {:.4f}\n'.format(
                val_prefix, epoch, proj_err, add, cm))
    recorder.rec_loss_batch(losses_batch, epoch, epoch, val_prefix)
    print('epoch {} {} cost {} s'.format(epoch, val_prefix, time.time() - test_begin), flush=True)


def build_val_loader(db_set, data_prefix, train_cfg, augment_cfg):
    val_set = LineModDatasetRealAug(db_set, data_prefix, VOTE_TYPE, augment=False, cfg=augment_cfg)
    val_loader = DataLoader(val_set,
                            batch_sampler=ImageSizeBatchSampler(SequentialSampler(val_set),
                                                                train_cfg['test_batch_size'], False,
                                                                cfg=augment_cfg),
                            num_workers=12)
    return val_loader


def load_net_weights(net, model_dir, epoch=-1):
    """load only net weights from a save_model dir (no optimizer state)."""
    if not os.path.exists(model_dir):
        return 0
    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir) if pth.endswith('.pth')]
    if len(pths) == 0:
        return 0
    ep = max(pths) if epoch == -1 else epoch
    ckpt = torch.load(os.path.join(model_dir, '{}.pth'.format(ep)), map_location='cpu')
    net.load_state_dict(ckpt['net'])
    return ep


def resolve_test_model_dir(train_cfg):
    """test_model: prefer stage2_joint, then stage1_rel_warmup, then the single-stage dir."""
    candidates = [os.path.join(EXP_DIR, 'model', 'stage2_joint'),
                  os.path.join(EXP_DIR, 'model', 'stage1_rel_warmup'),
                  os.path.join(EXP_DIR, 'model', train_cfg['model_name'])]
    for d in candidates:
        if os.path.isdir(d) and any(f.endswith('.pth') for f in os.listdir(d)):
            return d
    return candidates[-1]


def train_net(args, train_cfg):
    use_rel = train_cfg['use_reliability_vote']
    if use_rel:
        net = Resnet18_8sReliability(ver_dim=VOTE_NUM * 2, seg_dim=2)
    else:
        net = Resnet18_8s(ver_dim=VOTE_NUM * 2, seg_dim=2)
    net = NetWrapperRel(net, use_rel, train_cfg['reliability_sigma_px'],
                        train_cfg.get('reliability_tau1_px', 0.0),
                        train_cfg.get('reliability_tau2_px', 0.0))
    net = DataParallel(net).cuda()

    optimizer = optim.Adam(net.parameters(), lr=train_cfg['lr'])

    if args.test_model:
        model_dir = resolve_test_model_dir(train_cfg)
        print('testing with checkpoints from: {}'.format(model_dir), flush=True)
        torch.manual_seed(0)
        begin_epoch = load_model(net.module.net, optimizer, model_dir, args.load_epoch)

        # normal linemod: test_real + val_real (same protocol as baseline `--normal`)
        image_db = LineModImageDB(args.linemod_cls, has_render_set=False, has_fuse_set=False)
        test_db = image_db.test_real_set + image_db.val_real_set
        test_sets = [('test' if args.use_test_set else 'val', test_db, cfg.LINEMOD)]
        if args.linemod_cls in cfg.occ_linemod_cls_names:
            occ_image_db = OcclusionLineModImageDB(args.linemod_cls)
            test_sets.append(('occ_test' if args.use_test_set else 'occ_val',
                              occ_image_db.test_real_set, cfg.OCCLUSION_LINEMOD))

        for prefix, db, data_prefix in test_sets:
            print('testing {} ...'.format(prefix), flush=True)
            loader = build_val_loader(db, data_prefix, train_cfg, train_cfg['aug_cfg'])
            val(net, loader, begin_epoch, train_cfg, None, prefix, force_eval=True)
        return

    image_db = LineModImageDB(args.linemod_cls, has_fuse_set=train_cfg['use_fuse'],
                              has_render_set=True)
    train_db = list(image_db.render_set)
    if train_cfg['use_real_train']:
        train_db += image_db.train_real_set
    if train_cfg['use_fuse']:
        train_db += image_db.fuse_set

    train_set = LineModDatasetRealAug(train_db, cfg.LINEMOD, VOTE_TYPE, augment=True,
                                      cfg=train_cfg['aug_cfg'])
    train_loader = DataLoader(train_set,
                              batch_sampler=ImageSizeBatchSampler(RandomSampler(train_set),
                                                                  train_cfg['train_batch_size'], False,
                                                                  cfg=train_cfg['aug_cfg']),
                              num_workers=12)

    val_loader = build_val_loader(image_db.val_real_set, cfg.LINEMOD, train_cfg,
                                  train_cfg['aug_cfg'])
    occ_val_loader = None
    if args.linemod_cls in cfg.occ_linemod_cls_names:
        occ_image_db = OcclusionLineModImageDB(args.linemod_cls)
        occ_val_db = occ_image_db.test_real_set[:len(occ_image_db.test_real_set) // 2]
        occ_val_loader = build_val_loader(occ_val_db, cfg.OCCLUSION_LINEMOD, train_cfg,
                                          train_cfg['aug_cfg'])

    recorder = Recorder(True, os.path.join(EXP_DIR, 'record', train_cfg['model_name']),
                        os.path.join(EXP_DIR, 'record', train_cfg['model_name'] + '.log'))

    stage1_dir = os.path.join(EXP_DIR, 'model', 'stage1_rel_warmup')
    stage2_dir = os.path.join(EXP_DIR, 'model', 'stage2_joint')
    single_dir = os.path.join(EXP_DIR, 'model', train_cfg['model_name'])
    stage1_epochs = int(train_cfg.get('stage1_epochs', 0)) if use_rel else 0

    if use_rel and stage1_epochs > 0:
        # ================= Stage 1: baseline init, train rel_head only =================
        set_stage1_mode(net, True)
        optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad],
                               lr=train_cfg['lr'])
        begin_epoch = 0
        if train_cfg['resume'] and os.path.isdir(stage1_dir):
            begin_epoch = load_model(net.module.net, optimizer, stage1_dir)
        if begin_epoch == 0 and train_cfg.get('baseline_checkpoint'):
            ckpt_path = train_cfg['baseline_checkpoint']
            ckpt = torch.load(ckpt_path, map_location='cpu')
            net.module.net.load_state_dict(ckpt['net'], strict=False)   # rel_head stays fresh
            print('stage1: loaded baseline checkpoint {} (rel_head freshly initialized)'
                  .format(ckpt_path), flush=True)

        for epoch in range(begin_epoch, stage1_epochs):
            adjust_learning_rate(optimizer, epoch, train_cfg['lr_decay_rate'],
                                 train_cfg['lr_decay_epoch'])
            train(net, optimizer, train_loader, epoch, train_cfg, recorder, stage=1)
            val(net, val_loader, epoch, train_cfg, recorder)
            if occ_val_loader is not None:
                val(net, occ_val_loader, epoch, train_cfg, recorder, 'occ_val')
            save_model(net.module.net, optimizer, epoch, stage1_dir)

        # ================= Stage 2: joint fine-tuning =================
        set_stage1_mode(net, False)
        optimizer = optim.Adam(net.parameters(), lr=train_cfg['lr'])
        begin_epoch = stage1_epochs
        if train_cfg['resume'] and os.path.isdir(stage2_dir) \
                and any(f.endswith('.pth') for f in os.listdir(stage2_dir)):
            begin_epoch = load_model(net.module.net, optimizer, stage2_dir)
        else:
            if load_net_weights(net.module.net, stage1_dir) == 0:
                print('WARNING: no stage1 checkpoint found, starting stage2 from current weights',
                      flush=True)
            else:
                print('stage2: initialized from stage1_rel_warmup', flush=True)

        for epoch in range(begin_epoch, train_cfg['epoch_num']):
            adjust_learning_rate(optimizer, epoch, train_cfg['lr_decay_rate'],
                                 train_cfg['lr_decay_epoch'])
            train(net, optimizer, train_loader, epoch, train_cfg, recorder, stage=2)
            val(net, val_loader, epoch, train_cfg, recorder)
            if occ_val_loader is not None:
                val(net, occ_val_loader, epoch, train_cfg, recorder, 'occ_val')
            save_model(net.module.net, optimizer, epoch, stage2_dir)
    else:
        # single-stage path (also the exact-baseline switch when use_reliability_vote=false)
        begin_epoch = 0
        if train_cfg['resume']:
            begin_epoch = load_model(net.module.net, optimizer, single_dir)

        for epoch in range(begin_epoch, train_cfg['epoch_num']):
            adjust_learning_rate(optimizer, epoch, train_cfg['lr_decay_rate'],
                                 train_cfg['lr_decay_epoch'])
            train(net, optimizer, train_loader, epoch, train_cfg, recorder,
                  stage=1 if use_rel else 2)
            val(net, val_loader, epoch, train_cfg, recorder)
            if occ_val_loader is not None:
                val(net, occ_val_loader, epoch, train_cfg, recorder, 'occ_val')
            save_model(net.module.net, optimizer, epoch, single_dir)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='exp001 reliability-aware voting')
    parser.add_argument('--cfg_file', type=str, default='configs/exp001_reliability_voting.json')
    parser.add_argument('--linemod_cls', type=str, default='cat')
    parser.add_argument('--test_model', action='store_true')
    parser.add_argument('--use_test_set', action='store_true')
    parser.add_argument('--load_epoch', type=int, default=-1)
    args = parser.parse_args()

    with open(args.cfg_file, 'r') as f:
        train_cfg = json.load(f)
    train_cfg['model_name'] = '{}_{}'.format(args.linemod_cls, train_cfg['model_name'])
    train_cfg['linemod_cls'] = args.linemod_cls
    assert train_cfg['vote_type'] == 'Farthest', 'exp001 assumes Farthest voting'
    return args, train_cfg


def prepare_exp_dir(args, train_cfg):
    os.makedirs(os.path.join(EXP_DIR, 'model'), exist_ok=True)
    os.makedirs(os.path.join(EXP_DIR, 'record'), exist_ok=True)
    os.makedirs(os.path.join(EXP_DIR, 'results'), exist_ok=True)
    with open(os.path.join(EXP_DIR, 'config.json'), 'w') as f:
        json.dump({'train_cfg': train_cfg, 'args': vars(args),
                   'resolved': {'use_reliability_vote': train_cfg['use_reliability_vote'],
                                'vote_num': VOTE_NUM, 'exp_dir': EXP_DIR}}, f, indent=2)


if __name__ == '__main__':
    args, train_cfg = parse_args_and_config()
    prepare_exp_dir(args, train_cfg)
    train_net(args, train_cfg)
