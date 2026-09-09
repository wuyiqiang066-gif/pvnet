"""
exp001_reliability_voting: reliability head network (additive only).

`Resnet18_8s` in `model_repository.py` is NOT modified. This subclass reuses the
untouched baseline decoder and adds a small per-keypoint reliability head on the
shared raw-resolution feature map (the same feature that feeds `convraw`).

Outputs: (seg_pred, vertex_pred, rel_logits)
    seg_pred    [B, seg_dim, H, W]        identical to baseline
    vertex_pred [B, ver_dim, H, W]        identical to baseline
    rel_logits  [B, ver_dim//2, H, W]     one reliability logit per pixel per keypoint

A baseline checkpoint loads directly into this class (all parent parameter names
are unchanged; only `rel_head` is newly initialized).
"""
import torch
from torch import nn
from lib.networks.model_repository import Resnet18_8s


class Resnet18_8sReliability(Resnet18_8s):
    def __init__(self, ver_dim, seg_dim, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32):
        super(Resnet18_8sReliability, self).__init__(ver_dim, seg_dim, fcdim, s8dim, s4dim, s2dim, raw_dim)
        self.rel_dim = ver_dim // 2
        self.rel_head = nn.Sequential(
            nn.Conv2d(s2dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(s2dim, self.rel_dim, 1, 1)
        )
        # initialize the scoring layer like the baseline scoring conv
        self.rel_head[-1].weight.data.normal_(0, 0.01)
        self.rel_head[-1].bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x)

        fm = self.conv8s(torch.cat([xfc, x8s], 1))
        fm = self.up8sto4s(fm)

        fm = self.conv4s(torch.cat([fm, x4s], 1))
        fm = self.up4sto2s(fm)

        fm = self.conv2s(torch.cat([fm, x2s], 1))
        fm = self.up2storaw(fm)

        x = self.convraw(torch.cat([fm, x], 1))
        seg_pred = x[:, :self.seg_dim, :, :]
        ver_pred = x[:, self.seg_dim:, :, :]
        rel_logits = self.rel_head(fm)

        return seg_pred, ver_pred, rel_logits
