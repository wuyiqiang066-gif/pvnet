"""EXP007 model: Resnet18_8s + minimal local-context module (NEW file only).

Design (per EXP007 spec):
  shared feature map fm (32 ch, raw resolution)
      context_feature = PW(DW(fm))        DW: 3x3 depthwise, PW: 1x1 pointwise
      f_context       = fm + context_feature
      seg_pred        = convraw(cat[fm, img])[:, :2]        <- bit-exact baseline seg path
      ver_pred        = vertex_head(cat[f_context, img])[:, 2:]
  vertex_head is a weight-copy of the baseline shared head (convraw), so at
  initialization (and whenever context ~= 0) the whole network is exactly the
  baseline function. Segmentation head/loss untouched; vertex loss untouched;
  output shapes unchanged: seg [B,2,H,W], vertex [B,18,H,W] (unit direction GT).
"""
import copy
import torch
from torch import nn

from lib.networks.model_repository import Resnet18_8s


class Resnet18_8sContext(nn.Module):
    def __init__(self, ver_dim=18, seg_dim=2, fcdim=256, s8dim=128, s4dim=64,
                 s2dim=32, raw_dim=32):
        super(Resnet18_8sContext, self).__init__()
        self.ver_dim = ver_dim
        self.seg_dim = seg_dim
        self.base = Resnet18_8s(ver_dim=ver_dim, seg_dim=seg_dim, fcdim=fcdim,
                                s8dim=s8dim, s4dim=s4dim, s2dim=s2dim, raw_dim=raw_dim)
        C = s2dim  # channels of the shared feature map fm
        # context module: DW 3x3 + PW 1x1, residual fusion (channels unchanged)
        self.context_dw = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
        self.context_pw = nn.Conv2d(C, C, 1, bias=True)
        # vertex-only head: weight copy of the baseline shared head (convraw)
        self.vertex_head = copy.deepcopy(self.base.convraw)
        self.init_context()

    def init_context(self):
        """Repo convention for new layers (Resnet18_8s._normal_initialization):
        weights ~ N(0, 0.01), bias = 0 -> context residual starts ~0 so the
        initial function is baseline-equivalent up to O(0.01) perturbation."""
        for m in (self.context_dw, self.context_pw):
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        x2s, x4s, x8s, x16s, x32s, xfc = self.base.resnet18_8s(x)
        fm = self.base.conv8s(torch.cat([xfc, x8s], 1))
        fm = self.base.up8sto4s(fm)
        fm = self.base.conv4s(torch.cat([fm, x4s], 1))
        fm = self.base.up4sto2s(fm)
        fm = self.base.conv2s(torch.cat([fm, x2s], 1))
        fm = self.base.up2storaw(fm)

        # segmentation path: bit-exact baseline (original feature, original head)
        seg_all = self.base.convraw(torch.cat([fm, x], 1))
        seg_pred = seg_all[:, :self.seg_dim]

        # vertex path: context-enhanced feature -> copied vertex head
        f_context = fm + self.context_pw(self.context_dw(fm))
        ver_all = self.vertex_head(torch.cat([f_context, x], 1))
        ver_pred = ver_all[:, self.seg_dim:]
        return seg_pred, ver_pred


def load_from_baseline(model, baseline_net_sd):
    """Load a baseline Resnet18_8s state dict into Resnet18_8sContext.
    baseline params -> base.* ; convraw weights -> vertex_head.* (copy).
    Context module keeps its random init. Returns missing/unexpected keys."""
    new_sd = {}
    for k, v in baseline_net_sd.items():
        new_sd['base.' + k] = v
        if k.startswith('convraw.'):
            new_sd['vertex_head.' + k[len('convraw.'):]] = v.clone()
    ret = model.load_state_dict(new_sd, strict=False)
    ctx_keys = {'context_dw.weight', 'context_pw.weight', 'context_pw.bias'}
    assert set(ret.missing_keys) == ctx_keys, ret.missing_keys
    assert not ret.unexpected_keys, ret.unexpected_keys
    return ret


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    base = sum(p.numel() for p in model.base.parameters())
    ctx = sum(p.numel() for p in model.context_dw.parameters()) + \
          sum(p.numel() for p in model.context_pw.parameters())
    vhead = sum(p.numel() for p in model.vertex_head.parameters())
    return dict(total=total, baseline_params=base, context_module=ctx,
                vertex_head_copy=vhead, newly_added=ctx + vhead)
