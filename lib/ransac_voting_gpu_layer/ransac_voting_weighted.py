"""
exp001_reliability_voting: reliability-weighted vote aggregation.

New module (additive only). The baseline file `ransac_voting_gpu.py` and all of
its functions are untouched. This module reuses the same compiled CUDA kernels
(`generate_hypothesis`, `voting_for_hypothesis`) so the RANSAC framework
(hypothesis sampling, inlier threshold, confidence stopping) is identical to
`ransac_voting_layer_v3`. Only the aggregation changes:

    baseline : score(h,k) = sum_i inlier(i,h,k)                (equal weight)
    weighted : score(h,k) = sum_i r_i^k * inlier(i,h,k)        (reliability)

and the final least-squares refinement becomes weight-weighted (weights are
folded as sqrt(w) into the normals, which yields exactly the weighted normal
equations with the same code path).
"""
import torch
import lib.ransac_voting_gpu_layer.ransac_voting as ransac_voting
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import b_inv


def ransac_voting_layer_v3_weighted(mask, vertex, weights, round_hyp_num, inlier_thresh=0.999,
                                    confidence=0.99, max_iter=20, min_num=5, max_num=30000,
                                    min_weight=0.001):
    '''
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,vn,2]
    :param weights:   [b,h,w,vn] reliability in [0,1], one channel per keypoint
    :param min_weight: pixels with weight below this are excluded from scoring
                       and refinement; if ALL pixels of a keypoint are excluded,
                       that keypoint falls back to uniform weights (baseline).
    :return: [b,vn,2]
    '''
    b, h, w, vn, _ = vertex.shape
    batch_win_pts = []
    for bi in range(b):
        cur_mask = (mask[bi]).bool()
        foreground_num = torch.sum(cur_mask)

        # if too few points, just skip it
        if foreground_num < min_num:
            win_pts = torch.zeros([1, vn, 2], dtype=torch.float32, device=mask.device)
            batch_win_pts.append(win_pts)  # [1,vn,2]
            continue

        # if too many inliers, we randomly down sample it
        if foreground_num > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground_num.float()))
            cur_mask = cur_mask & selected_mask.bool()

        coords = torch.nonzero(cur_mask).float()  # [tn,2]
        coords = coords[:, [1, 0]]
        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3))  # [tn,vn,2]
        direct = direct.view([coords.shape[0], vn, 2])
        w_sel = weights[bi].masked_select(torch.unsqueeze(cur_mask, 2))      # [tn*vn]
        w_sel = w_sel.view([coords.shape[0], vn])
        # hard-cut unreliable pixels
        w_sel = w_sel * (w_sel >= min_weight).float()
        tn = coords.shape[0]

        # fallback: keypoints whose pixels are all unreliable vote with uniform weights
        w_fallback = (torch.sum(w_sel, 0) < 1e-6).float()       # [vn]
        w_eff = w_sel + torch.unsqueeze(w_fallback, 0)          # [tn,vn]
        w_total = torch.sum(w_eff, 0)                           # [vn]

        idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
        all_win_ratio = torch.zeros([vn], dtype=torch.float32, device=mask.device)
        all_win_pts = torch.zeros([vn, 2], dtype=torch.float32, device=mask.device)

        hyp_num = 0
        cur_iter = 0
        while True:
            # generate hypothesis (identical to baseline)
            cur_hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

            # voting for hypothesis (identical to baseline)
            cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh)

            # weighted aggregation: score(h,k)=sum_i r_i^k * inlier(i,h,k)
            cur_inlier_scores = torch.einsum('hvt,tv->hv', cur_inlier.float(), w_eff)  # [hn,vn]
            cur_win_scores, cur_win_idx = torch.max(cur_inlier_scores, 0)              # [vn]
            cur_win_pts = cur_hyp_pts[cur_win_idx, torch.arange(vn)]
            cur_win_ratio = cur_win_scores / torch.clamp(w_total, min=1e-6)            # normalize to [0,1]

            # update best point
            larger_mask = all_win_ratio < cur_win_ratio
            all_win_pts[larger_mask, :] = cur_win_pts[larger_mask, :]
            all_win_ratio[larger_mask] = cur_win_ratio[larger_mask]

            # check confidence (same criterion as baseline)
            hyp_num += round_hyp_num
            cur_iter += 1
            cur_min_ratio = torch.min(all_win_ratio)
            if (1 - (1 - cur_min_ratio ** 2) ** hyp_num) > confidence or cur_iter > max_iter:
                break

        # weighted least-squares refinement over the inliers of the winning hypothesis
        normal = torch.zeros_like(direct)   # [tn,vn,2]
        normal[:, :, 0] = direct[:, :, 1]
        normal[:, :, 1] = -direct[:, :, 0]
        all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
        all_win_pts = torch.unsqueeze(all_win_pts, 0)  # [1,vn,2]
        ransac_voting.voting_for_hypothesis(direct, coords, all_win_pts, all_inlier, inlier_thresh)

        all_inlier = torch.squeeze(all_inlier.float(), 0)           # [vn,tn]
        normal = normal.permute(1, 0, 2)                            # [vn,tn,2]
        w_in = all_inlier * w_eff.transpose(0, 1)                   # [vn,tn] (outlier weight is zero)
        normal = normal * torch.unsqueeze(torch.sqrt(torch.clamp(w_in, min=0.0)), 2)  # fold sqrt(w)

        b_vec = torch.sum(normal * torch.unsqueeze(coords, 0), 2)   # [vn,tn]
        ATA = torch.matmul(normal.permute(0, 2, 1), normal)         # [vn,2,2]
        ATb = torch.sum(normal * torch.unsqueeze(b_vec, 2), 1)      # [vn,2]
        # no ridge (keeps exact parity with ransac_voting_layer_v3 when weights
        # are uniform); only an exactly singular ATA falls back to pinv
        try:
            ATA_inv = b_inv(ATA)
        except RuntimeError:
            ATA_inv = torch.linalg.pinv(ATA)
        all_win_pts = torch.matmul(ATA_inv, torch.unsqueeze(ATb, 2))      # [vn,2,1]
        batch_win_pts.append(all_win_pts[None, :, :, 0])

    batch_win_pts = torch.cat(batch_win_pts)
    return batch_win_pts
