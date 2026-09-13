"""EXP003: inference-only consensus-based vote reliability (weighted voting).

Design constraints:
- Reuses baseline CUDA ops (generate_hypothesis / voting_for_hypothesis) unchanged.
- Hypothesis sampling, inlier threshold, hypothesis count, stopping rule and
  refinement structure are identical to lib ransac_voting_layer_v2 (one shared
  round loop over all 9 keypoints, min-ratio confidence stopping).
- The ONLY variable is the vote contribution weight in scoring and refinement.
"""
import torch
import lib.ransac_voting_gpu_layer.ransac_voting as ransac_voting


def get_voting_problem(mask, vertex, class_num, round_hyp_num, max_num=100, min_num=5):
    """Replicates the per-keypoint sampling sequence of ransac_voting_layer_v2
    under the current CUDA RNG state (uniform_ downsample draw then random_ idxs).
    Returns dict: coords [tn,2] (x,y), direct [tn,vn,2] raw, idxs [hn,vn,2], tn;
    or None if the keypoint has too few fg pixels."""
    h, w, vn = vertex.shape[1], vertex.shape[2], vertex.shape[3]
    k = class_num - 1  # single foreground class (cat)
    cur_mask = mask[0] == k
    foreground = torch.sum(cur_mask)
    if foreground < min_num:
        return None
    if foreground > max_num:
        selection = torch.zeros(cur_mask.shape, dtype=torch.float32,
                                device=mask.device).uniform_(0, 1)
        selected_mask = (selection < (max_num / foreground.float()))
        cur_mask = cur_mask * selected_mask
    coords = torch.nonzero(cur_mask).float()
    coords = coords[:, [1, 0]]
    direct = vertex[0].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3))
    direct = direct.view([coords.shape[0], vn, 2])
    tn = coords.shape[0]
    idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32,
                       device=mask.device).random_(0, tn)
    return dict(coords=coords, direct=direct, idxs=idxs, tn=tn)


def consensus_reliability(coords, direct_k, coarse_kp, tau=None, sigma=None,
                          mode='hard', weight_clip=1e-3):
    """Per-pixel reliability from coarse-consensus geometry.

    d_i = |(C - p_i) x v_i| with v_i the NORMALIZED vote direction (perpendicular
    point-to-line distance, NOT Euclidean distance).
    hard: r = 1 if d < tau else 0.   soft: r = max(exp(-d^2/(2 sigma^2)), clip).
    Returns (r [tn] float, d [tn] float)."""
    v = direct_k / (torch.norm(direct_k, dim=1, keepdim=True) + 1e-9)
    diff = coarse_kp.unsqueeze(0) - coords                       # [tn,2]
    d = torch.abs(diff[:, 0] * v[:, 1] - diff[:, 1] * v[:, 0])   # |cross| = perp dist
    if mode == 'hard':
        r = (d < tau).float()
    else:
        r = torch.exp(-d * d / (2.0 * sigma * sigma)).clamp(min=weight_clip)
    return r, d


def weighted_ransac_all(problem, weights_all, coarse, ransac_cfg, min_votes=32):
    """Round-2 weighted voting, all keypoints jointly. Mirrors
    ransac_voting_layer_v2: same hypothesis set (given identical problem/RNG),
    same inlier threshold, same min-ratio confidence stopping over valid
    keypoints, same refinement structure. Weights only change scoring and the
    (weighted) LS refinement.

    :param problem: dict from get_voting_problem (direct [tn,vn,2], idxs [hn,vn,2])
    :param weights_all: [vn,tn] reliability weights per keypoint
    :param coarse: [vn,2] pass-1 (baseline) keypoints, fallback target
    :return: finals [vn,2], per-kp stats list
    """
    coords, direct, idxs = problem['coords'], problem['direct'], problem['idxs']
    tn, vn = problem['tn'], direct.shape[1]
    w = weights_all.float()                                       # [vn,tn]
    w_sum = w.sum(1)                                              # [vn]
    valid = torch.isfinite(coarse).all(1) & (w_sum > 0)
    reliable_cnt = (w > 0).float().sum(1)                         # [vn]
    valid = valid & (reliable_cnt >= min_votes)                   # hard-filter guard

    stats = [dict(num_votes=int(tn), num_reliable=int(reliable_cnt[k].item()),
                  mean_d=-1.0, median_d=-1.0, fallback=None) for k in range(vn)]
    finals = coarse.clone()
    for k in range(vn):
        if not bool(valid[k].item()):
            stats[k]['fallback'] = ('no_fg' if problem is None else
                                    'coarse_nan' if not torch.isfinite(coarse[k]).all()
                                    else 'few_reliable_votes')
            valid[k] = False

    hyp_num, cur_iter = 0, 0
    all_win_score = torch.zeros([vn], dtype=torch.float32, device=coords.device)
    all_win_pts = torch.zeros([vn, 2], dtype=torch.float32, device=coords.device)
    while True:
        cur_hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]
        cur_inlier = torch.zeros([idxs.shape[0], vn, tn], dtype=torch.uint8,
                                 device=coords.device)
        ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts,
                                            cur_inlier, ransac_cfg['inlier_thresh'])
        cur_scores = (cur_inlier.float() * w.unsqueeze(0)).sum(2)  # [hn,vn] weighted count
        cur_win_score, cur_win_idx = torch.max(cur_scores, 0)      # [vn]
        cur_win_pts = cur_hyp_pts[cur_win_idx, torch.arange(vn)]
        larger_mask = all_win_score < cur_win_score
        all_win_pts[larger_mask, :] = cur_win_pts[larger_mask, :]
        all_win_score[larger_mask] = cur_win_score[larger_mask]
        hyp_num += idxs.shape[0]
        cur_iter += 1
        if bool(valid.any().item()):
            cur_min_ratio = float((all_win_score[valid] / w_sum[valid]).min().item())
        else:
            cur_min_ratio = 1.0
        if (1 - (1 - cur_min_ratio ** 2) ** hyp_num) > ransac_cfg['confidence'] \
                or cur_iter > ransac_cfg['max_iter']:
            break

    # refinement: same structure as baseline (re-vote best points, LS intersect)
    normal = torch.zeros_like(direct)
    normal[:, :, 0] = direct[:, :, 1]
    normal[:, :, 1] = -direct[:, :, 0]
    all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=coords.device)
    ransac_voting.voting_for_hypothesis(direct, coords, all_win_pts.unsqueeze(0),
                                        all_inlier, ransac_cfg['inlier_thresh'])
    for k in range(vn):
        if not bool(valid[k].item()):
            continue
        sel = all_inlier[0, k].bool()
        if sel.sum() == 0:
            stats[k]['fallback'] = 'refine_no_inlier'
            continue
        cur_coords = coords[sel]                                  # [cn,2]
        cur_normal = normal[:, k, :][sel]                         # [cn,2]
        w_sel = w[k, sel]                                         # [cn]
        if float(w_sel.sum().item()) <= 0:
            stats[k]['fallback'] = 'refine_zero_weight'
            continue
        b = torch.sum(cur_normal * cur_coords, 1)
        if bool((w_sel == w_sel[0]).all()):
            A = cur_normal                                        # w==1 -> baseline formula
        else:
            sw = torch.sqrt(w_sel).unsqueeze(1)
            A = sw * cur_normal
            b = sw[:, 0] * b
        refine_pt = torch.matmul(torch.pinverse(A), b)
        if not torch.isfinite(refine_pt).all():
            stats[k]['fallback'] = 'refine_nan'
            continue
        finals[k] = refine_pt
        v = direct[:, k, :] / (torch.norm(direct[:, k, :], dim=1, keepdim=True) + 1e-9)
        diff = coarse[k].unsqueeze(0) - coords
        d_all = torch.abs(diff[:, 0] * v[:, 1] - diff[:, 1] * v[:, 0])
        stats[k]['mean_d'] = float(d_all.mean().item())
        stats[k]['median_d'] = float(d_all.median().item())
    return finals, stats
