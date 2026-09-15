"""EXP018: Pose Hypothesis Oracle Diagnosis.

Research question (EXP017 follow-up): EXP017 proved (a) GT keypoints recover
the EXACT GT pose through the ORIGINAL PnP (0/20 flips, rot 0.000 deg), so no
intrinsic ambiguity exists near zero error; (b) D1's flip burst (14/20 vs D0
4/20) is NOT explained by keypoint error magnitude but is ASSOCIATED with the
keypoint residual geometry. The remaining question:

    CASE A: the correct pose hypothesis EXISTS and the original PnP's
            hypothesis SELECTION failed (solver picked the wrong basin), or
    CASE B: D1's keypoint geometry has already EXCLUDED the correct pose from
            the solver's candidate space (hypothesis GENERATION failure).

Code check (performed BEFORE writing any experiment code, verbatim facts):
  - The baseline path is lib/utils/evaluation_utils.py::pnp ->
    cv2.solvePnP(flags=cv2.SOLVEPNP_ITERATIVE): ONE deterministic solution.
    No candidate list, no intermediate iterates, no inlier masks, no random
    initializations, no RANSAC (solvePnPRansac appears only in a comment).
  - No solvePnPGeneric anywhere in the repo (and it is BANNED by this
    protocol as a "replacement method").
  - The only other solvePnP sites are uncertainty_pnp / uncertainty_pnp_v2
    (extend_utils) -- a DIFFERENT solver used by evaluate_uncertainty, NOT
    part of the baseline path; using it would violate "no new solver".
  - ransac_voting_layer_v3 performs RANSAC over VOTE LINES for KEYPOINTS,
    not object poses; it exposes no pose candidates.
  => Candidate hypotheses are NOT exposed by the current implementation, and
     instrumenting OpenCV's internal DLT init / Newton iterates would require
     reimplementing the solver (forbidden) or changing flags/method
     (forbidden). Therefore, per protocol section 2/7:

        candidate hypotheses are not exposed by the current implementation,
        therefore EXP018 can only provide partial evidence.

Partial evidence provided (ZERO solver modification, existing artifacts only):
  Objective-level oracle. For each image and condition, the fit of a pose to
  the INPUT keypoints is measured with the SAME original Projector.project_K
  that Evaluator.projection_2d uses in every evaluate() call, on the SAME 9
  VotingType.Farthest 3D points the solver consumed, under the SAME 'linemod'
  intrinsics:
      J_pred = mean_i || project_K(kp3d, pose_pred)_i - kp_input_i ||   (px)
      J_gt   = mean_i || project_K(kp3d, pose_gt )_i - kp_input_i ||   (px)
  {pose_pred, pose_gt} are the only poses observable without a new solver;
  pose_gt has rot error 0 by construction, so a correct hypothesis (rot<10
  deg) EXISTS among observable poses on every image -- the informative
  quantity is the objective's RANKING:
      J_gt < J_pred - eps : correct pose fits the input keypoints STRICTLY
                            better than the returned pose -> the solver left
                            the better-fitting correct hypothesis unfound
                            (SELECTION FAILURE, CASE A evidence)
      J_pred < J_gt - eps : the ~180 deg pose fits the input keypoints
                            STRICTLY better than the correct pose -> the
                            reprojection objective itself prefers the wrong
                            basin (IMAGE-SPACE REPROJECTION AMBIGUITY, Type D;
                            simultaneously the strongest available evidence
                            for the CASE B geometric reading, since a perfect
                            objective-based selector would also choose wrong)
      |J_gt - J_pred| <= eps: UNRESOLVED (near-tie)
  NO_CORRECT_HYPOTHESIS (Type C) is NOT assignable without the candidate
  space -> recorded as 0 with an explicit caveat.

Conditions (C0/C1/C2 re-verified through the ORIGINAL Evaluator.evaluate;
must reproduce EXP017 exactly):
  C0 GT keypoints, C1 EXP016/017 D0 voted keypoints, C2 EXP016/017 D1 voted
  keypoints. For C1/C2 the fresh pose must additionally match EXP016's stored
  pred_pose bit-exactly (inputs are bit-identical). Any mismatch -> STOP.

Additional required validation: EXP017's keypoint residual geometry findings
(D1 flip images: lower anisotropy, smaller coherent shift, magnitude not
explanatory) are re-derived read-only from EXP017's residual_geometry.csv.

Cost: CPU-only, no network forward, no GPU, ~2 s.
Pre-registered decision rules are written to config.json BEFORE any PnP call.
Run from repo root:
  python experiments/exp018_pose_hypothesis_oracle/scripts/run_exp018.py
  python experiments/exp018_pose_hypothesis_oracle/scripts/analyze_exp018.py
"""
import os, sys, json, csv, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)

import numpy as np

from lib.utils.evaluation_utils import Evaluator          # original, unmodified
from lib.datasets.linemod_dataset import VotingType

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../exp018_pose_hypothesis_oracle
RES_DIR = os.path.join(EXP_DIR, 'results')
os.makedirs(RES_DIR, exist_ok=True)
EXP17_DIR = os.path.join(ROOT, 'experiments', 'exp017_pose_hypothesis_landscape')
EXP16_DIR = os.path.join(ROOT, 'experiments', 'exp016_pnp_stability_diagnosis')

N_IMG, KP = 20, 9
FLIP_THRESH = (90.0, 120.0, 150.0, 170.0)   # EXP016/017 definition, not re-invented
VOTE_TYPE = VotingType.Farthest
COND_NAMES = ('C0_gt', 'C1_D0', 'C2_D1')
TIE_EPS = 1e-3                              # px, pre-registered near-tie band on J_gt - J_pred
AMBIG_ROT = 150.0                           # 'wrong hypothesis rotation ~ 180 deg' qualifier

T0 = time.time()
LOG = open(os.path.join(EXP_DIR, 'run.log'), 'w', buffering=1)


def log(msg):
    print(msg, flush=True)
    LOG.write(msg + '\n')


# ---------------- image IDs: assert == EXP017 ----------------
ids17 = json.load(open(os.path.join(EXP17_DIR, 'results', 'image_ids.json')))
OCC_RGB = list(ids17['occ_rgb'])
assert len(OCC_RGB) == N_IMG, 'EXP017 occ id count != 20 -> STOP'
json.dump(dict(source='EXP017 results/image_ids.json (asserted identical; '
                    'EXP017 asserted == EXP015 == EXP016 usage)',
               image_ids=list(range(N_IMG, 2 * N_IMG)), occ_rgb=OCC_RGB),
          open(os.path.join(EXP_DIR, 'image_ids.json'), 'w'), indent=2)
log('image IDs asserted == EXP017: '
    f'{[os.path.splitext(r)[0] for r in OCC_RGB]}')

# ---------------- bit-identical inputs from EXP016 npz (no re-sampling) ----------------
z0 = np.load(os.path.join(EXP16_DIR, 'results', 'raw_predictions', 'raw_occ_D0.npz'))
z1 = np.load(os.path.join(EXP16_DIR, 'results', 'raw_predictions', 'raw_occ_D1.npz'))
assert list(z0['image_ids']) == list(range(N_IMG, 2 * N_IMG)) == list(z1['image_ids'])
GT_KPS = z0['gt_kps'].astype(np.float64)
GT_POSE = z0['gt_pose'].astype(np.float64)
assert np.array_equal(GT_KPS, z1['gt_kps'].astype(np.float64))
assert np.array_equal(GT_POSE, z1['gt_pose'].astype(np.float64))
NPZ_POSE = {'C1_D0': z0['pred_pose'].astype(np.float64),
            'C2_D1': z1['pred_pose'].astype(np.float64)}
KP_IN = {'C0_gt': GT_KPS.copy(),
         'C1_D0': z0['pred_kps'].astype(np.float64),
         'C2_D1': z1['pred_kps'].astype(np.float64)}

# EXP017 stored reference (exact-match target)
REF17 = {c: [] for c in COND_NAMES}
with open(os.path.join(EXP17_DIR, 'results', 'per_image.csv')) as f:
    for row in csv.DictReader(f):
        if row['condition'] in REF17:
            REF17[row['condition']].append(row)
for c in COND_NAMES:
    assert len(REF17[c]) == N_IMG and [r['rgb'] for r in REF17[c]] == OCC_RGB, \
        f'EXP017 {c} rows/rgb order mismatch -> STOP'

# ---------------- pre-registered config (BEFORE any PnP call) ----------------
json.dump(dict(
    cls='cat', n_img=N_IMG,
    question='D1 ~180deg flips: correct hypothesis exists but selection failed (CASE A) '
             'vs correct pose excluded from candidate space by keypoint geometry (CASE B) '
             'vs reprojection objective ambiguity (CASE C)',
    data='EXP016 raw_occ_D{0,1}.npz bit-identical reuse; image IDs asserted == EXP017',
    pnp='original Evaluator.evaluate -> pnp (cv2.SOLVEPNP_ITERATIVE); UNMODIFIED',
    candidate_accessibility='cv2.solvePnP SOLVEPNP_ITERATIVE exposes ONE deterministic '
                            'solution; no candidate list/iterates/inliers accessible from '
                            'Python; instrumenting OpenCV internals requires reimplementing '
                            'the solver (forbidden) or changing flags (forbidden); '
                            'solvePnPGeneric banned; uncertainty_pnp is a different solver '
                            '(banned as new solver); voting RANSAC yields no pose candidates '
                            '=> PARTIAL-EVIDENCE MODE per protocol section 2/7',
    observable_poses='pose_pred (solver output) and pose_gt (rot 0 by construction); '
                     'correct_hypothesis_exists (rot<10deg) is TRUE on every image among '
                     'observable poses -- vacuously; the informative quantity is the '
                     'objective ranking J_gt vs J_pred',
    objective_oracle=dict(
        definition='J(pose) = mean_i || project_K(kp3d_Farthest9, pose)_i - kp_input_i || (px), '
                   'original Projector.project_K, original linemod intrinsics, the SAME 9 3D '
                   'points the solver consumed',
        J_pred='fit of the solver-returned pose to the input keypoints (its converged '
               'objective value)',
        J_gt='fit of the correct (GT) pose to the input keypoints',
        delta='J_gt - J_pred; delta > +tie_eps -> SELECTION_FAILURE (correct fits strictly '
              'better, solver left it unfound); delta < -tie_eps -> '
              'REPROJECTION_AMBIGUITY (wrong pose fits strictly better; Type D; ~180deg '
              'qualifier: wrong rot > 150deg recorded); |delta| <= tie_eps -> UNRESOLVED',
        tie_eps_px=TIE_EPS, ambiguity_rot_gt_deg=AMBIG_ROT,
        NO_CORRECT_HYPOTHESIS='NOT assignable without the candidate space; recorded 0 with '
                              'caveat (candidate hypotheses are not exposed by the current '
                              'implementation, therefore EXP018 can only provide partial '
                              'evidence)'),
    decision_rules_pre_registered=dict(
        order='Check B reproduction must PASS first, else STOP (report mismatch, no '
              'interpretation)',
        rule='among the D1 flip(>90deg) images, decision by PLURALITY of outcomes '
             '{SELECTION_FAILURE, REPROJECTION_AMBIGUITY(wrong rot>150deg), UNRESOLVED}: '
             'GO-A if SELECTION_FAILURE plurality; GO-C if REPROJECTION_AMBIGUITY plurality; '
             'STOP if UNRESOLVED plurality or an exact tie between the top two; '
             'GO-B strict condition (correct_hypothesis_exists == False in the candidate '
             'space) is NOT verifiable without candidate access -> documented as '
             'limitation; mean delta over D1 flip images reported as descriptive support',
        forbidden='no solver/flag/threshold/seed/keypoint/GT modification; no follow-up '
                  'experiment designed here'),
    language_limits='cat-only, 20-image mechanism diagnosis; no significance claims'),
    open(os.path.join(EXP_DIR, 'config.json'), 'w'), indent=2)
log('pre-registered config.json written (BEFORE any PnP call)')

# ---------------- C0/C1/C2 through the ORIGINAL PnP + objective oracle ----------------
ev0 = Evaluator()
KP3D = VotingType.get_pts_3d(VOTE_TYPE, 'cat')               # the solver's own 3D points
K = ev0.projector.intrinsic_matrix['linemod']                # original intrinsics

# sanity: projector(GT pose) vs dataset gt_kps (informational)
proj_gt_check = float(np.max(np.linalg.norm(
    ev0.projector.project_K(KP3D, GT_POSE[0], K) - GT_KPS[0], axis=1)))
log(f'sanity: max |project_K(kp3d, GT pose) - dataset gt_kps| on img20: '
    f'{proj_gt_check:.2e} px (informational; oracle uses the projector for BOTH '
    f'J_pred and J_gt -> no bias in delta)')

rows = []
repro_ok = True
for cond in COND_NAMES:
    ev = Evaluator()                                          # fresh original evaluator
    model = ev.linemod_db.get_ply_model('cat')
    for i in range(N_IMG):
        kp = KP_IN[cond][i]
        pnp_ok = True
        try:
            pose = ev.evaluate(kp, GT_POSE[i], 'cat', 'linemod', VOTE_TYPE, intri_matrix=None)
        except Exception as e:
            pnp_ok = False
            log(f'PnP EXCEPTION [{cond} {OCC_RGB[i]}]: {e}')
        # oracle fits (original projector; pose_gt / pose_pred on the solver's 3D points)
        if pnp_ok:
            proj_pred = ev.projector.project_K(KP3D, pose, K)
            proj_gtp = ev.projector.project_K(KP3D, GT_POSE[i], K)
            j_pred = float(np.mean(np.linalg.norm(proj_pred - kp, axis=1)))
            j_gt = float(np.mean(np.linalg.norm(proj_gtp - kp, axis=1)))
            rot = float(np.rad2deg(np.arccos(np.clip(
                (np.trace(pose[:, :3] @ GT_POSE[i][:, :3].T) - 1) / 2, -1, 1))))
            trans_mm = float(np.linalg.norm(pose[:, 3] - GT_POSE[i][:, 3]) * 1000.0)
            add_mm = float(ev.add_dists[-1]) * 1000.0
            reproj = float(ev.proj_mean_diffs[-1])            # original official metric
        else:
            j_pred = j_gt = rot = trans_mm = add_mm = reproj = float('nan')
        # reproduction checks vs EXP017 (exact)
        ref = REF17[cond][i]
        def _rf(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return float('nan')
        ref_rot, ref_reproj = _rf(ref['rotation_error_deg']), _rf(ref['mean_reprojection_error_px'])
        ref_trans, ref_flip = _rf(ref['translation_error_mm']), int(ref['flip_gt_90'])
        my_flip = bool(pnp_ok and rot > 90.0)
        ok_i = ((int(my_flip) == ref_flip)
                and (np.isfinite(rot) == np.isfinite(ref_rot))
                and (not np.isfinite(rot) or abs(rot - ref_rot) < 1e-3)
                and (not np.isfinite(reproj) or abs(reproj - ref_reproj) < 1e-3)
                and (not np.isfinite(trans_mm) or abs(trans_mm - ref_trans) < 1e-2))
        repro_ok &= ok_i
        bit_exact = ''
        if cond in NPZ_POSE and pnp_ok:
            d = float(np.max(np.abs(pose - NPZ_POSE[cond][i])))
            bit_exact = f'{d:.1e}'
            assert d < 1e-9, f'{cond} {OCC_RGB[i]}: fresh pose != EXP016 stored pred_pose -> STOP'
        rows.append(dict(condition=cond, image_id=N_IMG + i, rgb=OCC_RGB[i],
                         rotation_error_deg=rot, translation_error_mm=trans_mm,
                         add_mm=add_mm, mean_reprojection_error_px=reproj,
                         **{f'flip_gt_{int(t)}': bool(pnp_ok and rot > t) for t in FLIP_THRESH},
                         J_pred_px=j_pred, J_gt_px=j_gt,
                         delta_px=(j_gt - j_pred) if pnp_ok else float('nan'),
                         repro_ok_vs_exp017=ok_i))
    log(f'--- {cond}: 20 images through original PnP + oracle fits ({time.time() - T0:.1f}s) ---')

log(f'REPRODUCTION vs EXP017 (C0/C1/C2 flips exact + rot/reproj/trans): ok={repro_ok}')
if not repro_ok:
    log('REPRODUCTION FAILED -> STOP (mismatch reported; oracle NOT interpreted)')

# pose bit-exactness vs EXP016 npz was asserted inside the loop (log summary)
log('C1/C2 fresh poses bit-match EXP016 stored pred_pose (< 1e-9): asserted')

with open(os.path.join(RES_DIR, 'objective_oracle.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['condition', 'image_id', 'rgb',
                 'rotation_error_deg', 'translation_error_mm', 'add_mm',
                 'mean_reprojection_error_px',
                 'flip_gt_90', 'flip_gt_120', 'flip_gt_150', 'flip_gt_170',
                 'J_pred_px', 'J_gt_px', 'delta_px', 'repro_ok_vs_exp017'])
    for r in rows:
        wr.writerow([r['condition'], r['image_id'], r['rgb'],
                     f"{r['rotation_error_deg']:.4f}", f"{r['translation_error_mm']:.2f}",
                     f"{r['add_mm']:.4f}", f"{r['mean_reprojection_error_px']:.4f}",
                     int(r['flip_gt_90']), int(r['flip_gt_120']),
                     int(r['flip_gt_150']), int(r['flip_gt_170']),
                     f"{r['J_pred_px']:.6f}", f"{r['J_gt_px']:.6f}",
                     f"{r['delta_px']:.6f}", int(r['repro_ok_vs_exp017'])])

json.dump(dict(reproduction_ok=bool(repro_ok), pose_bitmatch_npz=True,
               proj_gt_vs_dataset_gt_kps_max_px=proj_gt_check,
               note='flips exact + rot<1e-3deg + reproj<1e-3px + trans<1e-2mm vs EXP017 '
                    'per_image.csv; C1/C2 fresh poses bit-match EXP016 npz pred_pose'),
          open(os.path.join(RES_DIR, 'reproduction_check.json'), 'w'), indent=2)

for cond in COND_NAMES:
    rs = [r for r in rows if r['condition'] == cond]
    log(f"flips [{cond}]: " + '  '.join(f'>{int(t)}deg {sum(r[f"flip_gt_{int(t)}"] for r in rs)}/20'
                                        for t in FLIP_THRESH))
log(f'DONE in {time.time() - T0:.1f}s (CPU-only; original PnP unmodified)')
