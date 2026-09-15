"""EXP017: Pose Hypothesis Landscape Diagnosis.

Research question (EXP016 follow-up): EXP016 showed that the EXP013 D1
decoder's improved OCC keypoints (median 3.150 -> 2.143 px) push the ORIGINAL
standard PnP into ~180 deg flip basins far more often than D0 (OCC >90 deg
flips 4/20 vs 14/20), while keypoint error MAGNITUDE does not separate flip
from stable images. EXP017 answers ONE question with a controlled keypoint
input landscape:

    Is the D1 flip burst
      A. intrinsic ambiguity of the cat keypoint/PnP geometry itself, or
      B. a stable-GT regime where D1's specific keypoint residual geometry
         pushes the ORIGINAL PnP into pre-existing wrong hypothesis basins,
      C. perturbation-sensitive numerical instability of the original PnP,
      D. inconclusive on this 20-image diagnosis?

Pure diagnosis. NOTHING is modified: no training, no baseline/PVNet/vertex/
voting/PnP change, no solver replacement, no tuning, no sigma sweep, no extra
seeds/datasets. The ONLY manipulated variable is the 2D keypoint vector fed
to the UNMODIFIED original PnP.

Data reuse (bit-identical, no re-sampling): EXP016
results/raw_predictions/raw_occ_D{0,1}.npz provide the EXACT voted keypoints
(pred_kps), GT projected keypoints (gt_kps) and GT poses (gt_pose) of the
EXP015/EXP016 held-out OCC 20 images. Image IDs asserted == EXP015
image_ids.json == EXP016 per_image.csv occ rows. No network forward, no
voting, no GPU needed.

Five conditions, all fed to the SAME original PnP
(Evaluator.evaluate -> pnp, cv2.SOLVEPNP_ITERATIVE, zero dist coeffs,
'linemod' intrinsics, VotingType.Farthest 3D points, cat ply model):
  C0 GT control      : 2D keypoints = GT projected keypoints
  C1 D0              : EXP016 D0 voted keypoints (must reproduce 4/20 flips)
  C2 D1              : EXP016 D1 voted keypoints (must reproduce 14/20 flips)
  C3 D0 + 1px noise  : kp_D0 + eps, eps_ij ~ N(0, 1.0 px^2) iid per coordinate
  C4 D1 + 1px noise  : kp_D1 + eps, SAME per-image/keypoint realization as C3
Noise: seed=0, sigma=1.0 px, ONE run (no sweep). Perturbation norms
(mean/median/p90) reported before PnP; identity (C3-C1 == C4-C2) asserted
elementwise (< 1e-12).

Checks:
  Check A (GT PnP)  : if C0 itself flips ~180 deg on multiple images ->
                      INTRINSIC_AMBIGUITY_SUSPECTED (C1-C4 still completed).
  Check B (repro)   : C1/C2 must reproduce EXP016 occ D0/D1 per-image flip
                      flags exactly (and rot within 1e-3 deg; reproj within
                      1e-3 px) -- inputs are bit-identical so any mismatch is
                      fatal -> STOP (no interpretation of C3/C4).
  Check C (noise)   : noise identity assert (above).

Per condition/image recorded: keypoint mean/median/p90 error, rotation error
(trace formula, EXP016 definition), translation error, ADD (original
Evaluator; cat non-symmetric, reported as in EXP016), ADD pass, mean AND
median 2D reprojection (mean = original Evaluator value; median = same
original projector, observed only), flip flags at 90/120/150/170 deg.

Pre-registered decision rules are written to results/config.json BEFORE any
PnP call and evaluated only in analyze_exp017.py. Run from repo root:
  python experiments/exp017_pose_hypothesis_landscape/run_exp017.py
  python experiments/exp017_pose_hypothesis_landscape/analyze_exp017.py
"""
import os, sys, json, csv, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import numpy as np

from lib.utils.evaluation_utils import Evaluator          # original, unmodified
from lib.datasets.linemod_dataset import VotingType

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(EXP_DIR, 'results')
os.makedirs(RES_DIR, exist_ok=True)
EXP15_DIR = os.path.join(ROOT, 'experiments', 'exp015_keypoint_localization_diagnosis')
EXP16_DIR = os.path.join(ROOT, 'experiments', 'exp016_pnp_stability_diagnosis')

N_IMG, KP = 20, 9
SEED, SIGMA = 0, 1.0
FLIP_THRESH = (90.0, 120.0, 150.0, 170.0)   # EXP016 definition, not re-invented
VOTE_TYPE = VotingType.Farthest
COND_NAMES = ('C0_gt', 'C1_D0', 'C2_D1', 'C3_D0_noise', 'C4_D1_noise')
GEOM_ASSERT = 1e-12

T0 = time.time()
LOG = open(os.path.join(EXP_DIR, 'run.log'), 'w', buffering=1)


def log(msg):
    print(msg, flush=True)
    LOG.write(msg + '\n')


# ---------------- data reuse + identity assertions ----------------
ids15 = json.load(open(os.path.join(EXP15_DIR, 'results', 'image_ids.json')))
OCC_RGB = list(ids15['heldout']['occ']['rgb'])
assert len(OCC_RGB) == N_IMG, f'EXP015 occ heldout count {len(OCC_RGB)} != 20'
json.dump(dict(source='EXP015 results/image_ids.json heldout.occ.rgb '
                    '(asserted == EXP016 per_image.csv occ rows == EXP016 npz ids)',
               image_ids=list(range(N_IMG, 2 * N_IMG)), occ_rgb=OCC_RGB),
          open(os.path.join(RES_DIR, 'image_ids.json'), 'w'), indent=2)
log(f'image IDs asserted == EXP015 heldout occ: {[os.path.splitext(r)[0] for r in OCC_RGB]}')

z0 = np.load(os.path.join(EXP16_DIR, 'results', 'raw_predictions', 'raw_occ_D0.npz'))
z1 = np.load(os.path.join(EXP16_DIR, 'results', 'raw_predictions', 'raw_occ_D1.npz'))
assert list(z0['image_ids']) == list(range(N_IMG, 2 * N_IMG)) == list(z1['image_ids']), \
    'EXP016 npz image ids mismatch -> STOP'
GT_KPS = z0['gt_kps'].astype(np.float64)
GT_POSE = z0['gt_pose'].astype(np.float64)
assert np.array_equal(GT_KPS, z1['gt_kps'].astype(np.float64)), 'gt_kps differ D0/D1 npz -> STOP'
assert np.array_equal(GT_POSE, z1['gt_pose'].astype(np.float64)), 'gt_pose differ D0/D1 npz -> STOP'
KP_D0 = z0['pred_kps'].astype(np.float64)
KP_D1 = z1['pred_kps'].astype(np.float64)
assert KP_D0.shape == KP_D1.shape == (N_IMG, KP, 2), 'unexpected npz kp shape -> STOP'

REF = []  # EXP016 per_image.csv occ rows, same order
with open(os.path.join(EXP16_DIR, 'results', 'per_image.csv')) as f:
    for row in csv.DictReader(f):
        if row['dataset'] == 'occ':
            REF.append(row)
assert len(REF) == N_IMG and [r['rgb'] for r in REF] == OCC_RGB, \
    'EXP016 per_image occ rgb order mismatch -> STOP'
log('EXP016 raw predictions loaded (bit-identical reuse; no re-sampling)')

# ---------------- conditions + pre-registered config (BEFORE any PnP) ----------------
rng = np.random.RandomState(SEED)
EPS = rng.normal(0.0, SIGMA, size=(N_IMG, KP, 2))     # ONE shared realization
CONDS = {'C0_gt': GT_KPS.copy(),
         'C1_D0': KP_D0.copy(),
         'C2_D1': KP_D1.copy(),
         'C3_D0_noise': KP_D0 + EPS,
         'C4_D1_noise': KP_D1 + EPS}

noise_diff = float(np.max(np.abs((CONDS['C3_D0_noise'] - KP_D0)
                                 - (CONDS['C4_D1_noise'] - KP_D1))))
assert noise_diff < GEOM_ASSERT, f'noise identity violated ({noise_diff}) -> STOP'
norms = np.linalg.norm(EPS, axis=2).ravel()
NOISE_STATS = dict(seed=SEED, sigma_px=SIGMA,
                   distribution='eps_x, eps_y ~ N(0, 1.0 px^2) iid per keypoint/coordinate',
                   realization='ONE realization shared by C3 and C4 (identical per-image/keypoint)',
                   n=norms.size, mean_norm_px=float(norms.mean()),
                   median_norm_px=float(np.median(norms)),
                   p90_norm_px=float(np.quantile(norms, 0.9)),
                   theory_mean_norm_px=float(SIGMA * np.sqrt(np.pi / 2)),
                   identity=dict(same_realization=True, max_abs_diff=noise_diff,
                                 tol=GEOM_ASSERT))
json.dump(NOISE_STATS, open(os.path.join(RES_DIR, 'noise_statistics.json'), 'w'), indent=2)
log(f"noise: mean {NOISE_STATS['mean_norm_px']:.4f}  median {NOISE_STATS['median_norm_px']:.4f}  "
    f"p90 {NOISE_STATS['p90_norm_px']:.4f} px (theory mean {NOISE_STATS['theory_mean_norm_px']:.4f}); "
    f"identity max|diff| {noise_diff:.1e}")

json.dump(dict(
    cls='cat', n_img=N_IMG,
    question='EXP016 D1 ~180deg pose flips: intrinsic keypoint/PnP ambiguity (A) vs '
             'residual-geometry-associated basin selection under stable GT (B) vs '
             'perturbation-sensitive numerical instability (C) vs inconclusive (D)',
    data='EXP016 results/raw_predictions/raw_occ_D{0,1}.npz reused bit-identically '
         '(voted keypoints, GT projected keypoints, GT pose); occ image ids == EXP015 '
         'heldout (asserted == EXP016 per_image.csv occ rows); NO re-sampling',
    conditions={
        'C0_gt': '2D keypoints = GT projected keypoints (EXP016 npz gt_kps)',
        'C1_D0': 'EXP016 D0 voted keypoints (reference: 4/20 flips >90deg)',
        'C2_D1': 'EXP016 D1 voted keypoints (reference: 14/20 flips >90deg)',
        'C3_D0_noise': 'C1 + eps (seed=0, sigma=1.0px, single run)',
        'C4_D1_noise': 'C2 + SAME eps realization as C3'},
    noise=dict(seed=SEED, sigma_px=SIGMA, n_runs=1,
               sigma_sweep='FORBIDDEN', extra_seeds='FORBIDDEN'),
    pnp='original Evaluator.evaluate -> pnp (cv2.SOLVEPNP_ITERATIVE, zero dist coeffs, '
        "'linemod' intrinsics, VotingType.Farthest 3D points, cat ply model) -- "
        'UNMODIFIED; no flag/initialization/iteration/RANSAC/refinement/'
        'multi-hypothesis/flip-correction/GT-based-solution-selection',
    flip_definition='rotation_error = acos((trace(R_pred @ R_gt^T)-1)/2); '
                    'flip = rot > 90/120/150/170 deg (EXP016 definition)',
    add_note='cat is non-symmetric -> original Evaluator add_metric (ADD); reported as in EXP016',
    median_reprojection_note='median reprojection uses the SAME original projector/model/K as '
                             "Evaluator.projection_2d (observation only; mean value is the "
                             "Evaluator's own proj_mean_diffs, cross-checked < 1e-9 px)",
    reproduction_reference=dict(C1_D0_flip_gt90=4, C2_D1_flip_gt90=14,
                                source='EXP015/EXP016 standard-PnP outputs'),
    checks=dict(
        check_A='C0 GT PnP: flip counts/rot/reproj; INTRINSIC_AMBIGUITY_SUSPECTED if any '
                'C0 rot > 170 deg; C1-C4 still completed, no extra experiments',
        check_B='C1/C2 must reproduce EXP016 occ D0/D1 per-image flip flags EXACTLY and rot '
                'within 1e-3 deg (reproj within 1e-3 px); inputs bit-identical, so any '
                'mismatch -> STOP (no interpretation of C3/C4)',
        check_C='noise identity: same eps realization C3/C4, max elementwise diff < 1e-12'),
    decision_rules_pre_registered=dict(
        priority='A -> C -> B -> D, first match wins (C before B: if matched noise erases '
                 'the D0/D1 gap, instability dominates and geometry association cannot stand)',
        CASE_A='C0 flip_gt90 >= 3 (multiple images) -> intrinsic ambiguity; FORBIDDEN to '
               'design a solver afterwards',
        CASE_C='C0 stable (flip_gt90 <= 1) AND (C3 flip_gt90 >= 10 OR C4 flip_gt90 >= 10) '
               'AND |C3 - C4| flip_gt90 <= 4 -> perturbation-sensitive numerical instability '
               '(D0/D1 difference largely disappears under matched perturbation)',
        CASE_B='C0 stable AND C1 flip_gt90 <= 6 AND C2 flip_gt90 >= 10 AND D1 flip-group '
               'kp_median <= 1.5x stable-group (magnitude not explanatory) AND >=1 of '
               '{relative_scatter_mean, coherent_shift_norm, anisotropy_ratio} shows D1 '
               'flip/stable median ratio >= 1.5 or <= 1/1.5 -> residual-geometry-associated '
               'basin selection (association language ONLY, no causality beyond diagnosis)',
        CASE_D='otherwise INCONCLUSIVE; no sigma sweep / seeds / datasets / new solver'),
    language_limits='cat-only, 20-image mechanism diagnosis; no significance claims'),
    open(os.path.join(EXP_DIR, 'config.json'), 'w'), indent=2)
log('pre-registered config.json written (BEFORE any PnP call)')

# ---------------- main: 5 conditions x 20 images x ORIGINAL PnP ----------------
rows = []
for cond in COND_NAMES:
    ev = Evaluator()                                   # fresh original evaluator per condition
    model = ev.linemod_db.get_ply_model('cat')
    K = ev.projector.intrinsic_matrix['linemod']
    for i in range(N_IMG):
        kp = CONDS[cond][i]
        err = np.linalg.norm(kp - GT_KPS[i], axis=1)
        fin = err[np.isfinite(err)]
        pnp_ok = True
        try:
            pose = ev.evaluate(kp, GT_POSE[i], 'cat', 'linemod', VOTE_TYPE, intri_matrix=None)
        except Exception as e:                          # record only (EXP016 convention)
            pnp_ok = False
            log(f'PnP EXCEPTION [{cond} {OCC_RGB[i]}]: {e}')
        if pnp_ok:
            rot = float(np.rad2deg(np.arccos(np.clip(
                (np.trace(pose[:, :3] @ GT_POSE[i][:, :3].T) - 1) / 2, -1, 1))))
            trans_mm = float(np.linalg.norm(pose[:, 3] - GT_POSE[i][:, 3]) * 1000.0)
            add_mm = float(ev.add_dists[-1]) * 1000.0
            add_pass = bool(ev.add_recorder[-1])
            reproj_mean = float(ev.proj_mean_diffs[-1])
            m2p = ev.projector.project_K(model, pose, K)          # original projector
            m2t = ev.projector.project_K(model, GT_POSE[i], K)
            dist = np.linalg.norm(m2p - m2t, axis=1)
            reproj_median = float(np.median(dist))
            assert abs(float(np.mean(dist)) - reproj_mean) < 1e-9, \
                f'reprojection mean mismatch [{cond} {OCC_RGB[i]}] -> STOP'
        else:
            rot = trans_mm = add_mm = reproj_mean = reproj_median = float('nan')
            add_pass = False
        rows.append(dict(
            condition=cond, image_id=N_IMG + i, rgb=OCC_RGB[i],
            keypoint_mean_error_px=float(fin.mean()) if fin.size else float('nan'),
            keypoint_median_error_px=float(np.median(fin)) if fin.size else float('nan'),
            keypoint_p90_error_px=float(np.quantile(fin, 0.9)) if fin.size else float('nan'),
            rotation_error_deg=rot, translation_error_mm=trans_mm,
            add_mm=add_mm, add_pass=add_pass,
            mean_reprojection_error_px=reproj_mean,
            median_reprojection_error_px=reproj_median,
            pnp_ok=pnp_ok,
            **{f'flip_gt_{int(t)}': bool(pnp_ok and rot > t) for t in FLIP_THRESH}))
    log(f'--- {cond}: 20 images through original PnP ({time.time() - T0:.1f}s) ---')


# ---------------- Check A: GT PnP (intrinsic ambiguity probe) ----------------
c0 = [r for r in rows if r['condition'] == 'C0_gt']
c0_flips = sum(r['flip_gt_90'] for r in c0)
c0_rots = [r['rotation_error_deg'] for r in c0]
intrinsic_suspect = any(np.isfinite(v) and v > 170.0 for v in c0_rots)
log(f'CHECK A (C0 GT): flip>90 {c0_flips}/20  rot list {np.round(c0_rots, 2).tolist()}')
log(f'CHECK A: INTRINSIC_AMBIGUITY_SUSPECTED = {intrinsic_suspect} '
    f'(any C0 rot > 170 deg); C1-C4 still completed, no extra experiments')


# ---------------- Check B: exact reproduction of EXP016 C1/C2 ----------------
def _ref_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


check_b_rows, check_b_ok = [], True
max_rot_diff = max_reproj_diff = 0.0
for i in range(N_IMG):
    for cond, arm in (('C1_D0', 'D0'), ('C2_D1', 'D1')):
        mine = next(r for r in rows if r['condition'] == cond and r['image_id'] == N_IMG + i)
        ref_rot = _ref_float(REF[i][f'{arm}_rotation_error'])
        ref_flip = int(REF[i][f'{arm}_flip'])
        ref_reproj = _ref_float(REF[i][f'{arm}_reprojection_error'])
        my_rot, my_flip = mine['rotation_error_deg'], int(mine['flip_gt_90'])
        my_reproj = mine['mean_reprojection_error_px']
        rot_diff = abs(my_rot - ref_rot) if (np.isfinite(my_rot) and np.isfinite(ref_rot)) \
            else (0.0 if (not np.isfinite(my_rot) and not np.isfinite(ref_rot)) else float('inf'))
        rep_diff = abs(my_reproj - ref_reproj) if (np.isfinite(my_reproj) and np.isfinite(ref_reproj)) \
            else (0.0 if (not np.isfinite(my_reproj) and not np.isfinite(ref_reproj)) else float('inf'))
        ok_i = (my_flip == ref_flip) and rot_diff < 1e-3 and rep_diff < 1e-3
        check_b_ok &= ok_i
        max_rot_diff = max(max_rot_diff, rot_diff if np.isfinite(rot_diff) else 1e9)
        max_reproj_diff = max(max_reproj_diff, rep_diff if np.isfinite(rep_diff) else 1e9)
        check_b_rows.append(dict(condition=cond, image_id=N_IMG + i, rgb=OCC_RGB[i],
                                 my_flip=my_flip, ref_flip=ref_flip,
                                 my_rot=my_rot, ref_rot=ref_rot, rot_diff=float(rot_diff),
                                 my_reproj=my_reproj, ref_reproj=ref_reproj,
                                 reproj_diff=float(rep_diff), ok=bool(ok_i)))
log(f'CHECK B: C1/C2 vs EXP016 occ per-image flips+rot+reproj -> ok={check_b_ok}  '
    f'max|rot diff| {max_rot_diff:.2e} deg  max|reproj diff| {max_reproj_diff:.2e} px')
if not check_b_ok:
    log('CHECK B FAILED -> STOP (mismatch reported; C3/C4 not interpreted)')

json.dump(dict(
    check_A=dict(c0_flip_gt90=int(c0_flips), c0_rotation_errors_deg=c0_rots,
                 intrinsic_ambiguity_suspected=bool(intrinsic_suspect)),
    check_B=dict(ok=bool(check_b_ok), max_rot_diff_deg=float(max_rot_diff),
                 max_reproj_diff_px=float(max_reproj_diff), per_image=check_b_rows),
    check_C=dict(ok=bool(noise_diff < GEOM_ASSERT), max_abs_diff=noise_diff,
                 same_realization=True)),
    open(os.path.join(RES_DIR, 'reproduction_check.json'), 'w'), indent=2)

# ---------------- per_image.csv (long format: 5 conditions x 20 images) ----------------
with open(os.path.join(RES_DIR, 'per_image.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['condition', 'image_id', 'rgb',
                 'keypoint_mean_error_px', 'keypoint_median_error_px', 'keypoint_p90_error_px',
                 'rotation_error_deg', 'translation_error_mm',
                 'add_mm', 'add_pass',
                 'mean_reprojection_error_px', 'median_reprojection_error_px',
                 'flip_gt_90', 'flip_gt_120', 'flip_gt_150', 'flip_gt_170', 'pnp_ok'])
    for r in rows:
        wr.writerow([r['condition'], r['image_id'], r['rgb'],
                     f"{r['keypoint_mean_error_px']:.4f}", f"{r['keypoint_median_error_px']:.4f}",
                     f"{r['keypoint_p90_error_px']:.4f}",
                     f"{r['rotation_error_deg']:.4f}" if r['pnp_ok'] else 'nan',
                     f"{r['translation_error_mm']:.2f}" if r['pnp_ok'] else 'nan',
                     f"{r['add_mm']:.4f}" if r['pnp_ok'] else 'nan',
                     int(r['add_pass']),
                     f"{r['mean_reprojection_error_px']:.4f}" if r['pnp_ok'] else 'nan',
                     f"{r['median_reprojection_error_px']:.4f}" if r['pnp_ok'] else 'nan',
                     int(r['flip_gt_90']), int(r['flip_gt_120']),
                     int(r['flip_gt_150']), int(r['flip_gt_170']),
                     int(r['pnp_ok'])])

for cond in COND_NAMES:
    rs = [r for r in rows if r['condition'] == cond]
    log(f"flips [{cond}]: " + '  '.join(
        f'>{int(t)}deg {sum(r[f"flip_gt_{int(t)}"] for r in rs)}/20' for t in FLIP_THRESH))
log(f'DONE in {time.time() - T0:.1f}s (no GPU used; original PnP unmodified)')
