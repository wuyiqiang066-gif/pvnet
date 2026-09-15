"""EXP020: Keypoint Residual Magnitude vs PnP Stability Threshold Diagnosis.

Research question (EXP019 follow-up): EXP019 closed geometry mining (D1-like
anisotropy / coherent shift received no causal support; matched-RMS isotropic
zero-shift noise alone reproduced most of the real D1 flip burst). EXP020 asks
the one remaining basic question:

    does the ORIGINAL PVNet PnP exhibit a stable keypoint residual MAGNITUDE
    -> pose-ambiguity THRESHOLD / phase-transition regime, or does flip
    probability degrade smoothly with residual magnitude?

NOT a claim of a universal threshold: only a diagnosis of whether a clear
nonlinear transition (knee) exists on the fixed 20-image OCC set.

Experiment boundary (protocol section 2, asserted at runtime):
  - CPU-only, seed=0, no training, no GPU, no new object, no full dataset;
  - the exact 20 OCC held-out images of EXP017/018/019 (asserted == EXP019
    image_ids.json);
  - GT 2D/3D keypoints, GT poses, linemod intrinsics, original 3D Farthest
    points -- bit-identical reuse from EXP016 raw npz;
  - ORIGINAL Evaluator.evaluate -> pnp() -> cv2.solvePnP(SOLVEPNP_ITERATIVE),
    flags/termination/thresholds UNTOUCHED; no solvePnPGeneric, no new solver;
  - keypoint count unchanged (9).
The ONLY manipulated variable: synthetic isotropic residual MAGNITUDE fed to
PnP (kp_input = GT_kp + r). Images, masks, vertices, voting, poses untouched.

GEOMETRY IS CLOSED (EXP019): residuals are isotropic Gaussian, ZERO coherent
shift by construction:
    r_i(sigma) = sigma * z_i,   z_i ~ N(0, I)
NO sample-mean removal, NO normalization, NO covariance shaping (protocol
section 7: the sampled mean need not be exactly zero and is NOT adjusted).

Shared realization (protocol section 8): ONE base field z ~ N(0,I), shape
(20, 9, 2), numpy default_rng(0) -- the SAME seed/shape/order as EXP019, so
the base field is bit-identical across the two experiments; every magnitude
level is the deterministic transform r = sigma * z of the SAME z. No
per-sigma re-sampling. sigma is therefore the ONLY experimental variable.

Magnitude levels (protocol section 9, 12 levels, NOT densified):
    0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00 px
sigma = 0.0 is the GT-keypoint level (identical input to gate C0; asserted
per-image identical output). Actual sampled magnitudes are RECORDED (protocol
section 10: per-condition RMS / mean_norm / median_norm / p90_norm, plus
per-image RMS) and the dose-response analysis uses ACTUAL RMS, not nominal
sigma.

Objective oracle (EXP018 corrected sign convention, unchanged):
    J(pose) = mean_i || project_K(kp3d_Farthest9, pose)_i - kp_input_i || (px)
    J_pred = fit of the returned pose; J_gt = fit of the GT pose
    delta_J = J_gt - J_pred; delta_J > 0 <=> J_pred < J_gt <=> the returned
    pose fits the input keypoints better than the correct pose (objective
    ambiguity signal). The sign is NOT re-inverted.

Flip: rotation error > 90/120/150/170 deg (EXP016/017 definition). Primary
pre-registered indicator: >150 deg.

Pathological-image policy (protocol sections 14/15, fixed BEFORE results):
  CLEAN18 := all images except color_00026.png and color_00027.png (fixed
  from EXP018/019; result-independent). SOLVER_DIVERGENCE := J_pred > 100 px
  or official reproj > 100 px (flagged, reported separately, never read as a
  normal phase transition). CATASTROPHIC_KEYPOINT := median input keypoint
  error > 100 px (real-residual conditions). ALL 20 and CLEAN 18 are BOTH
  always reported; nothing is deleted.

Pre-registration / anti-circularity:
  1. Real D0/D1 median per-image RMS placement anchors computed with numpy
     ONLY before any PnP call; D1 anchor cross-checked against EXP019's
     frozen config (assert < 1e-3).
  2. config.json (sigma levels, formula, objective, clean subset, decision
     criteria) written BEFORE any synthetic PnP; never modified afterwards.
  3. Baseline gate: C0 GT / C1 D0 / C2 D1 through the ORIGINAL pipeline must
     exactly reproduce EXP017 (flips + rot/reproj/trans), EXP016 poses
     bit-exactly, and EXP018 oracle J; any mismatch -> STOP (exit 1).
  4. Only then the 12-level synthetic dose-response runs.

Run from repo root:
  python experiments/exp020_residual_magnitude_pnp_threshold/scripts/run_exp020.py
"""
import os, sys, json, csv, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)

import numpy as np

from lib.utils.evaluation_utils import Evaluator          # original, unmodified
from lib.datasets.linemod_dataset import VotingType

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(EXP_DIR, 'results')
os.makedirs(RES_DIR, exist_ok=True)
EXP16_DIR = os.path.join(ROOT, 'experiments', 'exp016_pnp_stability_diagnosis')
EXP17_DIR = os.path.join(ROOT, 'experiments', 'exp017_pose_hypothesis_landscape')
EXP18_DIR = os.path.join(ROOT, 'experiments', 'exp018_pose_hypothesis_oracle')
EXP19_DIR = os.path.join(ROOT, 'experiments', 'exp019_residual_geometry_dose_response')

N_IMG, KP = 20, 9
SEED = 0
FLIP_THRESH = (90.0, 120.0, 150.0, 170.0)   # EXP016/017 definition
PRIMARY_FLIP = 150.0                        # ~180 deg ambiguity regime
SIGMAS = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00)
VOTE_TYPE = VotingType.Farthest
DIVERGENCE_J_PX = 100.0                     # SOLVER_DIVERGENCE flag threshold
CATASTROPHIC_KP_PX = 100.0                  # CATASTROPHIC_KEYPOINT flag threshold
PATHOLOGICAL = ('color_00026.png', 'color_00027.png')   # fixed (EXP018/019)


def sig_name(s):
    return 'SIG_S%03d' % round(s * 100)


COND_TABLE = [('C0_gt', 'control', None), ('C1_D0_real', 'control', None),
              ('C2_D1_real', 'control', None)] + \
             [(sig_name(s), 'synthetic', s) for s in SIGMAS]
EXP17_NAME = {'C0_gt': 'C0_gt', 'C1_D0_real': 'C1_D0', 'C2_D1_real': 'C2_D1'}

T0 = time.time()
LOG = open(os.path.join(EXP_DIR, 'run.log'), 'w', buffering=1)


def log(msg):
    print(msg, flush=True)
    LOG.write(msg + '\n')


def stop(msg):
    log(f'STOP: {msg}')
    json.dump(dict(stopped=True, reason=msg),
              open(os.path.join(RES_DIR, 'gate_failure.json'), 'w'), indent=2)
    sys.exit(1)


# ---------------- 1. image IDs: assert == EXP019 (== EXP017/018) -------------
ids19 = json.load(open(os.path.join(EXP19_DIR, 'image_ids.json')))
OCC_RGB = list(ids19['occ_rgb'])
assert len(OCC_RGB) == N_IMG, 'EXP019 occ id count != 20 -> STOP'
ids17 = json.load(open(os.path.join(EXP17_DIR, 'results', 'image_ids.json')))
assert list(ids17['occ_rgb']) == OCC_RGB, 'EXP017 ids != EXP019 ids -> STOP'
ids18 = json.load(open(os.path.join(EXP18_DIR, 'image_ids.json')))
assert list(ids18['occ_rgb']) == OCC_RGB, 'EXP018 ids != EXP019 ids -> STOP'
CLEAN_MASK = np.array([r not in PATHOLOGICAL for r in OCC_RGB])
json.dump(dict(source='EXP019 image_ids.json (asserted identical; EXP017/018 '
                    'cross-checked identical)',
               image_ids=list(range(N_IMG, 2 * N_IMG)), occ_rgb=OCC_RGB),
          open(os.path.join(EXP_DIR, 'image_ids.json'), 'w'), indent=2)
log(f'image IDs asserted == EXP019 == EXP017 == EXP018: '
    f'{[os.path.splitext(r)[0] for r in OCC_RGB]}')

# ---------------- 2. bit-identical inputs from EXP016 npz ----------------
z0 = np.load(os.path.join(EXP16_DIR, 'results', 'raw_predictions', 'raw_occ_D0.npz'))
z1 = np.load(os.path.join(EXP16_DIR, 'results', 'raw_predictions', 'raw_occ_D1.npz'))
assert list(z0['image_ids']) == list(range(N_IMG, 2 * N_IMG)) == list(z1['image_ids'])
GT_KPS = z0['gt_kps'].astype(np.float64)
GT_POSE = z0['gt_pose'].astype(np.float64)
assert np.array_equal(GT_KPS, z1['gt_kps'].astype(np.float64))
assert np.array_equal(GT_POSE, z1['gt_pose'].astype(np.float64))
NPZ_POSE = {'C1_D0': z0['pred_pose'].astype(np.float64),
            'C2_D1': z1['pred_pose'].astype(np.float64)}
KP_REAL = {'C1_D0_real': z0['pred_kps'].astype(np.float64),   # fed DIRECTLY
           'C2_D1_real': z1['pred_kps'].astype(np.float64)}
R0 = KP_REAL['C1_D0_real'] - GT_KPS                       # real residual fields
R1 = KP_REAL['C2_D1_real'] - GT_KPS

# ---------------- 3. D0/D1 magnitude placement anchors (numpy ONLY) ----------
def per_image_rms(R):
    return [float(np.sqrt(np.mean(np.linalg.norm(R[j], axis=1) ** 2)))
            for j in range(R.shape[0])]


RMS0_PER = per_image_rms(R0)
RMS1_PER = per_image_rms(R1)
D0_RMS_MED = float(np.median(RMS0_PER))
D1_RMS_MED = float(np.median(RMS1_PER))
cfg19 = json.load(open(os.path.join(EXP19_DIR, 'config.json')))
assert abs(D1_RMS_MED - cfg19['d1_like_targets']['rms_anchor_median_per_image']) \
    < 1e-3, 'D1 RMS anchor != EXP019 frozen anchor -> STOP'
log(f'real magnitude placement anchors (numpy only, BEFORE any PnP): '
    f'D0 median per-image RMS {D0_RMS_MED:.4f} px | D1 {D1_RMS_MED:.4f} px '
    f'(cross-checked == EXP019 frozen anchor)')

# ---------------- 4. pre-registered config (BEFORE any PnP call) ------------
json.dump(dict(
    seed=SEED, n_img=N_IMG, kp_per_img=KP,
    image_ids=list(range(N_IMG, 2 * N_IMG)), occ_rgb=OCC_RGB,
    cls='cat', vote_type='VotingType.Farthest',
    pnp_path='lib.utils.evaluation_utils.Evaluator.evaluate -> pnp() -> '
             'cv2.solvePnP(SOLVEPNP_ITERATIVE); UNMODIFIED (no flags/threshold/'
             'termination change; no solvePnPGeneric; no new solver)',
    geometry_closed_note='EXP019 closed geometry mining (GO-C): anisotropy / '
                         'coherent shift / covariance shape NOT studied here; '
                         'residuals are isotropic Gaussian with zero coherent '
                         'shift by construction',
    residual=dict(
        formula='r_i(sigma) = sigma * z_i; z_i ~ N(0,I); NO mean removal, NO '
                'normalization, NO covariance shaping (protocol section 7: '
                'sampled mean need not be exactly zero and is NOT adjusted)',
        shared_realization='ONE base z ~ N(0,I) (20,9,2), numpy default_rng(0), '
                           'same seed/shape/order as EXP019 -> bit-identical '
                           'base field; every level is the deterministic '
                           'transform r = sigma * z of the SAME z (protocol '
                           'section 8; no per-sigma re-sampling)',
        sigma_levels_px=list(SIGMAS),
        magnitude_stats_recorded=['per-image RMS', 'per-condition RMS (mean over '
                                  'images; primary x-axis)', 'pooled RMS',
                                  'mean_norm', 'median_norm', 'p90_norm',
                                  'coherent shift (recorded to confirm ~0, '
                                  'NOT studied)'],
        actual_vs_nominal='dose-response analysis uses ACTUAL RMS (protocol '
                          'section 10); nominal sigma kept only as label'),
    objective=dict(
        definition='J(pose) = mean_i || project_K(kp3d_Farthest9, pose)_i - '
                   'kp_input_i || (px); original Projector, linemod intrinsics',
        J_pred='fit of solver-returned pose to input keypoints',
        J_gt='fit of GT pose to input keypoints',
        delta_J='J_gt - J_pred (EXP018 corrected sign, NOT re-inverted); '
                'delta_J > 0 <=> J_pred < J_gt <=> returned pose fits input '
                'keypoints strictly better than the correct pose (objective '
                'ambiguity signal)'),
    flip=dict(thresholds_deg=list(FLIP_THRESH), primary_gt_deg=PRIMARY_FLIP,
              source='EXP016/017 definition reused'),
    clean_subset='CLEAN18 = ALL20 minus color_00026.png and color_00027.png '
                 '(fixed from EXP018/019 before any EXP020 result; '
                 'result-independent; ALL20 and CLEAN18 BOTH always reported',
    pathological_policy=dict(
        solver_divergence=f'J_pred > {DIVERGENCE_J_PX:.0f} px or official reproj '
                          f'> {DIVERGENCE_J_PX:.0f} px -> SOLVER_DIVERGENCE flag; '
                          'reported separately, never read as a normal phase '
                          'transition',
        catastrophic_keypoint=f'median input keypoint error > '
                              f'{CATASTROPHIC_KP_PX:.0f} px -> '
                              'CATASTROPHIC_KEYPOINT flag (real-residual conds)'),
    d0_d1_placement=dict(
        d0_median_per_image_rms_px=D0_RMS_MED,
        d1_median_per_image_rms_px=D1_RMS_MED,
        note='real residual RMS placement anchors, numpy ONLY pre-PnP; D1 '
             'anchor asserted == EXP019 frozen config (<1e-3); real D0/D1 are '
             'also run DIRECTLY as gate conditions C1/C2 (no modification)',
        known_caveat='D0 median RMS (3.42 px) > D1 (2.42 px) while D0 flips '
                     'LESS (4 vs 14 at >90deg): per protocol section 22 this '
                     'is recorded as "magnitude is a major driver but not the '
                     'sole determinant"; geometry mining stays CLOSED'),
    gate=dict(
        c0='GT keypoints: flips 0/20 at all thresholds, max rot < 1e-3 deg, '
           'max official reproj < 1e-2 px',
        c1_c2='D0/D1 real keypoints: flips exact vs EXP017 (4/20, 14/20 at '
              '>90deg) + rot<1e-3/reproj<1e-3/trans<1e-2 + poses bit-match '
              'EXP016 npz (<1e-9) + J_pred/J_gt match EXP018 objective_oracle.csv '
              '(<1e-4); SIG_S000 additionally asserted per-image identical to C0_gt',
        on_failure='STOP (exit 1), no synthetic run, no interpretation'),
    decision_rules_pre_registered=dict(
        primary='CLEAN18 flip150 rate vs ACTUAL per-condition RMS (mean over '
                'images); ALL20 reported alongside; descriptive statistics and '
                'simple finite differences ONLY (protocol section 18: no '
                'model fitting, no change-point optimization)',
        transition_diagnostic='delta_flip150[k] = rate[k+1] - rate[k] over the '
                              '12 ordered levels; jump_max = max over adjacent '
                              'pairs; transition pair = argmax; same for '
                              'P(delta_J>0) (delta_P_RA)',
        GO_A='clear transition: ALL of (i) jump_max >= 0.20 on CLEAN18 flip150 '
             '(>= ~4/18 images within one step); (ii) delta_J sync: '
             'median_delta_J non-decreasing across >= 9 of 11 adjacent steps '
             'AND delta_P_RA across the transition pair >= +0.10; (iii) not '
             '1-2 image driven: >= 3 images have first_sigma_flip150 inside '
             'the transition pair (CLEAN18 images; SOLVER_DIVERGENCE events '
             'not counted as drivers); (iv) placement: real D1 median '
             'per-image RMS within [lower_sigma - 0.5, upper_sigma + 0.5] of '
             'the transition pair. Conclusion: residual-magnitude-dependent '
             'pose-stability transition in the original PVNet PnP pipeline',
        GO_B='smooth degradation: flip150(sigma_max) >= flip150(sigma_min) + '
             '0.30 overall (dose-response exists) but GO_A criteria fail. '
             'Conclusion: stability degrades continuously; NO threshold '
             'wording',
        GO_C='magnitude insufficient: otherwise (weak/inconsistent relation). '
             'STOP; no geometry mining restart; revisit higher-level keypoint '
             'configuration / object geometry',
        statistics_discipline='n=20: NO significance claims, NO universal/'
                              'object-independent/dataset-independent/solver-'
                              'independent threshold wording; diagnostic '
                              'evidence on the fixed 20-image OCC set only'),
    forbidden='no solver/flag/threshold/seed/GT/intrinsics/3D-points/image-set '
              'change; no sigma level added post hoc; no geometry study '
              '(EXP019 closed); config frozen BEFORE any PnP (protocol '
              'sections 7/28)',
    runtime_budget='CPU < 5 min target; abort-and-report if solver modification '
                   'would ever be needed',
), open(os.path.join(EXP_DIR, 'config.json'), 'w'), indent=2)
log('pre-registered config.json written (BEFORE any PnP call)')

# ---------------- 5. baseline gate: C0/C1/C2 through ORIGINAL PnP ------------
REF17 = {c: [] for c in ('C0_gt', 'C1_D0', 'C2_D1')}
with open(os.path.join(EXP17_DIR, 'results', 'per_image.csv')) as f:
    for row in csv.DictReader(f):
        if row['condition'] in REF17:
            REF17[row['condition']].append(row)
for c in REF17:
    assert len(REF17[c]) == N_IMG and [r['rgb'] for r in REF17[c]] == OCC_RGB, \
        f'EXP017 {c} rows/rgb order mismatch -> STOP'
ORA18 = {}
with open(os.path.join(EXP18_DIR, 'results', 'objective_oracle.csv')) as f:
    for row in csv.DictReader(f):
        ORA18.setdefault(row['condition'], []).append(row)

ev = Evaluator()
KP3D = VotingType.get_pts_3d(VOTE_TYPE, 'cat')
K = ev.projector.intrinsic_matrix['linemod']


def run_pnp(kp_in, i):
    """original pipeline, unmodified; returns dict or exception flag."""
    out = dict(exception=False)
    try:
        pose = ev.evaluate(kp_in, GT_POSE[i], 'cat', 'linemod', VOTE_TYPE,
                           intri_matrix=None)
        proj_pred = ev.projector.project_K(KP3D, pose, K)
        proj_gtp = ev.projector.project_K(KP3D, GT_POSE[i], K)
        out.update(
            pose=pose,
            rot=float(np.rad2deg(np.arccos(np.clip(
                (np.trace(pose[:, :3] @ GT_POSE[i][:, :3].T) - 1) / 2, -1, 1)))),
            trans_mm=float(np.linalg.norm(pose[:, 3] - GT_POSE[i][:, 3]) * 1000.0),
            add_mm=float(ev.add_dists[-1]) * 1000.0,
            reproj=float(ev.proj_mean_diffs[-1]),
            J_pred=float(np.mean(np.linalg.norm(proj_pred - kp_in, axis=1))),
            J_gt=float(np.mean(np.linalg.norm(proj_gtp - kp_in, axis=1))))
    except Exception as e:                                   # noqa: BLE001
        out['exception'] = True
        out['err'] = str(e)
        log(f'PnP EXCEPTION [{i} {OCC_RGB[i]}]: {e}')
    return out


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


gate_rows = {}
for cond, kp_in_all in (('C0_gt', GT_KPS), ('C1_D0', KP_REAL['C1_D0_real']),
                        ('C2_D1', KP_REAL['C2_D1_real'])):
    rows_g = []
    for i in range(N_IMG):
        o = run_pnp(kp_in_all[i], i)
        ref = REF17[cond][i]
        rot = o.get('rot', float('nan'))
        reproj = o.get('reproj', float('nan'))
        ok = ((not o['exception'])
              and int(rot > 90.0) == int(ref['flip_gt_90'])
              and abs(rot - fnum(ref['rotation_error_deg'])) < 1e-3
              and abs(reproj - fnum(ref['mean_reprojection_error_px'])) < 1e-3
              and abs(o['trans_mm'] - fnum(ref['translation_error_mm'])) < 1e-2)
        if cond in NPZ_POSE:
            d = float(np.max(np.abs(o['pose'] - NPZ_POSE[cond][i])))
            assert d < 1e-9, f'{cond} {OCC_RGB[i]}: pose != EXP016 npz ({d}) -> STOP'
        ec = {'C0_gt': None, 'C1_D0': 'C1_D0', 'C2_D1': 'C2_D1'}[cond]
        if ec:
            r18 = ORA18[ec][i]
            assert abs(o['J_pred'] - float(r18['J_pred_px'])) < 1e-4 and \
                abs(o['J_gt'] - float(r18['J_gt_px'])) < 1e-4, \
                f'{cond} {OCC_RGB[i]}: oracle J != EXP018 -> STOP'
        rows_g.append((ok, o))
    gate_rows[cond] = rows_g
    log(f'gate {cond}: flips>90deg '
        f'{sum(int(r[1]["rot"] > 90) for r in rows_g)}/20, max rot '
        f'{max(r[1]["rot"] for r in rows_g):.4f} deg, max reproj '
        f'{max(r[1]["reproj"] for r in rows_g):.4f} px')

gate_ok = (all(r[0] for rs in gate_rows.values() for r in rs)
           and all(r[1]['rot'] <= 1e-3 and r[1]['reproj'] <= 1e-2
                   for r in gate_rows['C0_gt'])
           and sum(int(r[1]['rot'] > 90) for r in gate_rows['C1_D0']) == 4
           and sum(int(r[1]['rot'] > 90) for r in gate_rows['C2_D1']) == 14)
json.dump(dict(gate_ok=bool(gate_ok),
               c0_flips=sum(int(r[1]['rot'] > t) for r in gate_rows['C0_gt']
                            for t in FLIP_THRESH),
               c1_flips_gt90=sum(int(r[1]['rot'] > 90) for r in gate_rows['C1_D0']),
               c2_flips_gt90=sum(int(r[1]['rot'] > 90) for r in gate_rows['C2_D1']),
               checks='flips+rot+reproj+trans vs EXP017; poses bit-match EXP016 '
                      'npz; oracle J vs EXP018'),
          open(os.path.join(RES_DIR, 'reproduction_check.json'), 'w'), indent=2)
if not gate_ok:
    stop('baseline gate failed: C0/C1/C2 do not exactly reproduce EXP017/018')
log(f'GATE PASSED ({time.time() - T0:.1f}s): C0 0/20, C1 4/20, C2 14/20 '
    f'(>90deg); poses bit-match EXP016; oracle J matches EXP018')

# ---------------- 6. shared base realization (seed=0, ONCE) ------------------
# same seed/shape/order as EXP019 -> bit-identical base field across experiments
rng = np.random.default_rng(SEED)
Z = rng.standard_normal((N_IMG, KP, 2))
log(f'base realization Z generated once: shape {Z.shape}, seed {SEED}, '
    f'checksum {float(Z.sum()):.6f}')

# ---------------- 7. all conditions ------------------------------------------
ROWS = []
C0_ROT = None
for cond, grp, sigma in COND_TABLE:
    if grp == 'control':
        KP_IN = {'C0_gt': GT_KPS,
                 'C1_D0_real': KP_REAL['C1_D0_real'],
                 'C2_D1_real': KP_REAL['C2_D1_real']}[cond]
        R_FIELD = {'C0_gt': np.zeros_like(GT_KPS),
                   'C1_D0_real': R0, 'C2_D1_real': R1}[cond]
    else:
        R_FIELD = sigma * Z                      # r = sigma * z, NOTHING else
        KP_IN = GT_KPS + R_FIELD
    for i in range(N_IMG):
        o = run_pnp(KP_IN[i], i)
        r = R_FIELD[i]
        rn = np.linalg.norm(r, axis=1)
        kp_err = np.linalg.norm(KP_IN[i] - GT_KPS[i], axis=1)
        div = bool(o.get('J_pred', 0.0) > DIVERGENCE_J_PX
                   or o.get('reproj', 0.0) > DIVERGENCE_J_PX)
        cat = bool(np.median(kp_err) > CATASTROPHIC_KP_PX)
        rot = o.get('rot', float('nan'))
        if cond == 'C0_gt':
            C0_ROT = rot
        if cond == 'SIG_S000':
            assert abs(rot - C0_ROT) < 1e-9 and \
                abs(o.get('J_pred', 0) - gate_rows['C0_gt'][i][1]['J_pred']) < 1e-9, \
                f'SIG_S000 != C0_gt on {OCC_RGB[i]} -> STOP'
        ROWS.append(dict(
            condition=cond, group=grp, sigma=sigma,
            image_id=N_IMG + i, rgb=OCC_RGB[i],
            rot_err_deg=rot, trans_err_mm=o.get('trans_mm', float('nan')),
            add_mm=o.get('add_mm', float('nan')),
            reproj_official_px=o.get('reproj', float('nan')),
            **{f'flip_gt_{int(t)}': bool(np.isfinite(rot) and rot > t)
               for t in FLIP_THRESH},
            J_gt_px=o.get('J_gt', float('nan')),
            J_pred_px=o.get('J_pred', float('nan')),
            delta_J_px=(o['J_gt'] - o['J_pred'])
            if 'J_gt' in o else float('nan'),
            res_rms_px=float(np.sqrt(np.mean(rn ** 2))),
            res_meannorm_px=float(rn.mean()),
            res_coherent_px=float(np.linalg.norm(r.mean(axis=0))),
            res_median_px=float(np.median(rn)),
            res_p90_px=float(np.percentile(rn, 90)),
            kp_err_median_px=float(np.median(kp_err)),
            flag_solver_divergence=div, flag_catastrophic_kp=cat,
            flag_pnp_exception=bool(o['exception']),
            in_clean18=bool(CLEAN_MASK[i])))
    log(f'--- {cond}: 20 images done ({time.time() - T0:.1f}s) ---')

# ---------------- 8. persist --------------------------------------------------
COLS = ['condition', 'group', 'sigma', 'image_id', 'rgb', 'rot_err_deg',
        'trans_err_mm', 'add_mm', 'reproj_official_px', 'flip_gt_90',
        'flip_gt_120', 'flip_gt_150', 'flip_gt_170', 'J_gt_px', 'J_pred_px',
        'delta_J_px', 'res_rms_px', 'res_meannorm_px', 'res_coherent_px',
        'res_median_px', 'res_p90_px', 'kp_err_median_px',
        'flag_solver_divergence', 'flag_catastrophic_kp', 'flag_pnp_exception',
        'in_clean18']
FMT = {'rot_err_deg': '.4f', 'trans_err_mm': '.2f', 'add_mm': '.4f',
       'reproj_official_px': '.4f', 'J_gt_px': '.6f', 'J_pred_px': '.6f',
       'delta_J_px': '.6f', 'res_rms_px': '.6f', 'res_meannorm_px': '.6f',
       'res_coherent_px': '.6f', 'res_median_px': '.4f', 'res_p90_px': '.4f',
       'kp_err_median_px': '.4f'}


def cell(col, v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return format(v, FMT.get(col, '.6g')) if np.isfinite(v) else 'nan'
    return v


with open(os.path.join(RES_DIR, 'per_image.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(COLS)
    for r in ROWS:
        wr.writerow([cell(c, r[c]) for c in COLS])

# base-realization audit: the realized per-image RMS scale factors
Z_RMS = [float(np.sqrt(np.mean(np.linalg.norm(Z[j], axis=1) ** 2)))
         for j in range(N_IMG)]
json.dump(dict(gate_ok=True, n_rows=len(ROWS),
               base_z_checksum=float(Z.sum()),
               z_per_image_rms=Z_RMS,
               z_rms_median=float(np.median(Z_RMS)),
               d0_d1_placement=dict(d0_median_rms=D0_RMS_MED,
                                    d1_median_rms=D1_RMS_MED),
               runtime_s=round(time.time() - T0, 1)),
          open(os.path.join(RES_DIR, 'run_audit.json'), 'w'), indent=2)
log(f'base z per-image RMS: median {np.median(Z_RMS):.4f} '
    f'(range {min(Z_RMS):.4f}-{max(Z_RMS):.4f}); actual per-condition RMS = '
    f'sigma * z_rms per image (recorded, used as primary x-axis)')
log(f'DONE in {time.time() - T0:.1f}s (CPU-only; original PnP unmodified)')
