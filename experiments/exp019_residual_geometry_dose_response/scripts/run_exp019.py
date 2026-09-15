"""EXP019: Residual-Geometry Dose-Response (pure mechanism diagnosis).

Research question (EXP017/018 follow-up): EXP018 showed that on the clean D1
flip subset the WRONG ~180 deg pose fits the input keypoints STRICTLY BETTER
than the correct pose under the original reprojection objective (Type D /
image-space reprojection ambiguity, GO-C). EXP019 asks the stricter causal
question:

    is D1-like keypoint RESIDUAL GEOMETRY (anisotropy / coherent shift),
    at residual MAGNITUDE held comparable, causally SUFFICIENT to push the
    original PVNet reprojection objective into the ~180 deg ambiguity
    regime?

NOT the trivial "larger error -> more flips": every synthetic condition is
normalized to an exact residual RMS target, so magnitude is controlled and
only geometry varies.

Experiment boundary (protocol section 1/21, all asserted at runtime):
  - the exact 20 OCC held-out images of EXP017/018 (image_ids asserted);
  - GT 2D/3D keypoints, GT poses, linemod intrinsics, original 3D Farthest
    points -- bit-identical reuse from EXP016 raw npz (no re-sampling);
  - ORIGINAL Evaluator.evaluate -> pnp() -> cv2.solvePnP(SOLVEPNP_ITERATIVE),
    flags/termination/thresholds UNTOUCHED; no solvePnPGeneric, no new solver;
  - CPU-only, seed=0, no training, no GPU, no dataset expansion.
The ONLY manipulated variable: the 2D keypoint residual field fed to PnP
(kp_input = GT_kp + r). Images, masks, vertices, voting are never touched.

Pre-registration / anti-circularity (protocol section 22/23):
  1. Load EXP016 residuals; compute D1-like geometry targets from the REAL
     D1 residuals with numpy only (NO PnP involved).
     ANCHOR CORRECTION (recorded, made BEFORE any synthetic PnP ran): the
     first draft used POOLED RMS / pooled mean vector as the magnitude
     anchor; those are dominated by the two EXP018-confirmed pathological
     images color_00025/27 (residuals ~130-590 px -> pooled RMS 139.78 px,
     not representative of the typical D1 residual field). The run was
     aborted before ANY synthetic condition executed and before any number
     was produced or interpreted. Final anchors (robust to the 2/20
     pathological images):
       RMS_TARGET   = median over the 20 D1 images of per-image RMS
       MEANNORM_TGT = median over the 20 D1 images of per-image mean_norm
       SHIFT_TARGET = median over the 20 D1 images of per-image coherent
                      shift (== EXP017 residual_geometry.csv median 1.2445)
       ANISO_TARGET = median over the 20 D1 images of per-image covariance
                      anisotropy (== EXP017 csv median 3.5666)
       u            = unit direction of the mean of per-image residual mean
                      vectors over the fixed CLEAN18 (excl. 25?? no: excl.
                      color_00026/27 per EXP018 pathological set; 25 is
                      INCLUDED here -- it is catastrophic but its per-image
                      mean direction is not used because ... it IS excluded:
                      see below)
     NOTE on the direction anchor: CLEAN18 (fixed, result-independent,
     excludes color_00026/27) is used for u; color_00025 remains in the
     median-based targets (medians are robust to it).
  2. Write config.json (targets, scale/geometry/shift grids, normalization
     rule, RNG scheme, objective definition, decision thresholds, flags)
     BEFORE any PnP call. config.json is never modified afterwards.
  3. Baseline gate: C0 GT / C1 D0-real / C2 D1-real through the ORIGINAL
     pipeline must exactly reproduce EXP017 (flips + rot/reproj/trans) and
     EXP016 poses bit-exactly; C1/C2 objective fits must match EXP018's
     stored oracle values. Any mismatch -> STOP (exit 1).
  4. Only then run the synthetic grid.

Conditions (11; protocol section 9/10/11 -- grid NOT expanded):
  controls      C0_gt (r=0), C1_D0_real, C2_D1_real
  main grid     ISO_S05 / ISO_S10 / ISO_S15   (anisotropy=1,     shift=0)
                D1G_S05 / D1G_S10 / D1G_S15   (anisotropy=D1-like, shift=D1-like)
                (ISO_S10 == protocol C3 "isotropic matched";
                 D1G_S10 == protocol C4 "D1-like geometry")
  mechanism 2x2 ISO_S10_SH (isotropic + D1-like shift),
                D1G_S10_NS (D1-like anisotropy + zero shift)
                -- run only because cost is ~0; separates the two geometry
                axes at 1.0x. Scales: 0.5x / 1.0x / 1.5x of the real D1
                residual magnitude (1.0x matches the typical real D1 image).

Shared random realization (protocol section 7): ONE base Gaussian field
z ~ N(0,I), shape (20 images, 9 kps, 2), seed=0, generated ONCE; every
synthetic condition is a DETERMINISTIC transform of the same z. No
per-condition re-sampling.

Synthetic residual construction (pre-registered, per image):
  w_i = A z_i                       A = I (ISOTROPIC) or
                                    A = diag(sqrt(a_D1), 1) (D1-LIKE;
                                    principal axis fixed along image x)
  w_i <- w_i - mean_i(w_i)          remove sample mean (protocol ZERO rule)
  mu   = shift_level * scale * SHIFT_TARGET * u      (coherent shift)
  s    = sqrt(T^2 - ||mu||^2) / rms_i(w)             (T = scale * RMS_TARGET)
  r_i  = s * w_i + mu
  assert T > ||mu||
  => per-image RMS(r) == T EXACTLY (matched magnitude by construction) and
     per-image coherent shift == ||mu|| exactly; anisotropy of cov(r) == a
     up to sample noise. Achieved per-image geometry is recorded (protocol
     section 13); the mean_norm/RMS ratio is geometry-coupled and is
     REPORTED, not forced (RMS is the pre-registered magnitude statistic).

Objective oracle (EXP018 corrected sign convention, NOT re-inverted):
  J(pose) = mean_i || project_K(kp3d_Farthest9, pose)_i - kp_input_i || (px)
  J_pred  = fit of the returned pose; J_gt = fit of the GT pose
  delta_J = J_gt - J_pred; delta_J > 0 <=> J_pred < J_gt <=> the returned
  pose fits the input keypoints better than the correct pose (objective
  prefers the returned solution -- the REPROJECTION_AMBIGUITY direction).

Flip: rotation error > 90/120/150/170 deg (EXP016/017 definition). Primary
pre-registered indicator: >150 deg (the ~180 deg ambiguity regime).

Pathological-image policy (protocol section 20, fixed BEFORE results):
  CLEAN18 := all images except color_00026.png and color_00027.png (the two
  pathological images confirmed by EXP018; definition is result-independent).
  SOLVER_DIVERGENCE(image) := J_pred > 100 px (solver returned garbage;
  reported separately, never counted as ambiguity evidence).
  CATASTROPHIC_KEYPOINT(image) := median input keypoint error > 100 px
  (applies to real-residual conditions on 25/27).
  ALL 20 and CLEAN 18 are both always reported; nothing is deleted.

Run from repo root:
  python experiments/exp019_residual_geometry_dose_response/scripts/run_exp019.py
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

N_IMG, KP = 20, 9
SEED = 0
FLIP_THRESH = (90.0, 120.0, 150.0, 170.0)   # EXP016/017 definition
PRIMARY_FLIP = 150.0                        # ~180 deg ambiguity regime
SCALES = (0.5, 1.0, 1.5)
VOTE_TYPE = VotingType.Farthest
DIVERGENCE_J_PX = 100.0                     # SOLVER_DIVERGENCE flag threshold
CATASTROPHIC_KP_PX = 100.0                  # CATASTROPHIC_KEYPOINT flag threshold
PATHOLOGICAL = ('color_00026.png', 'color_00027.png')   # fixed (EXP018), CLEAN18 excludes
CLEAN_MASK = np.array([r not in PATHOLOGICAL for r in
                       ['color_%05d.png' % k for k in range(N_IMG, 2 * N_IMG)]])

# condition table: (name, group, scale, geometry, shift)  geometry: iso|d1g
COND_TABLE = [
    ('C0_gt',      'control', None, None, None),
    ('C1_D0_real', 'control', None, None, None),
    ('C2_D1_real', 'control', None, None, None),
    ('ISO_S05',    'grid',      0.5, 'iso', 0.0),
    ('ISO_S10',    'grid',      1.0, 'iso', 0.0),   # == protocol C3
    ('ISO_S15',    'grid',      1.5, 'iso', 0.0),
    ('D1G_S05',    'grid',      0.5, 'd1g', 1.0),   # shift 1.0 == D1-like
    ('D1G_S10',    'grid',      1.0, 'd1g', 1.0),   # == protocol C4
    ('D1G_S15',    'grid',      1.5, 'd1g', 1.0),
    ('ISO_S10_SH', 'mechanism', 1.0, 'iso', 1.0),   # 2x2: isolate shift
    ('D1G_S10_NS', 'mechanism', 1.0, 'd1g', 0.0),   # 2x2: isolate anisotropy
]
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


# ---------------- 1. image IDs: assert == EXP017 (and EXP018) ----------------
ids17 = json.load(open(os.path.join(EXP17_DIR, 'results', 'image_ids.json')))
OCC_RGB = list(ids17['occ_rgb'])
assert len(OCC_RGB) == N_IMG, 'EXP017 occ id count != 20 -> STOP'
ids18 = json.load(open(os.path.join(EXP18_DIR, 'image_ids.json')))
assert list(ids18['occ_rgb']) == OCC_RGB, 'EXP018 ids != EXP017 ids -> STOP'
CLEAN_MASK = np.array([r not in PATHOLOGICAL for r in OCC_RGB])
json.dump(dict(source='EXP017 results/image_ids.json (asserted identical; '
                    'EXP018 image_ids.json cross-checked identical)',
               image_ids=list(range(N_IMG, 2 * N_IMG)), occ_rgb=OCC_RGB),
          open(os.path.join(EXP_DIR, 'image_ids.json'), 'w'), indent=2)
log(f'image IDs asserted == EXP017 == EXP018: '
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

# ---------------- 3. D1-like geometry targets (numpy only, NO PnP) ----------
def per_image_geom(R):
    out = []
    for j in range(R.shape[0]):
        r = R[j]
        mu = r.mean(axis=0)
        c = np.cov(r.T, bias=True)
        ev = np.linalg.eigvalsh(c)
        n = np.linalg.norm(r, axis=1)
        out.append(dict(mean_vec=mu, coherent_shift=float(np.linalg.norm(mu)),
                        lam_min=float(ev[0]), lam_max=float(ev[1]),
                        anisotropy=float(ev[1] / ev[0]) if ev[0] > 1e-12 else float('inf'),
                        rms=float(np.sqrt(np.mean(n ** 2))),
                        mean_norm=float(n.mean())))
    return out


PER1 = per_image_geom(R1)
PER0 = per_image_geom(R0)
D1_ANISO = float(np.median([p['anisotropy'] for p in PER1]))
D1_SHIFT = float(np.median([p['coherent_shift'] for p in PER1]))
D1_RMS = float(np.median([p['rms'] for p in PER1]))
D1_MEANNORM = float(np.median([p['mean_norm'] for p in PER1]))
# pooled statistics REJECTED as anchor (documented correction): dominated by
# the 2 pathological images
POOL1_RMS = float(np.sqrt(np.mean(np.linalg.norm(R1.reshape(-1, 2), axis=1) ** 2)))
# coherent direction u: mean of per-image mean vectors over fixed CLEAN18
mu_clean = np.mean([PER1[j]['mean_vec'] for j in range(N_IMG) if CLEAN_MASK[j]],
                   axis=0)
U = mu_clean / np.linalg.norm(mu_clean)
log(f'D1-like targets (numpy only, BEFORE any PnP): aniso_median {D1_ANISO:.4f} | '
    f'shift_median {D1_SHIFT:.4f} px | RMS_anchor(median per-image) {D1_RMS:.4f} px '
    f'| mean_norm_anchor {D1_MEANNORM:.4f} px | u {U} '
    f'(from CLEAN18 mean-of-means {mu_clean})')
log(f'REJECTED pooled anchor (documented): pooled RMS {POOL1_RMS:.2f} px is '
    f'dominated by color_00025/27 catastrophic residuals')
log(f'D0 context: aniso_median '
    f'{float(np.median([p["anisotropy"] for p in PER0])):.4f} | shift_median '
    f'{float(np.median([p["coherent_shift"] for p in PER0])):.4f}')
with open(os.path.join(EXP17_DIR, 'results', 'residual_geometry.csv')) as f:
    g17 = [r for r in csv.DictReader(f) if r['arm'] == 'D1']
a17 = float(np.median([float(r['anisotropy_ratio']) for r in g17]))
s17 = float(np.median([float(r['coherent_shift_norm_px']) for r in g17]))
log(f'cross-check EXP017 residual_geometry.csv (D1 medians): aniso {a17:.4f} '
    f'(diff {abs(a17 - D1_ANISO):.2e}) | shift {s17:.4f} '
    f'(diff {abs(s17 - D1_SHIFT):.2e})')
assert abs(a17 - D1_ANISO) < 1e-3 and abs(s17 - D1_SHIFT) < 1e-3, \
    'EXP017 geometry cross-check mismatch -> STOP'
assert 1.0 < D1_ANISO < 100 and 0.1 < D1_SHIFT < D1_RMS < 10.0, \
    'implausible D1-like targets -> STOP'

# ---------------- 4. pre-registered config (BEFORE any PnP call) ------------
json.dump(dict(
    seed=SEED, n_img=N_IMG, kp_per_img=KP,
    image_ids=list(range(N_IMG, 2 * N_IMG)), occ_rgb=OCC_RGB,
    cls='cat', vote_type='VotingType.Farthest',
    pnp_path='lib.utils.evaluation_utils.Evaluator.evaluate -> pnp() -> '
             'cv2.solvePnP(SOLVEPNP_ITERATIVE); UNMODIFIED (no flags/threshold/'
             'termination change; no solvePnPGeneric; no new solver)',
    anchor_correction_note='first draft used POOLED RMS/mean-vector anchors; '
                           'REJECTED before any synthetic PnP ran (pooled RMS '
                           f'{POOL1_RMS:.2f} px dominated by color_00025/27); '
                           'final anchors are median per-image statistics, '
                           'robust to the 2/20 pathological images; no synthetic '
                           'result existed at correction time',
    objective=dict(
        definition='J(pose) = mean_i || project_K(kp3d_Farthest9, pose)_i - '
                   'kp_input_i || (px); original Projector, linemod intrinsics',
        J_pred='fit of solver-returned pose to input keypoints',
        J_gt='fit of GT pose to input keypoints',
        delta_J='J_gt - J_pred (EXP018 corrected sign); delta_J > 0 <=> J_pred < '
                'J_gt <=> returned pose fits input keypoints strictly better '
                'than the correct pose (REPROJECTION_AMBIGUITY direction); '
                'delta_J < 0 <=> correct pose fits better (SELECTION_FAILURE '
                'direction)'),
    flip=dict(thresholds_deg=list(FLIP_THRESH), primary_gt_deg=PRIMARY_FLIP,
              source='EXP016/017 definition reused'),
    scales=list(SCALES),
    anisotropy_levels=dict(ISOTROPIC=1.0, MODERATE='reserved (sqrt(D1_ANISO)); '
                           'NOT run in round 1 per protocol section 5/10',
                           D1_LIKE=D1_ANISO),
    coherent_shift_levels=dict(ZERO=0.0, D1_LIKE=D1_SHIFT),
    d1_like_targets=dict(anisotropy_median=D1_ANISO, coherent_shift_median=D1_SHIFT,
                         rms_anchor_median_per_image=D1_RMS,
                         meannorm_anchor_median_per_image=D1_MEANNORM,
                         unit_dir_u=U.tolist(),
                         u_source='mean of per-image residual mean vectors over '
                                  'fixed CLEAN18 (excl color_00026/27)',
                         rejected_pooled_rms=POOL1_RMS,
                         source='computed from EXP016 raw D1 residuals with numpy '
                                'ONLY before any PnP call (anti-circularity, '
                                'protocol section 22); cross-checked against '
                                'EXP017 residual_geometry.csv medians (assert '
                                '<1e-3)'),
    magnitude_anchor='1.0x scale == median per-image RMS of real D1 residuals; '
                     'per-image RMS(r) == scale * RMS_anchor EXACTLY by '
                     'construction (uniform dose across images)',
    normalization_rule='per image: w=A z; w <- w - mean(w); '
                       'mu = shift * scale * SHIFT_TARGET * u; '
                       's = sqrt(T^2 - ||mu||^2)/rms(w); r = s*w + mu; '
                       'T = scale*D1_RMS; assert T > ||mu||; kp = GT + r; '
                       'achieved per-image RMS exact, coherent shift exact, '
                       'anisotropy exact up to sample noise; mean_norm reported '
                       'not forced (geometry-coupled; RMS is the pre-registered '
                       'magnitude statistic)',
    shared_realization='ONE base z ~ N(0,I) (20,9,2), numpy default_rng(0), '
                       'generated ONCE after config write; every synthetic '
                       'condition is a deterministic transform of the same z '
                       '(protocol section 7; no per-condition re-sampling)',
    conditions=[dict(name=n, group=g, scale=s, geometry=ge, shift=sh)
                for n, g, s, ge, sh in COND_TABLE],
    condition_aliases=dict(C3_isotropic_matched='ISO_S10', C4_d1_like='D1G_S10'),
    pathological_policy=dict(
        clean18='ALL images except color_00026.png and color_00027.png '
                '(fixed from EXP018 before any EXP019 result; result-independent)',
        solver_divergence=f'J_pred > {DIVERGENCE_J_PX:.0f} px -> SOLVER_DIVERGENCE '
                          'flag; reported separately, never counted as ambiguity',
        catastrophic_keypoint=f'median input keypoint error > '
                              f'{CATASTROPHIC_KP_PX:.0f} px -> '
                              'CATASTROPHIC_KEYPOINT flag (real-residual conds)'),
    gate=dict(
        c0='GT keypoints: flips 0/20 at all thresholds, max rot < 1e-3 deg, '
           'max official reproj < 1e-2 px',
        c1_c2='D0/D1 real keypoints: flips exact vs EXP017 (4/20, 14/20 at >90deg) '
              '+ rot<1e-3/reproj<1e-3/trans<1e-2 + poses bit-match EXP016 npz '
              '(<1e-9) + J_pred/J_gt match EXP018 objective_oracle.csv (<1e-4)',
        on_failure='STOP (exit 1), no synthetic run, no interpretation'),
    decision_rules_pre_registered=dict(
        primary='flip rot>150deg on ALL 20; CLEAN 18 reported as required support '
                '(decision invalidated to GO-C if CLEAN 18 overall gap sign '
                'contradicts ALL 20)',
        GO_A='geometry-causal (protocol section 16 symmetric form; direction '
             'reported): g_k = flip150(D1G,k) - flip150(ISO,k) for k in '
             '{0.5,1.0,1.5}: (i) all g_k >= 0 with sum >= 3 (D1-promoting) OR all '
             'g_k <= 0 with -sum >= 3 (isotropy-promoting); (ii) RA-type evidence: '
             'RA(D1G,k) >= RA(ISO,k) at >= 2/3 scales in the promoted direction, '
             'RA := flip150 AND delta_J > 0; (iii) cross-scale punchline: '
             'flip150(promoted geometry @1.0x) >= flip150(other geometry @1.5x). '
             'Conclusion: residual geometry, not magnitude alone, is causally '
             'sufficient to move the objective toward the ~180 deg basin; the '
             'promoted direction is reported explicitly (D1-promoting matches '
             'protocol section 15 wording; isotropy-promoting matches EXP017 '
             'within-D1 association and is still geometry-causal per section 16)',
        GO_B='magnitude-only: |g_k| <= 1 at every scale AND flip150 rises by >= 3 '
             'from 0.5x to 1.5x within at least one geometry family; STOP, no '
             'geometry mining',
        GO_C='mixed/inconclusive otherwise; conservative wording only'),
    forbidden='no solver/flag/threshold/seed/GT/intrinsics/3D-points/image-set '
              'change; no level added post hoc; D1-like targets frozen above '
              'BEFORE any PnP (protocol section 21/22)',
    language_limits='cat-only, 20-image mechanism diagnosis; association->causal '
                    'evidence on synthetic residuals only; no significance claims',
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
rng = np.random.default_rng(SEED)
Z = rng.standard_normal((N_IMG, KP, 2))
log(f'base realization Z generated once: shape {Z.shape}, seed {SEED}, '
    f'checksum {float(Z.sum()):.6f}')

# ---------------- 7. all conditions ------------------------------------------
def fmtcol(col, v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return f'{v:.6f}' if not np.isfinite(v) else format(v, '.6f')
    return v


ROWS = []
for cond, grp, scale, geo, shift in COND_TABLE:
    if grp == 'control':
        KP_IN = {'C0_gt': GT_KPS,
                 'C1_D0_real': KP_REAL['C1_D0_real'],
                 'C2_D1_real': KP_REAL['C2_D1_real']}[cond]
        R_FIELD = {'C0_gt': np.zeros_like(GT_KPS),
                   'C1_D0_real': R0, 'C2_D1_real': R1}[cond]
    else:
        a = 1.0 if geo == 'iso' else D1_ANISO
        A = np.diag([np.sqrt(a), 1.0])
        KP_IN = np.empty_like(GT_KPS)
        R_FIELD = np.empty_like(GT_KPS)
        for j in range(N_IMG):
            w = Z[j] @ A.T
            w = w - w.mean(axis=0, keepdims=True)
            mu = shift * scale * D1_SHIFT * U
            T = scale * D1_RMS
            assert T > np.linalg.norm(mu) + 1e-12, \
                f'{cond}: T <= ||mu|| (scale {scale}) -> STOP'
            rms_w = float(np.sqrt(np.mean(np.linalg.norm(w, axis=1) ** 2)))
            s = np.sqrt(T ** 2 - float(np.linalg.norm(mu) ** 2)) / rms_w
            r = s * w + mu
            R_FIELD[j] = r
            KP_IN[j] = GT_KPS[j] + r
    for i in range(N_IMG):
        o = run_pnp(KP_IN[i], i)
        r = R_FIELD[i]
        rn = np.linalg.norm(r, axis=1)
        mu_i = r.mean(axis=0)
        c = np.cov(r.T, bias=True)
        evv = np.linalg.eigvalsh(c)
        kp_err = np.linalg.norm(KP_IN[i] - GT_KPS[i], axis=1)
        div = bool(o.get('J_pred', 0.0) > DIVERGENCE_J_PX
                   or o.get('reproj', 0.0) > DIVERGENCE_J_PX)
        cat = bool(np.median(kp_err) > CATASTROPHIC_KP_PX)
        rot = o.get('rot', float('nan'))
        ROWS.append(dict(
            condition=cond, group=grp, scale=scale,
            anisotropy_level=('n/a' if geo is None else
                              ('ISOTROPIC' if geo == 'iso' else 'D1_LIKE')),
            shift_level=('n/a' if shift is None else
                         ('ZERO' if shift == 0 else 'D1_LIKE')),
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
            res_coherent_px=float(np.linalg.norm(mu_i)),
            res_lambda_max=float(evv[1]), res_lambda_min=float(evv[0]),
            res_anisotropy=float(evv[1] / evv[0]) if evv[0] > 1e-12 else float('nan'),
            res_max_px=float(rn.max()), res_median_px=float(np.median(rn)),
            kp_err_median_px=float(np.median(kp_err)),
            flag_solver_divergence=div, flag_catastrophic_kp=cat,
            flag_pnp_exception=bool(o['exception']),
            in_clean18=bool(CLEAN_MASK[i])))
    log(f'--- {cond}: 20 images done ({time.time() - T0:.1f}s) ---')

# ---------------- 8. persist --------------------------------------------------
COLS = ['condition', 'group', 'scale', 'anisotropy_level', 'shift_level',
        'image_id', 'rgb', 'rot_err_deg', 'trans_err_mm', 'add_mm',
        'reproj_official_px', 'flip_gt_90', 'flip_gt_120', 'flip_gt_150',
        'flip_gt_170', 'J_gt_px', 'J_pred_px', 'delta_J_px',
        'res_rms_px', 'res_meannorm_px', 'res_coherent_px', 'res_lambda_max',
        'res_lambda_min', 'res_anisotropy', 'res_max_px', 'res_median_px',
        'kp_err_median_px', 'flag_solver_divergence', 'flag_catastrophic_kp',
        'flag_pnp_exception', 'in_clean18']
FMT = {'rot_err_deg': '.4f', 'trans_err_mm': '.2f', 'add_mm': '.4f',
       'reproj_official_px': '.4f', 'J_gt_px': '.6f', 'J_pred_px': '.6f',
       'delta_J_px': '.6f', 'res_rms_px': '.6f', 'res_meannorm_px': '.6f',
       'res_coherent_px': '.6f', 'res_lambda_max': '.6f',
       'res_lambda_min': '.6f', 'res_anisotropy': '.4f', 'res_max_px': '.4f',
       'res_median_px': '.4f', 'kp_err_median_px': '.4f'}


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

# achieved-geometry audit: synthetic RMS must equal target exactly
dev = [abs(r['res_rms_px'] - r['scale'] * D1_RMS)
       for r in ROWS if r['group'] != 'control']
json.dump(dict(gate_ok=True, n_rows=len(ROWS),
               synthetic_rms_max_abs_dev_px=float(max(dev)),
               d1_targets=dict(aniso=D1_ANISO, shift=D1_SHIFT, rms=D1_RMS,
                               mean_norm=D1_MEANNORM, u=U.tolist()),
               rejected_pooled_rms=POOL1_RMS,
               runtime_s=round(time.time() - T0, 1)),
          open(os.path.join(RES_DIR, 'run_audit.json'), 'w'), indent=2)
log(f'synthetic per-image RMS max |dev| vs target: {max(dev):.2e} px '
    '(matched magnitude by construction)')
log(f'DONE in {time.time() - T0:.1f}s (CPU-only; original PnP unmodified)')
