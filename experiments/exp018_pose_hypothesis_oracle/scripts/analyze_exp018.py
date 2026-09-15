"""EXP018 analysis: per-image diagnosis, D0 contrast, residual-geometry
validation, pre-registered decision.

Reads ONLY already-produced artifacts (no model, no solver, no GPU):
  results/objective_oracle.csv          (run_exp018.py)
  results/reproduction_check.json       (run_exp018.py)
  EXP017 results/residual_geometry.csv  (read-only re-validation, section 8)

Writes:
  results/per_image_diagnosis.csv       (protocol section 9 table)
  results/residual_geometry_validation.json
  results/summary.json, results/decision.json

Diagnosis categories (protocol section 9, EXACTLY these five; none added):
  NORMAL                  rot < 90 deg (solver selected the correct basin)
  SELECTION_FAILURE       flip AND J_gt < J_pred - tie_eps (correct pose fits
                          the input keypoints strictly better; solver left it
                          unfound) -> CASE A evidence
  NO_CORRECT_HYPOTHESIS   NOT assignable without the candidate space (count 0,
                          caveat recorded)
  REPROJECTION_AMBIGUITY  flip AND J_pred < J_gt - tie_eps (the ~180 deg pose
                          fits the input keypoints strictly better than the
                          correct pose; Type D) -> CASE C evidence; also the
                          strongest available evidence for the CASE B
                          geometric reading (objective-based selection would
                          also choose wrong)
  UNRESOLVED              flip AND |J_gt - J_pred| <= tie_eps
Run from repo root:
  python experiments/exp018_pose_hypothesis_oracle/scripts/analyze_exp018.py
"""
import os, sys, json, csv
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(EXP_DIR, 'results')
EXP17_DIR = os.path.join(ROOT, 'experiments', 'exp017_pose_hypothesis_landscape')

N_IMG = 20
TIE_EPS = 1e-3          # must match run_exp018.py pre-registration
AMBIG_ROT = 150.0       # ~180 deg qualifier


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


with open(os.path.join(RES_DIR, 'objective_oracle.csv')) as f:
    ORA = {}
    for row in csv.DictReader(f):
        ORA.setdefault(row['condition'], []).append(row)
for c in ORA:
    assert len(ORA[c]) == N_IMG, f'{c}: {len(ORA[c])} rows != 20'
CHECK = json.load(open(os.path.join(RES_DIR, 'reproduction_check.json')))


def med(vals):
    v = np.asarray(vals, np.float64)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float('nan')


# ---------------- per-image diagnosis (D1 primary, D0 contrast) ----------------
def diagnose(cond):
    out = []
    for r in ORA[cond]:
        rot = fnum(r['rotation_error_deg'])
        d = fnum(r['delta_px'])
        flipped = r['flip_gt_90'] == '1'
        if not flipped:
            diag = 'NORMAL'
        elif not np.isfinite(d):
            diag = 'UNRESOLVED'
        elif d < -TIE_EPS:  # J_gt < J_pred: correct pose fits strictly better,
                            # solver left it unfound -> selection failure
            diag = 'SELECTION_FAILURE'
        elif d > TIE_EPS:   # J_pred < J_gt: wrong ~180deg pose fits strictly
                            # better -> reprojection ambiguity (Type D)
            diag = 'REPROJECTION_AMBIGUITY'
        else:
            diag = 'UNRESOLVED'
        out.append(dict(
            image_id=int(r['image_id']), rgb=r['rgb'], rot=rot,
            flip=flipped, flip150=r['flip_gt_150'] == '1',
            J_pred=fnum(r['J_pred_px']), J_gt=fnum(r['J_gt_px']), delta=d,
            diagnosis=diag))
    return out


D0, D1 = diagnose('C1_D0'), diagnose('C2_D1')
for a, b in zip(D0, D1):
    assert a['image_id'] == b['image_id']

counts = {}
for name, rows in (('D0', D0), ('D1', D1)):
    counts[name] = {k: sum(r['diagnosis'] == k for r in rows)
                    for k in ('NORMAL', 'SELECTION_FAILURE', 'NO_CORRECT_HYPOTHESIS',
                              'REPROJECTION_AMBIGUITY', 'UNRESOLVED')}
    counts[name]['REPROJECTION_AMBIGUITY_rot_gt150'] = sum(
        r['diagnosis'] == 'REPROJECTION_AMBIGUITY' and r['flip150'] for r in rows)
    fl = [r for r in rows if r['flip']]
    counts[name]['n_flip'] = len(fl)
    counts[name]['mean_delta_flip_px'] = float(np.mean([r['delta'] for r in fl])) if fl else None
    counts[name]['median_delta_flip_px'] = med([r['delta'] for r in fl]) if fl else None

# ---------------- per_image_diagnosis.csv (protocol section 9 + extras) ----------------
with open(os.path.join(RES_DIR, 'per_image_diagnosis.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['image', 'D0_flip', 'D1_flip',
                 'D0_reproj', 'D1_reproj', 'D0_rot', 'D1_rot',
                 'D1_correct_hyp_exists', 'D1_best_correct_reproj', 'D1_best_wrong_reproj',
                 'D1_flip_gt_150', 'D1_delta_px', 'diagnosis',
                 'D0_J_gt_px', 'D0_J_pred_px', 'D0_delta_px', 'D0_diagnosis'])
    for a, b in zip(D0, D1):
        r1 = ORA['C2_D1'][b['image_id'] - N_IMG]
        r0 = ORA['C1_D0'][a['image_id'] - N_IMG]
        # correct hypothesis (rot<10 deg) exists among OBSERVABLE poses (pose_gt,
        # rot 0 by construction) on every image -> True; full candidate space NOT
        # observable (documented limitation, applies to every row)
        wr.writerow([a['rgb'], int(a['flip']), int(b['flip']),
                     r0['mean_reprojection_error_px'], r1['mean_reprojection_error_px'],
                     f"{a['rot']:.3f}", f"{b['rot']:.3f}",
                     'True', f"{b['J_gt']:.6f}", f"{b['J_pred']:.6f}",
                     int(b['flip150']), f"{b['delta']:.6f}", b['diagnosis'],
                     f"{a['J_gt']:.6f}", f"{a['J_pred']:.6f}", f"{a['delta']:.6f}",
                     a['diagnosis']])

# ---------------- EXP017 residual geometry re-validation (read-only) ----------------
with open(os.path.join(EXP17_DIR, 'results', 'residual_geometry.csv')) as f:
    GEO = [r for r in csv.DictReader(f)]
d1 = [r for r in GEO if r['arm'] == 'D1']
g = {}
for name, flag in (('flip', '1'), ('stable', '0')):
    rs = [r for r in d1 if r['flip_gt_90'] == flag]
    g[name] = dict(n=len(rs),
                   kp_median_px=med([fnum(r['kp_median_px']) for r in rs]),
                   coherent_shift_norm_px=med([fnum(r['coherent_shift_norm_px']) for r in rs]),
                   relative_scatter_mean_px=med([fnum(r['relative_scatter_mean_px']) for r in rs]),
                   anisotropy_ratio=med([fnum(r['anisotropy_ratio']) for r in rs]))
geom_validation = dict(
    source='EXP017 results/residual_geometry.csv (read-only)',
    d1_flip_vs_stable=g,
    checks=dict(
        flip_anisotropy_lower=bool(g['flip']['anisotropy_ratio'] < g['stable']['anisotropy_ratio']),
        flip_coherent_shift_smaller=bool(g['flip']['coherent_shift_norm_px']
                                         < g['stable']['coherent_shift_norm_px']),
        magnitude_not_explanatory=bool(g['flip']['kp_median_px']
                                       <= 1.5 * g['stable']['kp_median_px'])))
json.dump(geom_validation, open(os.path.join(RES_DIR, 'residual_geometry_validation.json'), 'w'),
          indent=2)

# ---------------- pre-registered decision ----------------
if not CHECK['reproduction_ok']:
    decision, rationale = 'STOP', \
        'C0/C1/C2 do not reproduce EXP017 exactly -> per protocol: report mismatch, STOP.'
else:
    sf = counts['D1']['SELECTION_FAILURE']
    ra = counts['D1']['REPROJECTION_AMBIGUITY_rot_gt150']
    un = counts['D1']['UNRESOLVED']
    top = max(sf, ra, un)
    n_flip = counts['D1']['n_flip']
    if [sf, ra, un].count(top) >= 2:
        decision, rationale = 'STOP', \
            f'Exact plurality tie among outcomes (SF {sf} / RA {ra} / UNRESOLVED {un} ' \
            f'of {n_flip} D1 flips) -> no single conclusion; report and STOP.'
    elif top == sf and sf > 0:
        decision, rationale = 'GO-A', \
            f'SELECTION_FAILURE plurality ({sf}/{n_flip} D1 flips): the correct pose fits ' \
            'the input keypoints strictly better than the returned ~180deg pose on the ' \
            'majority of flip images -> the original PnP left the better-fitting correct ' \
            'hypothesis unfound (hypothesis selection failure).'
    elif top == ra and ra > 0:
        decision, rationale = 'GO-C', \
            f'REPROJECTION_AMBIGUITY plurality ({ra}/{n_flip} D1 flips, wrong rot>150deg): ' \
            'the ~180deg pose fits the input keypoints strictly better than the correct ' \
            'pose -> the original reprojection objective itself is pose-ambiguous for the ' \
            'D1 keypoint configuration. NOTE: without candidate-space access this is also ' \
            'the strongest available evidence for the CASE B geometric reading (a perfect ' \
            'objective-based selector would also choose wrong); GO-B strict condition is ' \
            'not verifiable here (documented limitation).'
    else:
        decision, rationale = 'STOP', \
            f'UNRESOLVED plurality (SF {sf} / RA {ra} / UNRESOLVED {un} of {n_flip} D1 ' \
            'flips) -> near-tie objective values dominate; no conclusion.'

DEC = dict(decision=decision, rationale=rationale,
           reproduction=dict(ok=bool(CHECK['reproduction_ok']),
                             pose_bitmatch_npz=CHECK['pose_bitmatch_npz']),
           diagnosis_counts=counts,
           candidate_accessibility='NOT exposed by current implementation -> partial '
                                   'evidence mode (objective-level oracle over the two '
                                   'observable poses)',
           residual_geometry_validation=geom_validation,
           rules='pre-registered in config.json')
json.dump(DEC, open(os.path.join(RES_DIR, 'decision.json'), 'w'), indent=2)

summary = dict(decision=decision, rationale=rationale,
               diagnosis_counts=counts,
               d1_flip_images=[r['rgb'] for r in D1 if r['flip']],
               d1_flip_detail=[dict(rgb=r['rgb'], rot=r['rot'], flip150=r['flip150'],
                                    J_pred=r['J_pred'], J_gt=r['J_gt'],
                                    delta=r['delta'], diagnosis=r['diagnosis'])
                               for r in D1 if r['flip']],
               d0_flip_detail=[dict(rgb=r['rgb'], rot=r['rot'],
                                    J_pred=r['J_pred'], J_gt=r['J_gt'],
                                    delta=r['delta'], diagnosis=r['diagnosis'])
                               for r in D0 if r['flip']],
               residual_geometry_validation=geom_validation)
json.dump(summary, open(os.path.join(RES_DIR, 'summary.json'), 'w'), indent=2)

print('=== EXP018 diagnosis counts ===')
for name in ('D0', 'D1'):
    c = counts[name]
    print(f"{name}: flip {c['n_flip']}/20 | NORMAL {c['NORMAL']} | "
          f"SELECTION_FAILURE {c['SELECTION_FAILURE']} | NO_CORRECT_HYPOTHESIS "
          f"{c['NO_CORRECT_HYPOTHESIS']} (not assignable, no candidate space) | "
          f"REPROJECTION_AMBIGUITY {c['REPROJECTION_AMBIGUITY']} (rot>150: "
          f"{c['REPROJECTION_AMBIGUITY_rot_gt150']}) | UNRESOLVED {c['UNRESOLVED']} | "
          f"mean delta(flip) {c['mean_delta_flip_px']}")
print('=== D1 flip per-image ===')
for r in D1:
    if r['flip']:
        print(f"  {r['rgb']}  rot {r['rot']:7.2f}  J_pred {r['J_pred']:.3f}  "
              f"J_gt {r['J_gt']:.3f}  delta {r['delta']:+.3f}  {r['diagnosis']}")
print('=== D0 flip per-image ===')
for r in D0:
    if r['flip']:
        print(f"  {r['rgb']}  rot {r['rot']:7.2f}  J_pred {r['J_pred']:.3f}  "
              f"J_gt {r['J_gt']:.3f}  delta {r['delta']:+.3f}  {r['diagnosis']}")
print('=== EXP017 geometry re-validation ===')
print(f"  flip anisotropy lower: {geom_validation['checks']['flip_anisotropy_lower']} | "
      f"flip coherent shift smaller: {geom_validation['checks']['flip_coherent_shift_smaller']} | "
      f"magnitude not explanatory: {geom_validation['checks']['magnitude_not_explanatory']}")
print(f"DECISION: {decision}")
print(rationale)
