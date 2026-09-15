"""EXP017 analysis: residual geometry, flip/stable grouping, pre-registered decision.

Reads ONLY already-produced artifacts (no model, no GPU):
  results/per_image.csv            (run_exp017.py, 5 conditions x 20 images)
  results/reproduction_check.json  (run_exp017.py, Check A/B/C)
  results/noise_statistics.json    (run_exp017.py)
  EXP016 raw_occ_D{0,1}.npz        (bit-identical keypoint/GT source, read-only)

Writes:
  results/summary.csv              per-condition aggregates + flip counts
  results/residual_geometry.csv    per-image D0/D1 keypoint residual geometry
  results/decision.json            pre-registered CASE A/B/C/D evaluation

Residual geometry (per image, arm D0/D1, residual = predicted_kp - GT_kp, px):
  mean_dx/mean_dy                  residual mean
  std_dx/std_dy                    residual spread (ddof=1)
  mean_residual_norm               mean ||r_i||
  cov_xx/cov_xy/cov_yy             2x2 residual covariance (ddof=1)
  eig_max/eig_min, anisotropy_ratio=lambda_max/max(lambda_min,1e-9)
  coherent_shift_norm=||r_bar||    coherent displacement of ALL keypoints
  relative_scatter_mean/p90        de-meaned scatter ||r_i - r_bar||
    (separates "all KP translate together" from "structural deformation
     between keypoints")
Run from repo root:
  python experiments/exp017_pose_hypothesis_landscape/analyze_exp017.py
"""
import os, sys, json, csv
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(EXP_DIR, 'results')
EXP16_DIR = os.path.join(ROOT, 'experiments', 'exp016_pnp_stability_diagnosis')

N_IMG, KP = 20, 9
COND_NAMES = ('C0_gt', 'C1_D0', 'C2_D1', 'C3_D0_noise', 'C4_D1_noise')
GEOM_EPS = 1e-9
GEOM_METRICS = ('relative_scatter_mean_px', 'coherent_shift_norm_px', 'anisotropy_ratio')
RATIO_SEP = 1.5           # pre-registered separation ratio (or its inverse)
MAG_RATIO_MAX = 1.5       # pre-registered: flip-group kp_median <= 1.5x stable-group

with open(os.path.join(RES_DIR, 'per_image.csv')) as f:
    PER = {c: [] for c in COND_NAMES}
    for row in csv.DictReader(f):
        PER[row['condition']].append(row)
for c in COND_NAMES:
    assert len(PER[c]) == N_IMG, f'{c}: {len(PER[c])} rows != 20'
ids = json.load(open(os.path.join(RES_DIR, 'image_ids.json')))
OCC_RGB = ids['occ_rgb']
assert all(r['rgb'] == OCC_RGB[i] for c in COND_NAMES for i, r in enumerate(PER[c]))

CHECK = json.load(open(os.path.join(RES_DIR, 'reproduction_check.json')))
NOISE = json.load(open(os.path.join(RES_DIR, 'noise_statistics.json')))

z0 = np.load(os.path.join(EXP16_DIR, 'results', 'raw_predictions', 'raw_occ_D0.npz'))
z1 = np.load(os.path.join(EXP16_DIR, 'results', 'raw_predictions', 'raw_occ_D1.npz'))
GT_KPS = z0['gt_kps'].astype(np.float64)


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def med(vals):
    v = np.asarray(vals, np.float64)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float('nan')


# ---------------- summary.csv (per condition) ----------------
with open(os.path.join(RES_DIR, 'summary.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['condition', 'n_images', 'pnp_fail',
                 'kp_mean_of_image_means_px', 'kp_median_of_image_medians_px', 'kp_p90_median_px',
                 'rot_median_deg', 'rot_mean_deg',
                 'trans_median_mm', 'add_mm_median', 'add_pass_rate',
                 'reproj_mean_median_px', 'reproj_median_median_px',
                 'flip_gt_90', 'flip_gt_120', 'flip_gt_150', 'flip_gt_170'])
    for c in COND_NAMES:
        rs = PER[c]
        kp_means = [fnum(r['keypoint_mean_error_px']) for r in rs]
        kp_meds = [fnum(r['keypoint_median_error_px']) for r in rs]
        rots = [fnum(r['rotation_error_deg']) for r in rs]
        wr.writerow([c, len(rs), sum(1 for r in rs if r['pnp_ok'] == '0'),
                     f"{med(kp_means):.4f}",
                     f"{med(kp_meds):.4f}",
                     f"{med([fnum(r['keypoint_p90_error_px']) for r in rs]):.4f}",
                     f"{med(rots):.3f}",
                     f"{np.nanmean(rots):.3f}",
                     f"{med([fnum(r['translation_error_mm']) for r in rs]):.2f}",
                     f"{med([fnum(r['add_mm']) for r in rs]):.4f}",
                     f"{np.mean([r['add_pass'] == '1' for r in rs]):.3f}",
                     f"{med([fnum(r['mean_reprojection_error_px']) for r in rs]):.4f}",
                     f"{med([fnum(r['median_reprojection_error_px']) for r in rs]):.4f}",
                     *[sum(r[f'flip_gt_{int(t)}'] == '1' for r in rs) for t in (90, 120, 150, 170)]])

# ---------------- residual_geometry.csv (D0/D1, clean conditions) ----------------
geom_rows = []
for arm, zp in (('D0', z0), ('D1', z1)):
    pred = zp['pred_kps'].astype(np.float64)
    cond = 'C1_D0' if arm == 'D0' else 'C2_D1'
    for i in range(N_IMG):
        R = pred[i] - GT_KPS[i]                              # (9,2)
        r_bar = R.mean(axis=0)
        rel = R - r_bar
        rel_norms = np.linalg.norm(rel, axis=1)
        cov = np.cov(R, rowvar=False, ddof=1)                # 2x2
        eigvals = np.linalg.eigvalsh(cov)                    # ascending
        lam_max, lam_min = float(eigvals[1]), float(eigvals[0])
        err = np.linalg.norm(R, axis=1)
        row = PER[cond][i]
        geom_rows.append(dict(
            arm=arm, image_id=N_IMG + i, rgb=OCC_RGB[i],
            mean_dx=float(r_bar[0]), mean_dy=float(r_bar[1]),
            std_dx=float(R[:, 0].std(ddof=1)), std_dy=float(R[:, 1].std(ddof=1)),
            mean_residual_norm_px=float(err.mean()),
            cov_xx=float(cov[0, 0]), cov_xy=float(cov[0, 1]), cov_yy=float(cov[1, 1]),
            eig_max=lam_max, eig_min=lam_min,
            anisotropy_ratio=lam_max / max(lam_min, GEOM_EPS),
            coherent_shift_norm_px=float(np.linalg.norm(r_bar)),
            relative_scatter_mean_px=float(rel_norms.mean()),
            relative_scatter_p90_px=float(np.quantile(rel_norms, 0.9)),
            flip_gt_90=int(row['flip_gt_90']),
            kp_mean_px=fnum(row['keypoint_mean_error_px']),
            kp_median_px=fnum(row['keypoint_median_error_px']),
            mean_reprojection_error_px=fnum(row['mean_reprojection_error_px']),
            rotation_error_deg=fnum(row['rotation_error_deg'])))

with open(os.path.join(RES_DIR, 'residual_geometry.csv'), 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=list(geom_rows[0].keys()))
    wr.writeheader()
    for r in geom_rows:
        wr.writerow({k: (f'{v:.6f}' if isinstance(v, float) else v) for k, v in r.items()})

# ---------------- D1 flip vs stable grouping (descriptive medians) ----------------
def group_table(arm, cond):
    rows = [g for g in geom_rows if g['arm'] == arm]
    out = {}
    for gname in ('flip', 'stable'):
        gr = [r for r in rows if r['flip_gt_90'] == (1 if gname == 'flip' else 0)]
        out[gname] = dict(n=len(gr)) if gr else dict(n=0)
        if gr:
            out[gname].update(
                kp_mean_px=med([r['kp_mean_px'] for r in gr]),
                kp_median_px=med([r['kp_median_px'] for r in gr]),
                coherent_shift_norm_px=med([r['coherent_shift_norm_px'] for r in gr]),
                relative_scatter_mean_px=med([r['relative_scatter_mean_px'] for r in gr]),
                anisotropy_ratio=med([r['anisotropy_ratio'] for r in gr]),
                mean_reprojection_error_px=med([r['mean_reprojection_error_px'] for r in gr]),
                rotation_error_deg=med([r['rotation_error_deg'] for r in gr]))
    return out


def ratios(flip_v, stab_v):
    if not np.isfinite(flip_v) or not np.isfinite(stab_v) or abs(stab_v) < GEOM_EPS:
        return None
    return float(flip_v / stab_v)


G_D1 = group_table('D1', 'C2_D1')
geom_sep = {}
for m in GEOM_METRICS:
    if G_D1['flip'].get('n', 0) and G_D1['stable'].get('n', 0):
        geom_sep[m] = ratios(G_D1['flip'][m], G_D1['stable'][m])
    else:
        geom_sep[m] = None
mag_ratio = (G_D1['flip']['kp_median_px'] / G_D1['stable']['kp_median_px']
             if G_D1['flip'].get('n', 0) and G_D1['stable'].get('n', 0)
             and G_D1['stable']['kp_median_px'] > GEOM_EPS else None)

geom_d0d1 = {m: ratios(med([r[m] for r in geom_rows if r['arm'] == 'D1']),
                       med([r[m] for r in geom_rows if r['arm'] == 'D0']))
             for m in GEOM_METRICS}

# ---------------- pre-registered decision (priority A -> C -> B -> D) ----------------
def flips(cond, t=90):
    return sum(r[f'flip_gt_{t}'] == '1' for r in PER[cond])


c0f, c1f, c2f, c3f, c4f = (flips(c) for c in COND_NAMES)
rep_ok = CHECK['check_B']['ok'] and CHECK['check_C']['ok']
intrinsic_suspect = CHECK['check_A']['intrinsic_ambiguity_suspected']

if not rep_ok:
    decision, rationale = 'STOP_REPRODUCTION_MISMATCH', \
        'Check B/C failed: C1/C2 do not reproduce EXP016 (or noise identity broken). ' \
        'Per spec: STOP, report mismatch, do not interpret C3/C4.'
elif c0f >= 3:
    decision, rationale = 'CASE_A', \
        f'C0 GT keypoints already flip on {c0f}/20 images -> pose ambiguity exists even ' \
        'near zero keypoint error (intrinsic). Solver design FORBIDDEN afterwards.'
elif c0f <= 1 and (c3f >= 10 or c4f >= 10) and abs(c3f - c4f) <= 4:
    decision, rationale = 'CASE_C', \
        f'C0 stable ({c0f}/20) but matched 1px perturbation produces similar large flip ' \
        f'counts (C3 {c3f}/20, C4 {c4f}/20, gap {abs(c3f - c4f)}): the original PnP is highly ' \
        'sensitive to small keypoint perturbations; the D0/D1 difference largely disappears.'
elif c0f <= 1 and c1f <= 6 and c2f >= 10 and \
        (mag_ratio is not None and mag_ratio <= MAG_RATIO_MAX) and \
        any(r is not None and (r >= RATIO_SEP or r <= 1.0 / RATIO_SEP) for r in geom_sep.values()):
    decision, rationale = 'CASE_B', \
        f'C0 stable ({c0f}/20); C1 {c1f}/20 vs C2 {c2f}/20 reproduced; flip not explained by ' \
        f'KP magnitude (flip/stable kp_median ratio {mag_ratio:.2f}); flip is associated with ' \
        'keypoint residual geometry (see ratios). Association language ONLY.'
else:
    decision, rationale = 'INCONCLUSIVE', \
        'None of the pre-registered patterns met; no over-interpretation, no follow-up.'
if decision == 'CASE_A' and intrinsic_suspect:
    rationale += ' INTRINSIC_AMBIGUITY_SUSPECTED flag raised (C0 rot > 170 deg present).'

DEC = dict(
    flip_counts_gt90={c: flips(c) for c in COND_NAMES},
    flip_counts_all={c: {f'gt{t}': flips(c, t) for t in (90, 120, 150, 170)} for c in COND_NAMES},
    reproduction=dict(check_A_intrinsic_suspect=bool(intrinsic_suspect),
                      check_B_ok=bool(CHECK['check_B']['ok']),
                      check_C_ok=bool(CHECK['check_C']['ok'])),
    noise=NOISE,
    d1_flip_vs_stable=G_D1,
    d1_geom_separation_ratios_flip_over_stable=geom_sep,
    d0_vs_d1_geom_median_ratios=geom_d0d1,
    d1_mag_ratio_flip_over_stable_kp_median=mag_ratio,
    decision=decision, rationale=rationale,
    rules='pre-registered in results/config.json (priority A -> C -> B -> D)')
json.dump(DEC, open(os.path.join(RES_DIR, 'decision.json'), 'w'), indent=2)

print('=== EXP017 summary ===')
for c in COND_NAMES:
    fc = DEC['flip_counts_all'][c]
    print(f"{c:13s} flips: >90 {fc['gt90']:2d}/20  >120 {fc['gt120']:2d}  "
          f">150 {fc['gt150']:2d}  >170 {fc['gt170']:2d}")
print(f"Check A intrinsic_suspect={intrinsic_suspect}  Check B ok={CHECK['check_B']['ok']}  "
      f"Check C ok={CHECK['check_C']['ok']}")
print(f"D1 flip/stable group medians: "
      f"kp_median {G_D1['flip'].get('kp_median_px', float('nan')):.3f} vs "
      f"{G_D1['stable'].get('kp_median_px', float('nan')):.3f} px | "
      f"coherent {G_D1['flip'].get('coherent_shift_norm_px', float('nan')):.3f} vs "
      f"{G_D1['stable'].get('coherent_shift_norm_px', float('nan')):.3f} px | "
      f"scatter {G_D1['flip'].get('relative_scatter_mean_px', float('nan')):.3f} vs "
      f"{G_D1['stable'].get('relative_scatter_mean_px', float('nan')):.3f} px | "
      f"aniso {G_D1['flip'].get('anisotropy_ratio', float('nan')):.2f} vs "
      f"{G_D1['stable'].get('anisotropy_ratio', float('nan')):.2f}")
print(f"geom separation ratios (flip/stable): {geom_sep}")
print(f"D0 vs D1 geometry median ratios: {geom_d0d1}")
print(f"DECISION: {decision}")
print(rationale)
