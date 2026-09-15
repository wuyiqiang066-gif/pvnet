"""EXP019 analysis: condition summary, matched-magnitude audit, typical-case
classification, pre-registered decision (GO-A/GO-B/GO-C).

Reads ONLY run_exp019.py artifacts (no model, no solver, no PnP re-run):
  results/per_image.csv
  results/run_audit.json
  results/reproduction_check.json
  config.json (pre-registered decision rules; read-only)

Writes:
  results/condition_summary.csv   (protocol section 14 table)
  results/matched_magnitude_check.json
  results/typical_cases.json      (protocol section 19)
  results/decision.json

Run from repo root:
  python experiments/exp019_residual_geometry_dose_response/scripts/analyze_exp019.py
"""
import os, sys, json, csv
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(EXP_DIR, 'results')

CFG = json.load(open(os.path.join(EXP_DIR, 'config.json')))
AUDIT = json.load(open(os.path.join(RES_DIR, 'run_audit.json')))
CHECK = json.load(open(os.path.join(RES_DIR, 'reproduction_check.json')))
PRIMARY = CFG['flip']['primary_gt_deg']          # 150
D1_RMS = CFG['d1_like_targets']['rms_anchor_median_per_image']
D1_MEANNORM = CFG['d1_like_targets']['meannorm_anchor_median_per_image']
D1_ANISO = CFG['d1_like_targets']['anisotropy_median']
D1_SHIFT = CFG['d1_like_targets']['coherent_shift_median']
F150 = 'flip_gt_150'

rows = list(csv.DictReader(open(os.path.join(RES_DIR, 'per_image.csv'))))
CONDS = [c['name'] for c in CFG['conditions']]
BY = {c: [r for r in rows if r['condition'] == c] for c in CONDS}
for c in CONDS:
    assert len(BY[c]) == 20, f'{c}: {len(BY[c])} rows != 20'


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def ibool(v):
    return str(v) == '1' or str(v) == 'True'


def cond_stats(name):
    rs = BY[name]
    rs_c = [r for r in rs if ibool(r['in_clean18'])]
    scale_raw = rs[0]['scale']
    scale = float(scale_raw) if scale_raw not in ('', 'nan') else None

    def block(rs_):
        fl = [r for r in rs_ if ibool(r[F150])]
        ra = [r for r in fl if fnum(r['delta_J_px']) > 0]
        dj = [fnum(r['delta_J_px']) for r in fl]
        dj_all = [fnum(r['delta_J_px']) for r in rs_]
        return dict(
            n=len(rs_),
            flip_90=sum(ibool(r['flip_gt_90']) for r in rs_),
            flip_120=sum(ibool(r['flip_gt_120']) for r in rs_),
            flip_150=len(fl),
            flip_170=sum(ibool(r['flip_gt_170']) for r in rs_),
            RA_150=len(ra),
            SF_150=len(fl) - len(ra),
            mean_delta_J=float(np.nanmean(dj_all))
            if any(np.isfinite(dj_all)) else float('nan'),
            median_delta_J=float(np.nanmedian(dj_all))
            if any(np.isfinite(dj_all)) else float('nan'),
            P_deltaJ_gt0=float(np.mean([fnum(r['delta_J_px']) > 0
                                        for r in rs_])) if rs_ else float('nan'),
            median_delta_J_flip150=float(np.median(dj)) if dj else float('nan'),
            n_divergence=sum(ibool(r['flag_solver_divergence']) for r in rs_),
            n_exception=sum(ibool(r['flag_pnp_exception']) for r in rs_))
    rms = float(np.mean([fnum(r['res_rms_px']) for r in rs]))
    mn = float(np.mean([fnum(r['res_meannorm_px']) for r in rs]))
    ans = [fnum(r['res_anisotropy']) for r in rs]
    ans = [a for a in ans if np.isfinite(a)]
    an = float(np.median(ans)) if ans else float('nan')
    return dict(condition=name, group=rs[0]['group'], scale=scale,
                anisotropy_level=rs[0]['anisotropy_level'],
                shift_level=rs[0]['shift_level'],
                RMS=round(rms, 4), mean_norm=round(mn, 4),
                anisotropy_median_achieved=round(an, 3),
                **{f'all20_{k}': v for k, v in block(rs).items()},
                **{f'clean18_{k}': v for k, v in block(rs_c).items()})


SUM = [cond_stats(c) for c in CONDS]
HEAD = ['condition', 'group', 'scale', 'anisotropy_level', 'shift_level',
        'RMS', 'mean_norm', 'anisotropy_median_achieved']
KEYS = ['flip_120', 'flip_150', 'flip_170', 'RA_150', 'SF_150',
        'mean_delta_J', 'median_delta_J', 'P_deltaJ_gt0',
        'median_delta_J_flip150', 'n_divergence']
with open(os.path.join(RES_DIR, 'condition_summary.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(HEAD + [f'all20_{k}' for k in KEYS]
                + [f'clean18_{k}' for k in KEYS]
                + ['all20_flip_90', 'clean18_flip_90', 'clean18_flip_150'])
    for s in SUM:
        wr.writerow([s[h] if h != 'scale' else
                     ('n/a' if s[h] is None else s[h]) for h in HEAD]
                    + [s[f'all20_{k}'] for k in KEYS]
                    + [s[f'clean18_{k}'] for k in KEYS]
                    + [s['all20_flip_90'], s['clean18_flip_90'],
                       s['clean18_flip_150']])

# ---------------- matched-magnitude audit (protocol section 4/13) ------------
syn = [s for s in SUM if s['group'] != 'control']
mm = dict(
    rule='per-image RMS == scale * D1_RMS EXACT by construction; mean_norm '
         'reported not forced; audit = max |achieved - target| over images',
    rms_max_abs_dev_px=AUDIT['synthetic_rms_max_abs_dev_px'],
    per_condition=[dict(condition=s['condition'],
                        RMS=s['RMS'], target_rms=round(s['scale'] * D1_RMS, 4),
                        mean_norm=s['mean_norm'],
                        target_meannorm_at_scale=round(s['scale'] * D1_MEANNORM, 4),
                        mean_norm_over_RMS=round(s['mean_norm'] / s['RMS'], 4),
                        d1_real_mean_norm_over_RMS=round(D1_MEANNORM / D1_RMS, 4),
                        anisotropy_achieved=s['anisotropy_median_achieved'],
                        anisotropy_target=(1.0 if s['anisotropy_level'] ==
                                           'ISOTROPIC' else D1_ANISO))
                   for s in syn])
json.dump(mm, open(os.path.join(RES_DIR, 'matched_magnitude_check.json'), 'w'),
          indent=2)

# ---------------- typical cases (protocol section 19) -------------------------
img_of = {r['rgb']: r['image_id'] for r in BY['C0_gt']}
rgb_order = [r['rgb'] for r in BY['C0_gt']]
sensitive, insensitive, contradictory = [], [], []
for rgb in rgb_order:
    f = {c: ibool([r for r in BY[c] if r['rgb'] == rgb][0][F150])
         for c in ('ISO_S05', 'ISO_S10', 'ISO_S15', 'D1G_S05', 'D1G_S10', 'D1G_S15')}
    if (f['D1G_S10'] and not f['ISO_S10']) or (f['D1G_S15'] and not f['ISO_S15']) \
            or (f['D1G_S05'] and not f['ISO_S05']):
        sensitive.append(rgb)
    if not any(f.values()):
        insensitive.append(rgb)
    if (f['ISO_S10'] and not f['D1G_S10']) or (f['ISO_S15'] and not f['D1G_S15']) \
            or (f['ISO_S05'] and not f['D1G_S05']):
        contradictory.append(rgb)


def detail(cs, rgb):
    r = [x for x in BY[cs] if x['rgb'] == rgb][0]
    return dict(condition=cs, rot=round(fnum(r['rot_err_deg']), 2),
                flip150=ibool(r[F150]), delta_J=round(fnum(r['delta_J_px']), 3),
                J_pred=round(fnum(r['J_pred_px']), 3),
                J_gt=round(fnum(r['J_gt_px']), 3),
                divergence=ibool(r['flag_solver_divergence']))


FOCUS = ['color_00033.png', 'color_00042.png', 'color_00039.png',
         'color_00026.png', 'color_00027.png']
focus_detail = {rgb: {c: detail(c, rgb) for c in CONDS[3:]} for rgb in FOCUS}
TC = dict(
    geometry_sensitive=dict(n=len(sensitive), images=sensitive[:6],
                            detail={rgb: {c: detail(c, rgb)
                                          for c in ('ISO_S10', 'D1G_S10',
                                                    'ISO_S15', 'D1G_S15')}
                                    for rgb in sensitive[:3]}),
    geometry_insensitive=dict(n=len(insensitive), images=insensitive[:6],
                              detail={rgb: {c: detail(c, rgb)
                                            for c in ('D1G_S10', 'D1G_S15')}
                                      for rgb in insensitive[:3]}),
    contradictory=dict(n=len(contradictory), images=contradictory[:6],
                       detail={rgb: {c: detail(c, rgb)
                                     for c in ('ISO_S10', 'D1G_S10')}
                               for rgb in contradictory[:3]}),
    forced_focus_images=focus_detail,
    note='classification only; NO image removed from any statistic')
json.dump(TC, open(os.path.join(RES_DIR, 'typical_cases.json'), 'w'), indent=2)

# ---------------- pre-registered decision -------------------------------------
S = {s['condition']: s for s in SUM}
g = {k: S[f'D1G_S{k}']['all20_flip_150'] - S[f'ISO_S{k}']['all20_flip_150']
     for k in ('05', '10', '15')}
gsum = sum(g.values())
ra = {k: (S[f'D1G_S{k}']['all20_RA_150'], S[f'ISO_S{k}']['all20_RA_150'])
      for k in ('05', '10', '15')}
clean_ok = {k: (S[f'D1G_S{k}']['clean18_flip_150'], S[f'ISO_S{k}']['clean18_flip_150'])
            for k in ('05', '10', '15')}

if not CHECK['gate_ok']:
    decision, direction, rationale = 'STOP', 'n/a', 'baseline gate failed.'
else:
    d1_promoting = all(v >= 0 for v in g.values()) and gsum >= 3
    iso_promoting = all(v <= 0 for v in g.values()) and -gsum >= 3
    if d1_promoting or iso_promoting:
        direction = 'D1_promoting' if d1_promoting else 'isotropy_promoting'
        sign = 1 if d1_promoting else -1
        ra_wins = sum(1 for k in ra if sign * (ra[k][0] - ra[k][1]) > 0)
        punch = (S['D1G_S10']['all20_flip_150'] >= S['ISO_S15']['all20_flip_150']) \
            if d1_promoting else \
            (S['ISO_S10']['all20_flip_150'] >= S['D1G_S15']['all20_flip_150'])
        cg_raw = sum(a - b for a, b in clean_ok.values())   # clean18 D1G - ISO
        consistent = (cg_raw >= 0) if d1_promoting else (cg_raw <= 0)
        if ra_wins >= 2 and punch and consistent:
            decision, rationale = 'GO-A', (
                f'Geometry-causal ({direction}): g = {g} (sum {gsum:+d}/20); '
                f'RA ordering wins {ra_wins}/3 scales; punchline '
                f'(promoted geometry @1.0x >= other @1.5x) = {punch}; CLEAN18 '
                f'gap {cg_raw:+d} consistent. Residual geometry, not magnitude '
                f'alone, is causally sufficient to move the original PnP '
                f'reprojection objective toward the ~180deg basin.')
        else:
            decision = 'GO-C'
            rationale = (f'Flip-count separation present ({direction}, g = {g}) '
                         f'but supporting criteria incomplete: RA ordering wins '
                         f'{ra_wins}/3 scales (RA(D1G) vs RA(ISO) = '
                         f'{[ra[k] for k in ("05", "10", "15")]}), punchline '
                         f'{punch}, CLEAN18 gap {cg_raw:+d} '
                         f'{"consistent" if consistent else "INCONSISTENT"} '
                         f'with ALL20 direction -> mixed/inconclusive; '
                         f'conservative wording per protocol section 18.')
    elif all(abs(v) <= 1 for v in g.values()):
        rise = max(S['ISO_S15']['all20_flip_150'] - S['ISO_S05']['all20_flip_150'],
                   S['D1G_S15']['all20_flip_150'] - S['D1G_S05']['all20_flip_150'])
        if rise >= 3:
            decision, direction, rationale = 'GO-B', 'n/a', (
                f'Magnitude-only: geometry gaps {g} all <= 1 image, while '
                f'flip150 rises {rise} from 0.5x to 1.5x within a geometry '
                f'family. No geometry-causal effect established; STOP.')
        else:
            decision, direction = 'GO-C', 'n/a'
            rationale = (f'No geometry separation (g = {g}) and no clear '
                         f'magnitude dose-response (rise {rise}) -> inconclusive.')
    else:
        decision, direction = 'GO-C', 'n/a'
        rationale = (f'Mixed: geometry gaps {g} inconsistent in sign/magnitude '
                     f'-> suggestive but insufficient evidence (protocol '
                     f'section 18).')

DEC = dict(decision=decision, direction=direction, rationale=rationale,
           gate_ok=bool(CHECK['gate_ok']),
           gaps_flip150_all20=g, ra_150_by_scale=ra,
           clean18_flip150_by_scale=clean_ok,
           conditions=SUM, rules='pre-registered in config.json')
json.dump(DEC, open(os.path.join(RES_DIR, 'decision.json'), 'w'), indent=2)

# ---------------- console report ----------------------------------------------
print('=== EXP019 condition summary (ALL 20 | CLEAN 18) ===')
print(f"{'condition':<12} {'RMS':>6} {'mnorm':>6} {'aniso':>6} | "
      f"{'f90':>3} {'f120':>4} {'f150':>4} {'f170':>4} {'RA':>3} {'SF':>3} "
      f"{'medDJ':>7} {'P(DJ>0)':>7} | {'f150c':>5} {'RAc':>3}")
for s in SUM:
    print(f"{s['condition']:<12} {s['RMS']:>6.3f} {s['mean_norm']:>6.3f} "
          f"{s['anisotropy_median_achieved']:>6.2f} | "
          f"{s['all20_flip_90']:>3} {s['all20_flip_120']:>4} "
          f"{s['all20_flip_150']:>4} {s['all20_flip_170']:>4} "
          f"{s['all20_RA_150']:>3} {s['all20_SF_150']:>3} "
          f"{s['all20_median_delta_J']:>7.3f} {s['all20_P_deltaJ_gt0']:>7.2f} | "
          f"{s['clean18_flip_150']:>5} {s['clean18_RA_150']:>3}")
print('=== gaps flip150 D1G - ISO (ALL 20) ===')
print({k: g[k] for k in ('05', '10', '15')})
print('=== matched magnitude ===')
print(f"synthetic RMS max |dev| vs target: {mm['rms_max_abs_dev_px']:.2e} px")
print('=== typical cases ===')
print(f"geometry_sensitive (n={TC['geometry_sensitive']['n']}): "
      f"{TC['geometry_sensitive']['images']}")
print(f"geometry_insensitive (n={TC['geometry_insensitive']['n']}): "
      f"{TC['geometry_insensitive']['images']}")
print(f"contradictory (n={TC['contradictory']['n']}): "
      f"{TC['contradictory']['images']}")
print('=== forced focus ===')
for rgb, d in focus_detail.items():
    line = ' | '.join(f"{c}:{v['rot']:.0f}deg dJ{v['delta_J']:+.2f}"
                      + (' DIV' if v['divergence'] else '')
                      for c, v in d.items() if c in
                      ('ISO_S10', 'D1G_S10', 'D1G_S15', 'D1G_S05'))
    print(f"  {rgb}: {line}")
print(f"DECISION: {decision} ({direction})")
print(rationale)
