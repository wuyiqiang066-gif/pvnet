"""EXP020 analysis: dose-response curves, transition diagnostics, per-image
stability margins, D0/D1 placement, pre-registered decision (GO-A/GO-B/GO-C).

Reads ONLY run_exp020.py artifacts (no model, no solver, no PnP re-run):
  results/per_image.csv
  results/run_audit.json
  results/reproduction_check.json
  config.json (pre-registered decision rules; read-only)

Writes:
  results/condition_summary.csv     (per condition, ALL20 + CLEAN18)
  results/dose_response.csv         (protocol section 13 table, ordered by sigma)
  results/per_image_transition.csv  (protocol section 16: first flip / first
                                     positive delta_J sigma per image)
  results/decision.json
  results/dose_response_flip.png, results/dose_response_deltaJ.png
  (max 2 figures, protocol section 27)

Descriptive statistics and simple finite differences ONLY (protocol section
18): no model fitting, no change-point optimization, no hypothesis testing.

Run from repo root:
  python experiments/exp020_residual_magnitude_pnp_threshold/scripts/analyze_exp020.py
"""
import os, json, csv
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(EXP_DIR, 'results')

CFG = json.load(open(os.path.join(EXP_DIR, 'config.json')))
AUDIT = json.load(open(os.path.join(RES_DIR, 'run_audit.json')))
CHECK = json.load(open(os.path.join(RES_DIR, 'reproduction_check.json')))
PRIMARY = CFG['flip']['primary_gt_deg']                       # 150
SIGMAS = CFG['residual']['sigma_levels_px']                   # 12 levels
D0_RMS = CFG['d0_d1_placement']['d0_median_per_image_rms_px']
D1_RMS = CFG['d0_d1_placement']['d1_median_per_image_rms_px']
F150 = 'flip_gt_150'


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def ibool(v):
    return str(v).strip() == '1'


rows = list(csv.DictReader(open(os.path.join(RES_DIR, 'per_image.csv'))))
CONDS = (['C0_gt', 'C1_D0_real', 'C2_D1_real'] +
         ['SIG_S%03d' % round(s * 100) for s in SIGMAS])
BY = {c: {int(r['image_id']): r for r in rows if r['condition'] == c}
      for c in CONDS}
for c in CONDS:
    assert len(BY[c]) == 20, f'{c}: {len(BY[c])} rows != 20'
IMGS = sorted(BY['C0_gt'])
RGB = {i: BY['C0_gt'][i]['rgb'] for i in IMGS}
CLEAN = {i: ibool(BY['C0_gt'][i]['in_clean18']) for i in IMGS}
IMG_CLEAN = [i for i in IMGS if CLEAN[i]]
assert len(IMG_CLEAN) == 18
SIG_CONDS = ['SIG_S%03d' % round(s * 100) for s in SIGMAS]


def rate(cond, mask, key=F150):
    return float(np.mean([ibool(BY[cond][i][key]) for i in IMGS if mask(i)]))


def frac(cond, mask, fn):
    vals = [fn(BY[cond][i]) for i in IMGS if mask(i)]
    vals = [v for v in vals if np.isfinite(v)]
    return vals


def med(cond, mask, col):
    v = np.asarray(frac(cond, mask, lambda r: fnum(r[col])), np.float64)
    return float(np.median(v)) if v.size else float('nan')


# ---------------- condition summary (controls + synthetic) --------------------
csum = []
for c in CONDS:
    for tag, msk in (('all20', lambda i: True), ('clean18', lambda i: CLEAN[i])):
        n = len([i for i in IMGS if msk(i)])
        dj = frac(c, msk, lambda r: fnum(r['delta_J_px']))
        csum.append(dict(
            condition=c, subset=tag, n=n,
            sigma=(fnum(BY[c][IMGS[0]]['sigma']) if c.startswith('SIG_') else None),
            rms_mean_over_images=(float(np.mean([fnum(BY[c][i]['res_rms_px'])
                                                 for i in IMGS])) if c.startswith('SIG_')
                                  else float(np.median([fnum(BY[c][i]['res_rms_px'])
                                                        for i in IMGS]))),
            rms_stat_kind=('mean over images' if c.startswith('SIG_')
                           else 'median over images (real residuals are '
                                'heteroscedastic; median is the placement stat)'),
            mean_norm=float(np.mean([fnum(BY[c][i]['res_meannorm_px']) for i in IMGS])),
            median_norm=float(np.mean([fnum(BY[c][i]['res_median_px']) for i in IMGS])),
            p90_norm=float(np.mean([fnum(BY[c][i]['res_p90_px']) for i in IMGS])),
            coherent_shift=float(np.mean([fnum(BY[c][i]['res_coherent_px'])
                                          for i in IMGS])),
            flip_90=round(rate(c, msk, 'flip_gt_90'), 4),
            flip_120=round(rate(c, msk, 'flip_gt_120'), 4),
            flip_150=round(rate(c, msk), 4),
            flip_170=round(rate(c, msk, 'flip_gt_170'), 4),
            mean_rot=round(float(np.nanmean([fnum(BY[c][i]['rot_err_deg'])
                                             for i in IMGS if msk(i)])), 3),
            median_rot=round(med(c, msk, 'rot_err_deg'), 3),
            mean_reproj=round(float(np.nanmean([fnum(BY[c][i]['reproj_official_px'])
                                                for i in IMGS if msk(i)])), 3),
            median_reproj=round(med(c, msk, 'reproj_official_px'), 3),
            median_delta_J=round(float(np.median(dj)), 4) if dj else float('nan'),
            mean_delta_J=round(float(np.mean(dj)), 4) if dj else float('nan'),
            P_deltaJ_gt0=round(float(np.mean([d > 0 for d in dj])), 4) if dj else float('nan'),
            n_divergence=sum(1 for i in IMGS if msk(i)
                             and ibool(BY[c][i]['flag_solver_divergence'])),
            n_exception=sum(1 for i in IMGS if msk(i)
                            and ibool(BY[c][i]['flag_pnp_exception']))))
with open(os.path.join(RES_DIR, 'condition_summary.csv'), 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=list(csum[0]))
    wr.writeheader()
    wr.writerows(csum)

# ---------------- dose-response (12 synthetic levels) -------------------------
S = {}
prev = None
for cond in SIG_CONDS:
    rms_m = float(np.mean([fnum(BY[cond][i]['res_rms_px']) for i in IMGS]))
    rms_pooled = float(np.sqrt(np.mean([fnum(BY[cond][i]['res_rms_px']) ** 2
                                        for i in IMGS])))
    f150a, f150c = rate(cond, lambda i: True), rate(cond, CLEAN.__getitem__)
    pra_a = float(np.mean([fnum(BY[cond][i]['delta_J_px']) > 0 for i in IMGS]))
    pra_c = float(np.mean([fnum(BY[cond][i]['delta_J_px']) > 0
                           for i in IMG_CLEAN]))
    row = dict(
        condition=cond, nominal_sigma=fnum(BY[cond][IMGS[0]]['sigma']),
        actual_rms_mean=round(rms_m, 4), actual_rms_pooled=round(rms_pooled, 4),
        mean_norm=round(float(np.mean([fnum(BY[cond][i]['res_meannorm_px'])
                                       for i in IMGS])), 4),
        median_norm=round(float(np.mean([fnum(BY[cond][i]['res_median_px'])
                                         for i in IMGS])), 4),
        p90_norm=round(float(np.mean([fnum(BY[cond][i]['res_p90_px'])
                                      for i in IMGS])), 4),
        coherent_shift=round(float(np.mean([fnum(BY[cond][i]['res_coherent_px'])
                                            for i in IMGS])), 4),
        flip150_all20=round(f150a, 4), flip150_clean18=round(f150c, 4),
        flip170_all20=round(rate(cond, lambda i: True, 'flip_gt_170'), 4),
        flip170_clean18=round(rate(cond, CLEAN.__getitem__, 'flip_gt_170'), 4),
        median_rot_clean18=round(med(cond, CLEAN.__getitem__, 'rot_err_deg'), 3),
        median_reproj_clean18=round(med(cond, CLEAN.__getitem__,
                                        'reproj_official_px'), 3),
        P_deltaJ_gt0_all20=round(pra_a, 4), P_deltaJ_gt0_clean18=round(pra_c, 4),
        median_delta_J_clean18=round(med(cond, CLEAN.__getitem__,
                                         'delta_J_px'), 4),
        mean_delta_J_clean18=round(float(np.mean(frac(cond, CLEAN.__getitem__,
                                                      lambda r: fnum(r['delta_J_px'])))), 4),
        n_divergence=sum(1 for i in IMGS
                         if ibool(BY[cond][i]['flag_solver_divergence'])))
    if prev is not None:
        row['delta_flip150_all20'] = round(row['flip150_all20']
                                           - prev['flip150_all20'], 4)
        row['delta_flip150_clean18'] = round(row['flip150_clean18']
                                             - prev['flip150_clean18'], 4)
        row['delta_P_RA_clean18'] = round(row['P_deltaJ_gt0_clean18']
                                          - prev['P_deltaJ_gt0_clean18'], 4)
        row['delta_median_dJ_clean18'] = round(
            (row['median_delta_J_clean18'] - prev['median_delta_J_clean18'])
            if np.isfinite(row['median_delta_J_clean18'])
            and np.isfinite(prev['median_delta_J_clean18']) else float('nan'), 4)
    else:
        row['delta_flip150_all20'] = row['delta_flip150_clean18'] = \
            row['delta_P_RA_clean18'] = row['delta_median_dJ_clean18'] = None
    S[cond] = row
    prev = row
with open(os.path.join(RES_DIR, 'dose_response.csv'), 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=list(S[SIG_CONDS[0]]))
    wr.writeheader()
    wr.writerows(S[c] for c in SIG_CONDS)

# ---------------- per-image transitions (protocol section 16) -----------------
TRANS = []
for i in IMGS:
    first_flip, first_flip_rms = None, None
    first_djpos = None
    n_flip = 0
    for cond in SIG_CONDS:
        r = BY[cond][i]
        if ibool(r[F150]):
            n_flip += 1
            if first_flip is None:
                first_flip = fnum(r['sigma'])
                first_flip_rms = fnum(r['res_rms_px'])
        if first_djpos is None and fnum(r['delta_J_px']) > 0:
            first_djpos = fnum(r['sigma'])
    TRANS.append(dict(
        image_id=i, rgb=RGB[i], in_clean18=int(CLEAN[i]),
        first_sigma_flip150=('NA' if first_flip is None else first_flip),
        first_flip_actual_rms_px=('NA' if first_flip_rms is None
                                  else round(first_flip_rms, 4)),
        first_sigma_deltaJ_positive=('NA' if first_djpos is None else first_djpos),
        n_levels_flip150=n_flip,
        rot_at_sigma4=round(fnum(BY[SIG_CONDS[-1]][i]['rot_err_deg']), 2)))
with open(os.path.join(RES_DIR, 'per_image_transition.csv'), 'w',
          newline='') as f:
    wr = csv.DictWriter(f, fieldnames=list(TRANS[0]))
    wr.writeheader()
    wr.writerows(TRANS)

# ---------------- transition diagnostics (pre-registered) ---------------------
flips_c = [S[c]['flip150_clean18'] for c in SIG_CONDS]
flips_a = [S[c]['flip150_all20'] for c in SIG_CONDS]
dflips_c = [S[c]['delta_flip150_clean18'] for c in SIG_CONDS[1:]]
pra_c = [S[c]['P_deltaJ_gt0_clean18'] for c in SIG_CONDS]
dmedj_c = [S[c]['delta_median_dJ_clean18'] for c in SIG_CONDS[1:]]
k_star = int(np.argmax(dflips_c))            # jump between level k_star and k_star+1
jump_max = dflips_c[k_star]
sig_lo, sig_hi = SIGMAS[k_star], SIGMAS[k_star + 1]
conds_lo, conds_hi = SIG_CONDS[k_star], SIG_CONDS[k_star + 1]

# drivers: CLEAN18 images whose FIRST flip150 happens inside the transition pair
drivers, div_drivers = [], []
for i in IMG_CLEAN:
    ff = TRANS[[t['image_id'] for t in TRANS].index(i)]['first_sigma_flip150']
    if ff != 'NA' and sig_lo <= ff <= sig_hi:
        (div_drivers if ibool(BY[conds_hi][i]['flag_solver_divergence'])
         else drivers).append(RGB[i])

# delta_J sync: finite-difference monotonicity of median delta_J on CLEAN18
mono_steps = int(np.sum([d >= 0 for d in dmedj_c if d is not None
                         and np.isfinite(d)]))
n_finite_steps = int(np.sum([d is not None and np.isfinite(d) for d in dmedj_c]))
dPRA_pair = S[conds_hi]['P_deltaJ_gt0_clean18'] - S[conds_lo]['P_deltaJ_gt0_clean18']

# placement: real D1 RMS vs transition window (nominal sigmas, as registered)
d1_in_window = bool((D1_RMS >= sig_lo - 0.5) and (D1_RMS <= sig_hi + 0.5))
# nearest synthetic level to each real anchor (by nominal sigma)
def nearest(s):
    k = int(np.argmin([abs(s - v) for v in SIGMAS]))
    return SIG_CONDS[k], S[SIG_CONDS[k]]
near_d1, near_d1_row = nearest(D1_RMS)
near_d0, near_d0_row = nearest(D0_RMS)

# ---------------- pre-registered decision --------------------------------------
crit = dict(
    i_jump_max_ge_020=bool(jump_max >= 0.20),
    iia_median_dJ_mono_steps=f'{mono_steps}/{n_finite_steps}',
    iia_median_dJ_mono=bool(mono_steps >= 9),
    iib_delta_P_RA_pair=round(dPRA_pair, 4),
    iib_delta_P_RA_ge_010=bool(dPRA_pair >= 0.10),
    iii_n_drivers_in_pair=len(drivers),
    iii_not_1_2_image_driven=bool(len(drivers) >= 3),
    iv_d1_rms_in_window=bool(d1_in_window))
go_a = all([crit['i_jump_max_ge_020'], crit['iia_median_dJ_mono'],
            crit['iib_delta_P_RA_ge_010'], crit['iii_not_1_2_image_driven'],
            crit['iv_d1_rms_in_window']])
overall_rise = flips_c[-1] - flips_c[0]       # sigma=4.0 vs sigma=0.0 (registered)
go_b = (not go_a) and bool(overall_rise >= 0.30)
if go_a:
    decision = 'GO-A'
elif go_b:
    decision = 'GO-B'
else:
    decision = 'GO-C'
rationale = (
    f"CLEAN18 flip150 vs actual RMS: jump_max {jump_max:+.2f} at pair "
    f"[{sig_lo}, {sig_hi}] px (nominal); criteria {crit}; overall rise "
    f"sigma0.25->4.0 {overall_rise:+.2f}. "
    + ('All GO-A criteria met.' if go_a else
       'GO-A criteria NOT all met; '
       + ('overall dose-response present without a qualifying knee -> GO-B.'
          if go_b else 'no qualifying dose-response -> GO-C/STOP.')))

# margin classes (descriptive, protocol section 20)
def margin_class(t):
    if t == 'NA':
        return 'never (NA at sigma<=4.0)'
    if t <= 1.0:
        return 'early (<=1.0 px)'
    if t <= 2.5:
        return 'middle (1.25-2.5 px)'
    return 'late (>2.5 px)'


margins = {}
for t in TRANS:
    if t['in_clean18']:
        margins.setdefault(margin_class(t['first_sigma_flip150']), []).append(t['rgb'])

# ---------------- plots (max 2, protocol section 27) ---------------------------
plot_note = 'skipped'
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = [S[c]['actual_rms_mean'] for c in SIG_CONDS]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, flips_a, 'o-', label='ALL20', color='#1f77b4')
    ax.plot(x, flips_c, 's-', label='CLEAN18', color='#d62728')
    for s_val, lbl, col in ((D0_RMS, 'D0 real RMS', '#2ca02c'),
                            (D1_RMS, 'D1 real RMS', '#9467bd')):
        ax.axvline(s_val, ls='--', lw=1, color=col)
        ax.text(s_val, 1.02, lbl, rotation=90, va='bottom', fontsize=7, color=col)
    ax.axvspan(S[conds_lo]['actual_rms_mean'], S[conds_hi]['actual_rms_mean'],
               color='orange', alpha=0.15,
               label=f'max-jump pair [{sig_lo},{sig_hi}]')
    ax.set_xlabel('actual residual RMS (px, mean over images)')
    ax.set_ylabel(f'flip rot>{PRIMARY:.0f} deg rate')
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7)
    ax.set_title('EXP020 flip dose-response (isotropic, shared realization)')
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, 'dose_response_flip.png'), dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, [S[c]['P_deltaJ_gt0_clean18'] for c in SIG_CONDS], 's-',
            label='P(delta_J>0) CLEAN18', color='#d62728')
    ax.plot(x, [S[c]['P_deltaJ_gt0_all20'] for c in SIG_CONDS], 'o-',
            label='P(delta_J>0) ALL20', color='#1f77b4')
    for s_val, lbl, col in ((D0_RMS, 'D0 real RMS', '#2ca02c'),
                            (D1_RMS, 'D1 real RMS', '#9467bd')):
        ax.axvline(s_val, ls='--', lw=1, color=col)
        ax.text(s_val, 1.02, lbl, rotation=90, va='bottom', fontsize=7, color=col)
    ax.set_xlabel('actual residual RMS (px, mean over images)')
    ax.set_ylabel('P(delta_J > 0)')
    ax.set_ylim(-0.03, 1.12)
    ax.legend(fontsize=7)
    ax.set_title('EXP020 objective-ambiguity signal dose-response')
    fig.tight_layout()
    fig.savefig(os.path.join(RES_DIR, 'dose_response_deltaJ.png'), dpi=150)
    plt.close(fig)
    plot_note = '2 figures written'
except Exception as e:                                       # noqa: BLE001
    plot_note = f'plots skipped: {e}'

# ---------------- persist -------------------------------------------------------
json.dump(dict(
    gate_ok=bool(CHECK['gate_ok']),
    decision=decision,
    jump_max_clean18=round(jump_max, 4),
    transition_pair_nominal_sigma=[sig_lo, sig_hi],
    transition_pair_actual_rms=[S[conds_lo]['actual_rms_mean'],
                                S[conds_hi]['actual_rms_mean']],
    drivers_first_flip_in_pair=drivers,
    divergence_flagged_in_pair=div_drivers,
    all20_jump_max=round(max(dflips := [S[c]['delta_flip150_all20']
                                        for c in SIG_CONDS[1:]]), 4),
    all20_jump_pair=[SIGMAS[int(np.argmax(dflips))],
                     SIGMAS[int(np.argmax(dflips)) + 1]],
    criteria=crit,
    overall_rise_025_to_400=round(overall_rise, 4),
    flips_clean18_by_sigma={c: S[c]['flip150_clean18'] for c in SIG_CONDS},
    flips_all20_by_sigma={c: S[c]['flip150_all20'] for c in SIG_CONDS},
    d0_d1_placement=dict(
        d0_median_rms=D0_RMS, d1_median_rms=D1_RMS,
        nearest_level_d0=near_d0, nearest_flip150_clean18_d0=near_d0_row['flip150_clean18'],
        nearest_level_d1=near_d1, nearest_flip150_clean18_d1=near_d1_row['flip150_clean18'],
        real_C1_D0_flip150=round(rate('C1_D0_real', lambda i: True), 4),
        real_C2_D1_flip150=round(rate('C2_D1_real', lambda i: True), 4),
        real_C1_D0_flip150_clean18=round(rate('C1_D0_real', CLEAN.__getitem__), 4),
        real_C2_D1_flip150_clean18=round(rate('C2_D1_real', CLEAN.__getitem__), 4),
        d1_rms_inside_transition_window=d1_in_window),
    margin_classes_clean18={k: v for k, v in sorted(margins.items())},
    pathological_note='color_00026/27 kept in ALL20 everywhere; CLEAN18 fixed '
                      'a priori; divergence events flagged per row',
    plots=plot_note,
    rules='pre-registered in config.json; descriptive statistics only',
    rationale=rationale,
), open(os.path.join(RES_DIR, 'decision.json'), 'w'), indent=2)

print('=== EXP020 dose-response (actual RMS | flip150 ALL20/CLEAN18 | '
      'P(dJ>0) CLEAN18 | med dJ CLEAN18) ===')
for c in SIG_CONDS:
    print(f"{c} sig={S[c]['nominal_sigma']:.2f} rms={S[c]['actual_rms_mean']:.3f} "
          f"mn={S[c]['mean_norm']:.3f} p90={S[c]['p90_norm']:.3f} | "
          f"f150 {S[c]['flip150_all20']:.2f}/{S[c]['flip150_clean18']:.2f} "
          f"f170 {S[c]['flip170_clean18']:.2f} | "
          f"RA {S[c]['P_deltaJ_gt0_clean18']:.2f} "
          f"dJ {S[c]['median_delta_J_clean18']:+.3f} | div {S[c]['n_divergence']}")
print('=== adjacent deltas (CLEAN18) ===')
print('d_flip150:', [S[c]['delta_flip150_clean18'] for c in SIG_CONDS[1:]])
print('d_P_RA   :', [S[c]['delta_P_RA_clean18'] for c in SIG_CONDS[1:]])
print('=== real anchors ===')
print(f"D0 real: median RMS {D0_RMS:.3f} px -> nearest {near_d0} "
      f"(synthetic flip150 clean18 {near_d0_row['flip150_clean18']:.2f}); "
      f"real C1 flip150 {rate('C1_D0_real', lambda i: True):.2f} "
      f"(clean {rate('C1_D0_real', CLEAN.__getitem__):.2f})")
print(f"D1 real: median RMS {D1_RMS:.3f} px -> nearest {near_d1} "
      f"(synthetic flip150 clean18 {near_d1_row['flip150_clean18']:.2f}); "
      f"real C2 flip150 {rate('C2_D1_real', lambda i: True):.2f} "
      f"(clean {rate('C2_D1_real', CLEAN.__getitem__):.2f})")
print('=== margins (CLEAN18) ===')
for k, v in sorted(margins.items()):
    print(f'  {k}: {len(v)} {v}')
print('=== pathological ===')
for rgb in ('color_00026.png', 'color_00027.png'):
    i = [k for k, v in RGB.items() if v == rgb][0]
    r26 = [(S[c]['nominal_sigma'],
            int(ibool(BY[c][i][F150])),
            int(ibool(BY[c][i]['flag_solver_divergence'])),
            round(fnum(BY[c][i]['rot_err_deg']), 1)) for c in SIG_CONDS]
    print(f'  {rgb}: (sigma, flip150, div, rot) {r26}')
print(f'DECISION: {decision}')
print(rationale)
