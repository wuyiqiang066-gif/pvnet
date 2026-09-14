"""EXP007 figures (3 max) from eval CSVs."""
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def read_csv(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def get(rows, tag, split, region, col):
    for r in rows:
        if r['tag'] == tag and r['split'] == split and r['region'] == region:
            return float(r[col])
    raise KeyError((tag, split, region))


def getkp(rows, tag, kp, col='dir_mean_deg'):
    for r in rows:
        if r['tag'] == tag and r['kp'] == kp:
            return float(r[col])
    raise KeyError((tag, kp))


d = read_csv('direction_error.csv') if os.path.exists(os.path.join(RES, 'direction_error.csv')) else None
# merge per-tag direction CSVs
rows_d = []
for t in ('ref199', 'controlA', 'exp007'):
    for r in read_csv(f'direction_error_{t}.csv'):
        rows_d.append(r)
rows_k = []
for t in ('ref199', 'controlA', 'exp007'):
    for r in read_csv(f'keypoint_error_{t}.csv'):
        if r['split'] == 'occ':
            rows_k.append(r)
rows_p = []
for t in ('ref199', 'controlA', 'exp007'):
    for r in read_csv(f'occlusion_proximity_{t}.csv'):
        rows_p.append(r)

TAGS = [('ref199', '199.pth (ref)'), ('controlA', 'Control A\n(baseline 20ep)'), ('exp007', 'EXP007\n(context 20ep)')]
COLORS = ['gray', 'steelblue', 'indianred']

# ---- Fig 1: OCC direction error by region ----
fig, ax = plt.subplots(figsize=(9, 4.8))
regs = ['all', 'interior', 'normal_boundary', 'occlusion_boundary']
x = np.arange(len(regs)); wd = 0.26
for i, (tag, lab) in enumerate(TAGS):
    vals = [get(rows_d, tag, 'occ', r, 'mean_deg') for r in regs]
    ax.bar(x + (i - 1) * wd, vals, wd, label=lab, color=COLORS[i])
    for j, v in enumerate(vals):
        ax.text(j + (i - 1) * wd, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(regs)
ax.set_ylabel('direction error (deg, mean)'); ax.legend()
ax.set_title('OCC direction error: 199.pth vs Control A vs EXP007')
ax.grid(alpha=.3, axis='y')
plt.tight_layout(); plt.savefig(os.path.join(RES, 'fig1_occ_direction.png'), dpi=120); plt.close()

# ---- Fig 2: kp2/kp3/kp7 + other6 pooled ----
fig, ax = plt.subplots(figsize=(9, 4.8))
groups = ['kp2', 'kp3', 'kp7', 'other6 pooled']
x = np.arange(4); wd = 0.26
for i, (tag, lab) in enumerate(TAGS):
    vals = [getkp(rows_k, tag, 'kp2'), getkp(rows_k, tag, 'kp3'), getkp(rows_k, tag, 'kp7')]
    others = [getkp(rows_k, tag, f'kp{k}') for k in range(9) if f'kp{k}' not in ('kp2', 'kp3', 'kp7')]
    vals.append(np.mean(others))
    ax.bar(x + (i - 1) * wd, vals, wd, label=lab, color=COLORS[i])
    for j, v in enumerate(vals):
        ax.text(j + (i - 1) * wd, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel('OCC direction error (deg, mean)'); ax.legend()
ax.set_title('EXP006-identified worst keypoints: baseline vs EXP007')
ax.grid(alpha=.3, axis='y')
plt.tight_layout(); plt.savefig(os.path.join(RES, 'fig2_keypoints.png'), dpi=120); plt.close()

# ---- Fig 3: direction error vs occlusion proximity ----
fig, ax = plt.subplots(figsize=(8, 4.8))
bins = ['20-inf', '10-20', '5-10', '2-5', '0-2']
xpos = np.arange(5)
for i, (tag, lab) in enumerate(TAGS):
    vals = []
    for b in bins:
        for r in rows_p:
            if r['tag'] == tag and r['dist_bin_px'] == b:
                vals.append(float(r['dir_mean_deg']))
    ax.plot(xpos, vals, 'o-', label=lab, color=COLORS[i])
ax.set_xticks(xpos); ax.set_xticklabels(bins)
ax.set_xlabel('distance to occlusion region (px)')
ax.set_ylabel('OCC direction error (deg, mean)')
ax.set_title('direction error vs occlusion proximity')
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(RES, 'fig3_proximity.png'), dpi=120); plt.close()
print('figures saved')
