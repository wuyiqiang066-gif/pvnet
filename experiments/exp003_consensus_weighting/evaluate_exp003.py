"""EXP003 metric aggregation / comparison table.

Usage: python experiments/exp003_consensus_weighting/evaluate_exp003.py \
    --metrics results/metrics_linemod_full.json results/metrics_occ_full.json
"""
import os, sys, json, argparse

EXP_DIR = os.path.dirname(os.path.abspath(__file__))


def table(metrics_files):
    for mf in metrics_files:
        data = json.load(open(mf))
        print('\n=== {} ==='.format(os.path.basename(mf)))
        for seed, methods in sorted(data.items()):
            base = methods.get('baseline', {}).get('add')
            print('seed {}:'.format(seed))
            print('  {:<10} {:>10} {:>10} {:>10} {:>12}'.format(
                'method', 'ADD(-S)', 'proj', '5cm5deg', 'dADD vs base'))
            for m, v in methods.items():
                d = (v['add'] - base) if base is not None else 0.0
                print('  {:<10} {:>10.4f} {:>10.4f} {:>10.4f} {:>+12.4f}'.format(
                    m, v['add'], v['projection_error'], v['cm_degree_5'], d))
            print('  fallbacks: ' + '; '.join(
                '{}: {} (rate {:.4f})'.format(m, v['fallback_counts'],
                                              v['fallback_rate'])
                for m, v in methods.items()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metrics', nargs='+', required=True)
    a = p.parse_args()
    table(a.metrics)
