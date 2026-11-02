"""Summarize experiments/loso_pilot/results.csv.

  python scripts/summarize_loso_pilot.py [--csv PATH] [--final]

Prints mean +- std (sample std, ddof=1) of held-out gesture accuracy and
seen-person val identity accuracy per model per fold and overall, then paired
differences B-A and C-A matched on (fold, seed).  --final uses the last-epoch
columns instead of the val-selected epoch.
"""

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, 'experiments', 'loso_pilot', 'results.csv')


def mean_std(xs):
    if not xs:
        return 'n/a'
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else float('nan')
    return f'{m:6.2f} +- {s:5.2f} (n={len(xs)})'


def t_p_value(t, df):
    """Two-sided p for Student's t; scipy if available, else None."""
    try:
        from scipy import stats
    except ImportError:
        return None
    return float(2 * stats.t.sf(abs(t), df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--final', action='store_true',
                    help='use last-epoch metrics instead of val-selected epoch')
    a = ap.parse_args()
    g_key = 'gesture_acc_heldout_final' if a.final else 'gesture_acc_heldout'
    i_key = 'identity_acc_val_final' if a.final else 'identity_acc_val'

    with open(a.csv, newline='') as fh:
        rows = list(csv.DictReader(fh))
    print(f'{len(rows)} finished runs in {a.csv}')
    print('epoch used:', 'final' if a.final else 'selected by seen-person val gesture acc')
    if not rows:
        return

    models = sorted({r['model'] for r in rows})
    folds = sorted({r['fold_person'] for r in rows})
    g = defaultdict(list)
    ident = defaultdict(list)
    by_key = {}
    for r in rows:
        gv, iv = float(r[g_key]), float(r[i_key])
        for f in (r['fold_person'], 'ALL'):
            g[(r['model'], f)].append(gv)
            ident[(r['model'], f)].append(iv)
        by_key[(r['model'], r['fold_person'], r['seed'])] = gv

    print('\n== Held-out-person gesture accuracy (%) ==')
    for m in models:
        for f in folds + ['ALL']:
            print(f'  {m}  fold={f:>3}  {mean_std(g[(m, f)])}')
        fold_means = [statistics.mean(g[(m, f)]) for f in folds if g[(m, f)]]
        print(f'  {m}  mean of fold means = {statistics.mean(fold_means):.2f}')

    print('\n== Seen-person val identity accuracy (%) (closed set, dual models) ==')
    for m in models:
        if all(r['task'] == 'gesture' for r in rows if r['model'] == m):
            continue
        for f in folds + ['ALL']:
            print(f'  {m}  fold={f:>3}  {mean_std(ident[(m, f)])}')

    print('\n== Paired differences on held-out gesture acc, matched on (fold, seed) ==')
    for other in ('B', 'C'):
        if 'A' not in models or other not in models:
            continue
        diffs = []
        per_fold = defaultdict(list)
        for (m, f, s), v in by_key.items():
            if m == other and ('A', f, s) in by_key:
                d = v - by_key[('A', f, s)]
                diffs.append(d)
                per_fold[f].append(d)
        if not diffs:
            continue
        for f in sorted(per_fold):
            print(f'  {other}-A  fold={f:>3}  {mean_std(per_fold[f])}')
        line = f'  {other}-A  ALL       {mean_std(diffs)}  positive={sum(d > 0 for d in diffs)}/{len(diffs)}'
        if len(diffs) > 1 and statistics.stdev(diffs) > 0:
            t = statistics.mean(diffs) / (statistics.stdev(diffs) / math.sqrt(len(diffs)))
            p = t_p_value(t, len(diffs) - 1)
            line += f'  paired t={t:.2f}' + (f' p={p:.3f}' if p is not None else '')
        print(line)
    print('\nNote: (fold, seed) pairs are not independent across seeds of one fold; '
          'with 3 folds treat p-values as descriptive only.')


if __name__ == '__main__':
    main()
