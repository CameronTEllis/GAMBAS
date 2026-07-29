#!/usr/bin/env python3
"""
aggregate_cv.py
===============
Pool the per-fold `per_volume.csv` files from a cross-validation run into one
cross-validated estimate, overall and per subgroup.

Why pool per-volume rather than average the fold means
------------------------------------------------------
In K-fold CV each volume is held out exactly once, so concatenating the folds'
per-volume rows gives exactly one out-of-sample prediction per volume. The mean
over those rows is the cross-validated point estimate, and it weights every
volume equally. Averaging the five fold-means instead weights small folds more
heavily, which matters here because subgroup counts per fold are uneven.

Honest caveat on the interval
-----------------------------
The per-volume values are **not independent**: any two volumes in different folds
were scored by models that shared most of their training data, and any two in the
same fold were scored by the identical model. The reported interval therefore
understates the true uncertainty. It is a useful summary of spread, not a valid
frequentist CI for "the performance of this method on a new cohort". Treat
non-overlapping intervals between subgroups as suggestive, not as a test. If you
need a test, the defensible options are a paired comparison against the sinc
baseline within volume (which cancels most of the shared-model dependence) or a
permutation scheme over subjects -- `--paired` does the former.

Usage
-----
    python -m sr.aggregate_cv --eval_root /scratch/$USER/eval/sr_2mm_to_1mm_cv \\
        --out /scratch/$USER/eval/cv_summary

    # explicit list instead of a glob
    python -m sr.aggregate_cv --csv f0/per_volume.csv f1/per_volume.csv --out /tmp/cv
"""

import argparse
import csv
import glob
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    from scipy import stats as _st
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

LEAD = ('volume', 'stem', 'subject', 'session', 'subgroup', 'age', 'age_num',
        'fold')
# Identity columns that are nonetheless numeric: kept out of the metric tables
# (they are not metrics) but coerced to float so they can be binned.
NUMERIC_LEAD = ('age_num',)


def read_rows(paths):
    rows = []
    for p in paths:
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                out = {}
                for k, v in r.items():
                    if k in NUMERIC_LEAD:
                        try:
                            out[k] = float(v)
                        except (TypeError, ValueError):
                            out[k] = None
                    elif k in LEAD:
                        out[k] = v
                    else:
                        try:
                            out[k] = float(v)
                        except (TypeError, ValueError):
                            out[k] = float('nan')
                out.setdefault('subgroup', 'unknown')
                if not out.get('fold'):
                    # Fall back to the containing directory name.
                    out['fold'] = os.path.basename(os.path.dirname(os.path.abspath(p)))
                out['_src'] = p
                rows.append(out)
    return rows


def ci95(vals):
    """Mean and half-width of a 95% interval. See the caveat in the docstring."""
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    n = v.size
    if n == 0:
        return float('nan'), float('nan'), 0
    if n == 1:
        return float(v[0]), float('nan'), 1
    sem = v.std(ddof=1) / np.sqrt(n)
    if _HAVE_SCIPY:
        crit = _st.t.ppf(0.975, n - 1)
    else:
        crit = 1.96
    return float(v.mean()), float(crit * sem), n


def paired_delta(rows, a_key, b_key):
    """Within-volume difference a - b. Cancels the shared-model dependence that
    makes the raw per-volume interval optimistic, so this is the comparison to
    quote when claiming the model beats a baseline."""
    d = []
    for r in rows:
        if a_key in r and b_key in r and np.isfinite(r[a_key]) and np.isfinite(r[b_key]):
            d.append(r[a_key] - r[b_key])
    return ci95(d)


def main(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--eval_root', help='Directory containing per-fold eval dirs; '
                                         'globs */per_volume.csv beneath it')
    src.add_argument('--csv', nargs='+', help='Explicit per_volume.csv paths')
    p.add_argument('--out', required=True, help='Output directory')
    p.add_argument('--metrics', nargs='*',
                   default=['pred_psnr', 'pred_ssim', 'pred_mae',
                            'sinc_psnr', 'sinc_ssim', 'sinc_mae',
                            'pred_hf_energy_pred', 'sinc_hf_energy_pred',
                            'hf_energy_target'],
                   help='Metrics to summarise (default: the useful subset)')
    p.add_argument('--all_metrics', action='store_true',
                   help='Summarise every numeric column found')
    p.add_argument('--paired', action='store_true', default=True,
                   help='Report within-volume pred-minus-sinc deltas')
    p.add_argument('--subgroup_name', default='anatomical weighting (confounded with age)')
    p.add_argument('--age_bins', type=int, default=3,
                   help='Number of equal-count (quantile) age bins for the '
                        'secondary age breakdown. 0 or 1 disables it. Ignored when '
                        '--age_bin_edges is given.')
    p.add_argument('--age_bin_edges', type=float, nargs='+', default=None,
                   help='Explicit bin edges in MONTHS, e.g. --age_bin_edges 0 3 6 '
                        '12 24. Fixed edges are usually more interpretable than '
                        'quantiles for a developmental cohort, since they line up '
                        'with real milestones rather than with your sampling.')
    p.add_argument('--age_unit', default='months',
                   help='Label only; the age field is parsed verbatim')
    args = p.parse_args(argv)

    if args.eval_root:
        paths = sorted(glob.glob(os.path.join(args.eval_root, '*', 'per_volume.csv')))
        if not paths:
            paths = sorted(glob.glob(os.path.join(args.eval_root, '**',
                                                  'per_volume.csv'), recursive=True))
    else:
        paths = args.csv
    if not paths:
        sys.exit('no per_volume.csv found')

    print('pooling %d fold file(s):' % len(paths))
    for q in paths:
        print('  %s' % q)

    rows = read_rows(paths)
    if not rows:
        sys.exit('no rows read')

    # Each volume must appear exactly once across folds, or the pooling is wrong.
    seen = {}
    for r in rows:
        key = r.get('stem') or r['volume']
        seen.setdefault(key, []).append(r['fold'])
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if dupes:
        print('\nWARNING: %d volume(s) appear in more than one fold, e.g. %s.'
              % (len(dupes), list(dupes.items())[:3]), file=sys.stderr)
        print('         Either the folds overlap or you globbed the same eval dir '
              'twice. The pooled mean will double-count these.', file=sys.stderr)

    numeric = [k for k in rows[0] if k not in LEAD and k != '_src']
    metrics = numeric if args.all_metrics else [m for m in args.metrics if m in numeric]
    subgroups = sorted({r['subgroup'] for r in rows})
    folds = sorted({r['fold'] for r in rows})

    os.makedirs(args.out, exist_ok=True)

    print('\n%d volumes, %d folds, subgroups %s'
          % (len(rows), len(folds), {s: sum(1 for r in rows if r['subgroup'] == s)
                                     for s in subgroups}))

    # ---- pooled table --------------------------------------------------------
    groups = OrderedDict()
    groups['ALL'] = rows
    if len(subgroups) > 1:
        for s in subgroups:
            groups[s] = [r for r in rows if r['subgroup'] == s]

    out_csv = os.path.join(args.out, 'cv_summary.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subgroup', 'metric', 'n', 'mean', 'ci95_halfwidth',
                    'std', 'median', 'min', 'max'])
        for gname, sel in groups.items():
            for m in metrics:
                vals = [r[m] for r in sel if m in r]
                mu, hw, n = ci95(vals)
                if n == 0:
                    continue
                v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
                w.writerow([gname, m, n, '%.6f' % mu, '%.6f' % hw,
                            '%.6f' % v.std(ddof=1) if n > 1 else '',
                            '%.6f' % np.median(v), '%.6f' % v.min(),
                            '%.6f' % v.max()])

    for gname, sel in groups.items():
        print('\n=== %s (n=%d) ===' % (gname, len(sel)))
        print('%-24s %>6s %>22s'.replace('>', '') % ('metric', 'n', 'mean [95% interval]'))
        for m in metrics:
            mu, hw, n = ci95([r[m] for r in sel if m in r])
            if n == 0:
                continue
            if np.isfinite(hw):
                print('%-24s %6d   %8.4f [%.4f, %.4f]'
                      % (m, n, mu, mu - hw, mu + hw))
            else:
                print('%-24s %6d   %8.4f' % (m, n, mu))

    # ---- paired comparison against the free baseline -------------------------
    if args.paired:
        print('\n' + '=' * 66)
        print('Within-volume gain over sinc interpolation (paired)')
        print('=' * 66)
        print('%-14s %>5s %>34s'.replace('>', '')
              % ('subgroup', 'n', 'delta PSNR (dB), mean [95% int]'))
        prows = [['subgroup', 'metric', 'n', 'mean_delta', 'ci95_halfwidth']]
        for gname, sel in groups.items():
            for metric, a, b in (('psnr', 'pred_psnr', 'sinc_psnr'),
                                 ('ssim', 'pred_ssim', 'sinc_ssim')):
                mu, hw, n = paired_delta(sel, a, b)
                if n == 0:
                    continue
                prows.append([gname, metric, n, '%.6f' % mu, '%.6f' % hw])
                if metric == 'psnr':
                    flag = ''
                    if np.isfinite(hw) and (mu - hw) <= 0 < (mu + hw):
                        flag = '   <- interval includes zero'
                    print('%-14s %5d   %+8.3f [%+.3f, %+.3f]%s'
                          % (gname, n, mu, mu - hw, mu + hw, flag))
        with open(os.path.join(args.out, 'cv_paired.csv'), 'w', newline='') as f:
            csv.writer(f).writerows(prows)

    # ---- age breakdown -------------------------------------------------------
    # In this cohort weighting is confounded with age, so an age effect and a
    # weighting effect cannot be separated. This table is still worth having: if
    # performance degrades monotonically with age WITHIN a weighting, that points
    # at age rather than protocol.
    ages = [r for r in rows
            if isinstance(r.get('age_num'), float) and np.isfinite(r['age_num'])]
    use_ages = len(ages) >= 6 and (args.age_bin_edges or args.age_bins > 1)
    if use_ages:
        vals = np.array([r['age_num'] for r in ages], dtype=float)
        if args.age_bin_edges:
            edges = np.array(sorted(float(e) for e in args.age_bin_edges),
                             dtype=float)
            lo, hi = vals.min(), vals.max()
            if lo < edges[0] or hi > edges[-1]:
                print('\nNOTE: ages span %.2f-%.2f %s but --age_bin_edges cover '
                      '%.2f-%.2f; volumes outside that range are omitted from the '
                      'age table.' % (lo, hi, args.age_unit, edges[0], edges[-1]),
                      file=sys.stderr)
            binning = 'fixed edges (%s)' % args.age_unit
        else:
            edges = np.quantile(vals, np.linspace(0, 1, args.age_bins + 1))
            binning = 'equal-count quantiles'
        edges = edges.astype(float)
        edges[-1] += 1e-9
        n_bins = len(edges) - 1
        print('\n' + '=' * 66)
        print('By age bin in %s -- %s' % (args.age_unit, binning))
        print('=' * 66)
        print('%-18s %-16s %>5s %>12s %>14s'.replace('>', '')
              % ('age (%s)' % args.age_unit, 'weighting', 'n', 'pred_psnr',
                 'delta vs sinc'))
        arows = [['age_lo', 'age_hi', 'subgroup', 'n', 'pred_psnr',
                  'delta_psnr', 'ci95']]
        for b in range(n_bins):
            sel_b = [r for r in ages if edges[b] <= r['age_num'] < edges[b + 1]]
            if not sel_b:
                continue
            for sg in ['ALL'] + (subgroups if len(subgroups) > 1 else []):
                sel = sel_b if sg == 'ALL' else [r for r in sel_b
                                                 if r['subgroup'] == sg]
                if not sel:
                    continue
                mu, _, n = ci95([r.get('pred_psnr', float('nan')) for r in sel])
                dmu, dhw, _ = paired_delta(sel, 'pred_psnr', 'sinc_psnr')
                arows.append(['%.3f' % edges[b], '%.3f' % edges[b + 1], sg, n,
                              '%.4f' % mu, '%.4f' % dmu,
                              '%.4f' % dhw if np.isfinite(dhw) else ''])
                print('%-18s %-16s %5d %12.3f %+14.3f'
                      % ('%.2f-%.2f' % (edges[b], edges[b + 1] - 1e-9),
                         sg, n, mu, dmu))
        with open(os.path.join(args.out, 'cv_by_age.csv'), 'w', newline='') as f:
            csv.writer(f).writerows(arows)
        print('\nAge is read from the age field of --name_schema (months, '
              'decimal, e.g. 2.2). The token is carried through verbatim and only '
              'the leading number is parsed.')
        print('Weighting is confounded with age in this cohort, so the ALL rows '
              'here mix the two effects; compare within a weighting instead.')

    # ---- per-fold spread, to expose an unstable fold -------------------------
    print('\nper-fold mean pred_psnr (a wild outlier here means one fold diverged):')
    for fd in folds:
        sel = [r for r in rows if r['fold'] == fd]
        mu, _, n = ci95([r.get('pred_psnr', float('nan')) for r in sel])
        print('  fold %-8s n=%-3d %.3f' % (fd, n, mu))

    print('\nwrote %s and cv_paired.csv' % out_csv)
    print('\nSubgroup axis: %s.' % args.subgroup_name)
    print('Interval caveat: per-volume scores are not independent across folds '
          '(models share training data). Quote the paired deltas for claims '
          'about beating the baseline; see this module\'s docstring.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
