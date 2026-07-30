#!/usr/bin/env python3
"""
make_folds.py
=============
Build subject-grouped, subgroup-stratified splits and cross-validation folds,
and write them to a `folds.json` that `build_sr_dataset.py` can materialise.

Two hard requirements, in priority order:

  1. **Subject grouping is inviolable.** Every volume from one subject goes to
     exactly one side of every split. If a subject has both a T1w and a T2w
     scan, or repeat sessions, they travel together. Violating this leaks the
     same anatomy into train and test, and for a super-resolution task -- where
     the target IS the input's own anatomy at higher frequency -- that leak is
     catastrophic and produces beautiful, meaningless test metrics.

  2. **Subgroup balance is the objective, subject to (1).** Because groups are
     atomic, exact per-fold subgroup counts are usually unachievable. We use the
     standard greedy heuristic (as in sklearn's StratifiedGroupKFold): process
     groups largest-first, and assign each to whichever fold minimises the
     spread of per-subgroup counts across folds afterwards.

On the confound in this dataset
-------------------------------
The subgroup label here is the anatomical weighting (T1w / T2w), which in this
cohort is confounded with age -- the T2w scans are the younger infants. So the
stratum is a *protocol-and-age subgroup*, not a contrast effect that can be
separated from an age effect. Nothing in this script can disentangle them, and
the reports it produces label the axis accordingly rather than implying two
independent factors. Separating them needs subjects scanned with both weightings
at matched ages.

Modes
-----
`--mode single`  one development split. Use this to choose patch size, epoch
                 count, lambda_adv -- then FREEZE those and go to CV. Selecting
                 hyperparameters on CV folds and then reporting CV numbers is
                 selection bias.

`--mode cv`      K folds. Each volume is tested exactly once across the K runs,
                 giving one held-out prediction per volume, which is the only way
                 to get a per-subgroup estimate with a usable interval at n~50.

    --val_mode rotate  test = fold k, val = fold k+1, train = the rest.
                       (default) Every volume serves as test once and val once.
                       With K=5: 20/20/60. Costs training data but selects the
                       best checkpoint on ~10 volumes instead of ~5.
    --val_mode carve   test = fold k, val = a --val_frac slice carved out of the
                       remainder. With K=5: 20/10/70. More training data, noisier
                       checkpoint selection.

Filenames are parsed positionally as <id>_<session>_<age>_<weighting>, split on
underscores; change the order with --name_schema. Names that do not tokenise to
the schema are a hard error, not a warning: a mis-parsed name yields a wrong
grouping key, hence a subject leak nothing downstream can detect.

Usage
-----
    # development split matching the recommendation for 38 t1w / 14 t2w
    python -m sr.make_folds --sim_dir $DATA --method kspace --out folds_dev.json \\
        --mode single --test_counts t1w=8,t2w=3 --val_counts t1w=5,t2w=2

    # 5-fold CV for the numbers you report
    python -m sr.make_folds --sim_dir $DATA --method kspace --out folds_cv.json \\
        --mode cv --k 5
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sr.naming import (DEFAULT_NAME_SCHEMA, describe, group_key_of,
                       list_stems as _list_stems, schema_fields,
                       subgroup_of as label_of, subject_of, validate_stems)


def list_stems(d):
    try:
        return _list_stems(d)
    except NotADirectoryError:
        sys.exit('not a directory: %s' % d)


def parse_counts(s):
    """'t1w=8,T2w=3' -> {'t1w': 8, 't2w': 3}

    Keys are lowercased to match `naming.subgroup_of`, which normalises labels so
    that `_t1w` and `_T1w` filenames do not become two separate strata. So both
    spellings work on the command line.
    """
    if not s:
        return None
    out = {}
    for part in s.replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            raise argparse.ArgumentTypeError('expected LABEL=N, got %r' % part)
        k, v = part.split('=', 1)
        out[k.strip().lower()] = int(v)
    return out


# --------------------------------------------------------------------------- #
# Grouped, stratified assignment
# --------------------------------------------------------------------------- #

def group_data(stems, subject_regex, subgroup_regex, group_by='subject',
               session_regex=None, schema=DEFAULT_NAME_SCHEMA):
    groups = OrderedDict()
    labels = {}
    for s in stems:
        labels[s] = label_of(s, subgroup_regex, schema=schema)
        key = group_key_of(s, group_by, subject_regex, session_regex, schema)
        groups.setdefault(key, []).append(s)
    group_strata = {g: Counter(labels[s] for s in v) for g, v in groups.items()}
    return groups, labels, group_strata


def grouped_stratified_kfold(groups, group_strata, strata, k, seed=0):
    """Greedy: largest groups first, each to the fold that minimises the summed
    per-subgroup standard deviation across folds. Deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    order = sorted(groups, key=lambda g: (-len(groups[g]),
                                          rng.random(), g))
    fold_counts = [Counter() for _ in range(k)]
    fold_sizes = [0] * k
    assign = {}

    for g in order:
        best_f, best_key = None, None
        for f in range(k):
            fold_counts[f].update(group_strata[g])
            cost = sum(float(np.std([fold_counts[i][s] for i in range(k)]))
                       for s in strata)
            for s, n in group_strata[g].items():   # undo
                fold_counts[f][s] -= n
            key = (cost, fold_sizes[f], f)
            if best_key is None or key < best_key:
                best_key, best_f = key, f
        fold_counts[best_f].update(group_strata[g])
        fold_sizes[best_f] += len(groups[g])
        assign[g] = best_f

    return assign, fold_counts


def grouped_quota_split(groups, group_strata, quotas, seed=0, taken=None):
    """Pick whole groups to fill per-subgroup `quotas` as closely as possible.

    Greedy by 'how much of the remaining need does this group satisfy, without
    overshooting'. Returns the set of chosen group ids.
    """
    rng = np.random.default_rng(seed)
    remaining = Counter(quotas)
    taken = set(taken or ())
    chosen = set()
    candidates = [g for g in groups if g not in taken]
    # Shuffle for tie-breaking, then repeatedly take the best fit.
    rng.shuffle(candidates)

    while sum(remaining.values()) > 0 and candidates:
        best_g, best_key = None, None
        for g in candidates:
            gs = group_strata[g]
            useful = sum(min(gs[s], max(0, remaining[s])) for s in gs)
            over = sum(max(0, gs[s] - max(0, remaining[s])) for s in gs)
            if useful == 0:
                continue
            # Minimise NET misfit (over - useful), not raw usefulness.
            #
            # The previous key was (-useful, over, ...), which maximised useful
            # and used overshoot only as a tie-break. That is degenerate when one
            # group is large: a 22-volume longitudinal subject scores
            # useful = min(17,8) + min(5,3) = 11, the maximum attainable, so it
            # beat every 1-volume group (useful = 1) despite overshooting by
            # (17-8) + (5-3) = 11. Real consequence: a requested 8 t1w + 3 t2w
            # test set came back as 17 t1w + 5 t2w -- all 22 volumes of a SINGLE
            # subject, making every held-out score one infant's anatomy and one
            # acquisition protocol.
            #
            # over - useful ranks that group at 11 - 11 = 0 while a single t1w
            # volume scores 0 - 1 = -1, so small groups fill the quota first and
            # a large group is taken only when it genuinely fits.
            key = (over - useful, over, -useful, len(groups[g]), g)
            if best_key is None or key < best_key:
                best_key, best_g = key, g
        if best_g is None:
            break
        chosen.add(best_g)
        candidates.remove(best_g)
        for s, n in group_strata[best_g].items():
            remaining[s] -= n

    unmet = {s: n for s, n in remaining.items() if n > 0}

    # Belt and braces: even with the corrected key, a cohort where one group is
    # most of the data can still end up dominating a split (e.g. if the quota is
    # large enough that no combination of small groups fills it). That silently
    # turns "held-out test set" into "one subject", so say so loudly rather than
    # letting it pass as a normal split.
    if chosen:
        n_tot = sum(len(groups[g]) for g in chosen)
        big = max(chosen, key=lambda g: len(groups[g]))
        share = len(groups[big]) / max(n_tot, 1)
        if share > 0.5 and len(chosen) > 1 or (len(chosen) == 1 and n_tot > 3):
            print('WARNING: group %r supplies %d of %d volumes (%.0f%%) in this '
                  'split. Held-out scores will mostly describe that one subject '
                  'and its acquisition protocol, not the cohort. Consider '
                  '--group_by session, or smaller --test_counts.'
                  % (big, len(groups[big]), n_tot, 100 * share), file=sys.stderr)
    return chosen, unmet


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def crosstab(rows, strata, title, subgroup_name):
    w = max(10, max(len(str(r[0])) for r in rows) + 2)
    print('\n%s' % title)
    head = ('%-*s' % (w, 'split')) + ''.join('%>8s'.replace('>', '') % s for s in strata) + '%8s' % 'total'
    print(head)
    print('-' * len(head))
    for name, counts in rows:
        line = '%-*s' % (w, name)
        for s in strata:
            line += '%8d' % counts.get(s, 0)
        line += '%8d' % sum(counts.values())
        print(line)
    print('(subgroup axis = %s)' % subgroup_name)


def build_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split('Modes')[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--sim_dir', help='simulate_lowres.py output dir. Reads '
                                      'lowres-<method>/ when --method is given, '
                                      'else lowres_on_hr_grid/')
    src.add_argument('--stems_from', help='Directory of volumes to read names from')
    p.add_argument('--method', default=None,
                   help="Simulation method tag, e.g. 'kspace'. Only the filenames "
                        "are read, so any of the method directories gives the same "
                        "split -- this just has to point somewhere that exists.")

    p.add_argument('--out', required=True, help='Output folds.json path')
    p.add_argument('--mode', choices=['single', 'cv'], default='cv')
    p.add_argument('--k', type=int, default=5, help='Folds, --mode cv')
    p.add_argument('--val_mode', choices=['rotate', 'carve'], default='rotate')

    p.add_argument('--group_by', choices=['subject', 'session', 'volume'],
                   default='subject',
                   help='What must stay on one side of a split. '
                        'subject: no anatomy from one individual ever crosses '
                        '(strictest, recommended). '
                        'session: longitudinal timepoints may cross, but the t1w '
                        'and t2w of the same session may not. '
                        'volume: no grouping -- maximum leakage, use only to '
                        'MEASURE the leakage by comparing against subject.')
    p.add_argument('--name_schema', default=DEFAULT_NAME_SCHEMA,
                   help='Comma-separated field order in the filename, split on '
                        'underscores. The names id, session, age and weighting are '
                        'the ones this pipeline understands; extras are ignored.')
    p.add_argument('--allow_bad_names', action='store_true',
                   help='Continue even when filenames do not match --name_schema. '
                        'Off by default: a mis-tokenised name yields a wrong '
                        'grouping key, hence an undetectable subject leak.')
    p.add_argument('--session_regex', default=None,
                   help='Override the session key (default: the id+session fields '
                        'from --name_schema)')
    p.add_argument('--subject_regex', default=None,
                   help='Override the subject key (default: the id field from '
                        '--name_schema). Use for BIDS-style names, e.g. '
                        r"'sub-[A-Za-z0-9]+'")
    p.add_argument('--subgroup_regex', default=None,
                   help='Override the subgroup label (default: the weighting field '
                        'from --name_schema)')
    p.add_argument('--subgroup_name', default='anatomical weighting (confounded with age)',
                   help='Human-readable name for the stratification axis, used in '
                        'reports so the confound is not silently dropped')

    p.add_argument('--test_frac', type=float, default=0.20)
    p.add_argument('--val_frac', type=float, default=0.15)
    p.add_argument('--test_counts', type=parse_counts, default=None,
                   help="Explicit per-subgroup test counts, e.g. 'T1w=8,T2w=3'. "
                        "Overrides --test_frac. --mode single only.")
    p.add_argument('--val_counts', type=parse_counts, default=None,
                   help="Explicit per-subgroup val counts. --mode single only.")
    p.add_argument('--seed', type=int, default=1234)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.sim_dir:
        sub = 'lowres-%s' % args.method if args.method else 'lowres_on_hr_grid'
        src = os.path.join(args.sim_dir, sub)
        if not os.path.isdir(src):
            avail = sorted(d for d in os.listdir(args.sim_dir)
                           if d.startswith('lowres')) if os.path.isdir(args.sim_dir) else []
            sys.exit('%s does not exist.%s' % (
                src, ('\nAvailable: %s -- pass --method, or --stems_from <dir>.'
                      % avail) if avail else ''))
    else:
        src = args.stems_from
    stems = list_stems(src)
    if not stems:
        sys.exit('no volumes found in %s' % src)
    print('reading names from: %s' % src)

    schema = args.name_schema
    fields = schema_fields(schema)

    # Validate BEFORE anything else. A filename that does not tokenise to the
    # schema produces a wrong grouping key, which produces a subject leak that no
    # downstream check can see.
    ok, problems = validate_stems(stems, schema)
    if not ok:
        print('\n%d filename(s) do not match --name_schema %s:'
              % (len(problems), ','.join(fields)), file=sys.stderr)
        for s, why in problems[:20]:
            print('  %-44s %s' % (s, why), file=sys.stderr)
        if len(problems) > 20:
            print('  ... and %d more' % (len(problems) - 20), file=sys.stderr)
        if not args.allow_bad_names:
            sys.exit('\nRefusing to build a split from names that cannot be parsed. '
                     'Fix the filenames, adjust --name_schema, or pass '
                     '--allow_bad_names to treat the unparseable ones as their own '
                     'singleton subjects (safe against leakage, but they will not '
                     'be stratified).')
        print('--allow_bad_names given: continuing; unparseable names fall back to '
              'singleton subjects.', file=sys.stderr)

    groups, labels, group_strata = group_data(
        stems, args.subject_regex, args.subgroup_regex, args.group_by,
        args.session_regex, schema)
    strata = sorted(set(labels.values()))
    totals = Counter(labels.values())

    # Parse preview. A wrong schema silently merges or splits subjects, which is
    # invisible in the fold tables but wrecks the split, so show the actual parse.
    n_subj = len({subject_of(s, args.subject_regex, schema) for s in stems})
    n_sess = len({group_key_of(s, 'session', args.subject_regex,
                               args.session_regex, schema) for s in stems})
    print('schema   : %s' % ','.join(fields))
    print('volumes  : %d' % len(stems))
    print('subjects : %d' % n_subj)
    print('sessions : %d' % n_sess)
    print('groups   : %d  (--group_by %s)' % (len(groups), args.group_by))
    print('subgroups: %s' % dict(totals))
    print('\nparse preview -- check these are right before going further:')
    print('  %-30s %-10s %-12s %-12s %-8s %-6s'
          % ('stem', 'subject', 'session', 'group', 'weighting', 'age'))
    for s in stems[:6]:
        d = describe(s, schema, args.group_by, args.subject_regex,
                     args.subgroup_regex, args.session_regex)
        print('  %-30s %-10s %-12s %-12s %-8s %-6s'
              % (d['stem'], d['subject'], d['session'], d['group'],
                 d['subgroup'], d['age'] or '-'))
    if len(stems) > 6:
        print('  ... (%d more)' % (len(stems) - 6))

    if n_subj == len(stems) and len(stems) > 1:
        print('\nNOTE: every volume parsed to a distinct subject. If your cohort is '
              'longitudinal or has both weightings per visit, the id field of '
              '--name_schema is not picking up what you expect.')
    if 'unknown' in strata:
        print('\nWARNING: %d volume(s) have no weighting label. Check that the '
              '"weighting" field of --name_schema points at the t1w/t2w token.'
              % totals['unknown'])

    if args.group_by != 'subject':
        reps = sum(1 for c in Counter(
            subject_of(s, args.subject_regex, schema)
            for s in stems).values() if c > 1)
        print('\n' + '!' * 72)
        print('LEAKAGE WARNING: --group_by %s' % args.group_by)
        print('!' * 72)
        print('%d subject(s) contribute more than one volume, and their volumes '
              'may now be split across train and test.' % reps)
        print('For super-resolution the target IS the input\'s own anatomy at '
              'higher frequency, so a model that has seen an individual\'s')
        print('cortical folding in training gets an advantage on that same '
              'individual at test. Reported gains will be optimistic, and the')
        print('bias grows with how much of the cohort is longitudinal.')
        print('Grouping does NOT cost you training data in K-fold CV -- each fold '
              'still trains on the same NUMBER of volumes either way; only')
        print('which volumes land in test changes. Run --group_by subject too and '
              'compare: the difference is the size of the leak.')
        print('!' * 72)
    if 'unknown' in strata:
        n_unk = totals['unknown']
        print('WARNING: %d volume(s) did not match --subgroup_regex %r and were '
              'labelled "unknown". Check your filenames.'
              % (n_unk, args.subgroup_regex))

    mixed = {g: dict(c) for g, c in group_strata.items() if len(c) > 1}
    if mixed:
        print('\nNOTE: %d subject(s) have more than one subgroup, e.g. %s.'
              % (len(mixed), list(mixed.items())[:3]))
        print('      They are kept intact (no leakage) but they constrain how '
              'evenly the subgroups can be balanced.')
    else:
        print('\nEach subject has a single subgroup, so grouping and '
              'stratification do not conflict.')

    spec = {'meta': {'args': {k: v for k, v in vars(args).items()},
                     'name_schema': fields,
                     'group_by': args.group_by,
                     'leakage_risk': ('none: no subject crosses a split'
                                      if args.group_by == 'subject' else
                                      'PRESENT: --group_by %s allows one '
                                      'subject\'s volumes to span train and '
                                      'test; reported metrics are optimistic'
                                      % args.group_by),
                     'n_volumes': len(stems), 'n_subjects': len(groups),
                     'subgroup_totals': dict(totals),
                     'strata': strata,
                     'mixed_subgroup_subjects': mixed},
            'mode': args.mode, 'folds': []}

    def counts_of(stem_list):
        return Counter(labels[s] for s in stem_list)

    # ---------------- single development split ----------------------------- #
    if args.mode == 'single':
        test_q = args.test_counts or {s: int(round(args.test_frac * totals[s]))
                                      for s in strata}
        val_q = args.val_counts or {s: int(round(args.val_frac * totals[s]))
                                    for s in strata}
        for q, nm in ((test_q, 'test'), (val_q, 'val')):
            for s in q:
                if s not in totals:
                    sys.exit('--%s_counts mentions unknown subgroup %r; known: %s'
                             % (nm, s, strata))

        test_g, unmet_t = grouped_quota_split(groups, group_strata, test_q,
                                             args.seed)
        val_g, unmet_v = grouped_quota_split(groups, group_strata, val_q,
                                            args.seed + 1, taken=test_g)
        train_g = [g for g in groups if g not in test_g and g not in val_g]

        for nm, unmet in (('test', unmet_t), ('val', unmet_v)):
            if unmet:
                print('WARNING: could not fill %s quota exactly, short by %s. '
                      'Whole subjects cannot be split, so exact counts are not '
                      'always reachable.' % (nm, unmet))

        expand = lambda gg: sorted(s for g in gg for s in groups[g])
        fold = {'fold': 'dev',
                'train': expand(train_g), 'val': expand(val_g),
                'test': expand(test_g)}
        spec['folds'].append(fold)

        crosstab([('train', counts_of(fold['train'])),
                  ('val', counts_of(fold['val'])),
                  ('test', counts_of(fold['test']))],
                 strata, 'Development split', args.subgroup_name)

    # ---------------- cross-validation ------------------------------------- #
    else:
        k = args.k
        if k < 2:
            sys.exit('--k must be >= 2')
        if k > len(groups):
            sys.exit('--k %d exceeds the number of subjects (%d)' % (k, len(groups)))
        assign, fold_counts = grouped_stratified_kfold(
            groups, group_strata, strata, k, args.seed)

        by_fold = [[] for _ in range(k)]
        for g, f in assign.items():
            by_fold[f].extend(groups[g])
        by_fold = [sorted(v) for v in by_fold]

        crosstab([('fold %d' % i, counts_of(v)) for i, v in enumerate(by_fold)],
                 strata, 'Fold composition (each fold is the TEST set once)',
                 args.subgroup_name)

        for i in range(k):
            test = list(by_fold[i])
            if args.val_mode == 'rotate':
                val = list(by_fold[(i + 1) % k])
                train = [s for j in range(k) if j not in (i, (i + 1) % k)
                         for s in by_fold[j]]
            else:
                pool_groups = [g for g in groups if assign[g] != i]
                sub = {g: groups[g] for g in pool_groups}
                sub_strata = {g: group_strata[g] for g in pool_groups}
                pool_totals = Counter(labels[s] for g in pool_groups
                                      for s in groups[g])
                q = {s: int(round(args.val_frac * pool_totals[s]))
                     for s in strata}
                val_g, _ = grouped_quota_split(sub, sub_strata, q,
                                               args.seed + 100 + i)
                val = sorted(s for g in val_g for s in groups[g])
                train = sorted(s for g in pool_groups if g not in val_g
                               for s in groups[g])
            spec['folds'].append({'fold': i, 'train': sorted(train),
                                  'val': sorted(val), 'test': sorted(test)})

        # Integrity checks -- cheap, and they catch a bad heuristic immediately.
        seen_test = Counter()
        for f in spec['folds']:
            seen_test.update(f['test'])
            gk = lambda v: group_key_of(v, args.group_by,
                                        args.subject_regex, args.session_regex,
                                        schema)
            s_tr = {gk(s) for s in f['train']}
            s_va = {gk(s) for s in f['val']}
            s_te = {gk(s) for s in f['test']}
            assert not (s_tr & s_te), 'fold %s: subject leak train/test' % f['fold']
            assert not (s_tr & s_va), 'fold %s: subject leak train/val' % f['fold']
            assert not (s_va & s_te), 'fold %s: subject leak val/test' % f['fold']
            assert set(f['train']) | set(f['val']) | set(f['test']) == set(stems), \
                'fold %s does not cover every volume' % f['fold']
        bad = {s: c for s, c in seen_test.items() if c != 1}
        assert not bad, 'volumes tested != once: %s' % list(bad.items())[:5]
        print('\nintegrity: no %s spans two splits in any fold; every volume '
              'is tested exactly once.' % args.group_by)

        sizes = [(len(f['train']), len(f['val']), len(f['test']))
                 for f in spec['folds']]
        print('per-fold train/val/test sizes: %s' % sizes)

    with open(args.out, 'w') as f:
        json.dump(spec, f, indent=2)
    print('\nwrote %s (%d fold(s))' % (args.out, len(spec['folds'])))
    print('\nMaterialise a fold with:')
    print('  python -m sr.build_sr_dataset --sim_dir %s --out_root <DIR> '
          '--folds_json %s --fold %s --link'
          % (args.sim_dir or '<SIM_DIR>', args.out, spec['folds'][0]['fold']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
