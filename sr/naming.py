#!/usr/bin/env python3
"""
naming.py
=========
Filename and identity helpers shared across the SR pipeline. Deliberately
dependency-light (stdlib only) so that the pre-flight and reporting tools can
import it without pulling in torch or mamba_ssm.

Filename schema
---------------
Underscore is a reserved delimiter, so filenames are parsed **positionally**
rather than by pattern matching. The default schema is

    <id>_<session>_<age>_<weighting>.nii.gz      e.g.  12345_02_2.2_t1w.nii.gz
     |     |        |     |
     |     |        |     `-- weighting: t1w / t2w  (the stratification axis)
     |     |        `-------- age:       AGE IN MONTHS, decimal (2.2 = 2.2 months)
     |     `----------------- session:   visit number within the subject
     `----------------------- id:        the subject

Note that the age field contains a '.', which is why `strip_ext` refuses to fall
back to `os.path.splitext` -- see its docstring.

Positional parsing is used because it cannot go wrong in the ways regexes do: a
subject id that happens to contain the substring `t1w`, or an extra field
appearing mid-schema, both silently corrupt a regex-based parse and neither
silently corrupts this one. Change the layout with `--name_schema` (a
comma-separated field list) rather than by editing patterns.

`validate_stems` refuses to guess: any filename whose token count does not match
the schema is reported as an error rather than parsed partially, because a
mis-tokenised name produces a wrong grouping key, which produces a subject leak
that nothing downstream can detect.

Every accessor also takes an optional explicit `regex=` override for the cases
positional parsing genuinely cannot express (mixed-convention cohorts, BIDS
names). When `regex` is None the schema is used.

`lst_files` reproduces `utils.NiftiDataset.lstFiles` exactly, including its
numeric sort. That sort is load-bearing: the dataloader pairs images with labels
by sorted *position*, not by name, so anything reasoning about pairing has to sort
identically or it will draw the wrong conclusions.
"""

import json
import os
import re

NIFTI_EXTS = ('.nii.gz', '.nii', '.mgz', '.mha', '.mhd', '.nrrd')

DELIM = '_'

# Field order in the filename. Fields named here are what the accessors below
# look for; the names 'id', 'session', 'age' and 'weighting' are the ones the
# pipeline understands. Extra fields are parsed and ignored.
DEFAULT_NAME_SCHEMA = 'id,session,age,weighting'

# Which fields compose each grouping level.
SUBJECT_FIELDS = ('id',)
SESSION_FIELDS = ('id', 'session')

# Escape hatches, used only when a caller passes an explicit regex.
FALLBACK_SUBJECT_REGEX = r'^[^_]+'
FALLBACK_SUBGROUP_REGEX = r'(t1w|t2w)'

# Kept as module attributes so existing callers that referenced them still work;
# `None` now means "derive from --name_schema", which is the preferred path.
DEFAULT_SUBJECT_REGEX = None
DEFAULT_SUBGROUP_REGEX = None
DEFAULT_SESSION_REGEX = None
DEFAULT_AGE_REGEX = None


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def strip_ext(path_or_name):
    """Remove a recognised image extension. Idempotent.

    Deliberately does NOT fall back to `os.path.splitext`. The age field contains
    a decimal point (age is in months, e.g. `2.2`), so splitext on an
    already-stripped stem splits at the LAST dot and mangles it:

        os.path.splitext('12345_01_2.2_t1w')[0]  ->  '12345_01_2'

    which silently drops the weighting token and changes the tokenisation, hence
    the grouping key. Anything unrecognised is returned unchanged instead, which
    makes the function safe to apply twice.
    """
    base = os.path.basename(path_or_name)
    low = base.lower()
    for e in NIFTI_EXTS:
        if low.endswith(e):
            return base[: -len(e)]
    return base


def _numerical_sort(value):
    parts = re.compile(r'(\d+)').split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def lst_files(path):
    """Recursive, numerically sorted image list. Mirrors NiftiDataset.lstFiles."""
    out = []
    for dirname, _, filelist in os.walk(path):
        for fn in filelist:
            if fn.startswith('.'):
                continue
            low = fn.lower()
            if low.endswith('.nii.gz') or low.endswith('.nii') or low.endswith('.mhd'):
                out.append(os.path.join(dirname, fn))
    return sorted(out, key=_numerical_sort)


def list_stems(path):
    """Sorted stems of image files directly under `path` (non-recursive)."""
    if not os.path.isdir(path):
        raise NotADirectoryError(path)
    out = []
    for fn in sorted(os.listdir(path)):
        if fn.startswith('.'):
            continue
        if any(fn.lower().endswith(e) for e in NIFTI_EXTS):
            out.append(strip_ext(fn))
    return sorted(out)


# --------------------------------------------------------------------------- #
# Schema parsing
# --------------------------------------------------------------------------- #

def schema_fields(schema=DEFAULT_NAME_SCHEMA):
    if isinstance(schema, (list, tuple)):
        return [str(f).strip() for f in schema if str(f).strip()]
    return [f.strip() for f in str(schema).split(',') if f.strip()]


def parse_stem(stem, schema=DEFAULT_NAME_SCHEMA):
    """Split `stem` on DELIM and map tokens onto the schema field names.

    Returns a dict of field -> token, plus:
       '_tokens'  the raw split
       '_n'       token count
       '_ok'      whether the token count matches the schema exactly

    Fields beyond the available tokens are ''. Callers that care about
    correctness should check '_ok' (or use validate_stems).
    """
    fields = schema_fields(schema)
    toks = str(stem).split(DELIM)
    out = {'_tokens': toks, '_n': len(toks), '_fields': fields,
           '_ok': len(toks) == len(fields)}
    for i, name in enumerate(fields):
        out[name] = toks[i] if i < len(toks) else ''
    return out


def validate_stems(stems, schema=DEFAULT_NAME_SCHEMA):
    """(ok, problems) where problems is a list of (stem, reason).

    Refuses to guess about token-count mismatches. A stem with too few tokens
    yields empty fields; one with too many silently shifts every field after the
    extra one. Both corrupt grouping, so both are errors here rather than
    warnings.
    """
    fields = schema_fields(schema)
    problems = []
    for s in stems:
        p = parse_stem(s, fields)
        if not p['_ok']:
            problems.append((s, 'has %d underscore-delimited field(s), schema '
                                'expects %d (%s)'
                             % (p['_n'], len(fields), ','.join(fields))))
            continue
        empty = [f for f in fields if not p[f]]
        if empty:
            problems.append((s, 'empty field(s): %s' % ','.join(empty)))
    return (not problems), problems


def _field(stem, field, schema, regex=None, group=1):
    """Field value from the schema, or from an explicit regex override."""
    if regex:
        m = re.search(regex, stem, flags=re.IGNORECASE)
        if not m:
            return ''
        return m.group(group) if m.groups() and group <= len(m.groups()) else m.group(0)
    return parse_stem(stem, schema).get(field, '')


# --------------------------------------------------------------------------- #
# Identity accessors
# --------------------------------------------------------------------------- #

def subject_of(stem, regex=None, schema=DEFAULT_NAME_SCHEMA):
    """Subject id. Falls back to the whole stem if it cannot be determined.

    Falling back to the stem is the safe default: an unparseable file becomes its
    own singleton group, so it can never be silently merged with another
    subject's data and create a leak.
    """
    if regex:
        m = re.search(regex, stem)
        return m.group(0) if m else stem
    p = parse_stem(stem, schema)
    vals = [p.get(f, '') for f in SUBJECT_FIELDS]
    return DELIM.join(vals) if all(vals) else stem


def session_of(stem, regex=None, schema=DEFAULT_NAME_SCHEMA):
    """Session id: subject + visit, e.g. '12345_02' from '12345_02_06mo_t1w'.

    Grouping here lets different visits of one subject fall on opposite sides of
    a split while keeping the t1w and t2w of the SAME visit together. Two
    weightings from one visit are the same brain at the same moment, so
    separating them leaks the exact anatomy.
    """
    if regex:
        m = re.search(regex, stem)
        return m.group(0) if m else stem
    p = parse_stem(stem, schema)
    vals = [p.get(f, '') for f in SESSION_FIELDS]
    return DELIM.join(vals) if all(vals) else stem


def subgroup_of(stem, regex=None, default='unknown', normalize=True,
                schema=DEFAULT_NAME_SCHEMA):
    """Weighting label (t1w / t2w), lowercased by default.

    Normalising matters: a cohort mixing `_t1w` and `_T1w` filenames would
    otherwise produce two distinct strata that stratification would try to
    balance separately.
    """
    val = _field(stem, 'weighting', schema, regex)
    if not val:
        return default
    return val.lower() if normalize else val


def age_of(stem, regex=None, schema=DEFAULT_NAME_SCHEMA):
    """(age_token, age_numeric_or_None).

    In this cohort the field is AGE IN MONTHS with a decimal point, e.g. '2.2'
    meaning 2.2 months, so the numeric parse is the value you want directly.

    The token is still returned verbatim alongside it, so nothing downstream has
    to trust the parse, and the parse itself stays unit-agnostic: '2.2', '06mo'
    and '0.5y' all round-trip as tokens and yield 2.2 / 6.0 / 0.5 numerically.
    """
    tok = _field(stem, 'age', schema, regex)
    if not tok:
        return '', None
    num = re.search(r'[-+]?\d*\.?\d+', tok)
    try:
        val = float(num.group(0)) if num else None
    except ValueError:
        val = None
    return tok, val


def group_key_of(stem, level='subject', subject_regex=None, session_regex=None,
                 schema=DEFAULT_NAME_SCHEMA):
    """Grouping key at the requested level.

    'subject' -> no anatomy from one individual crosses a split (strictest)
    'session' -> visits may cross, same-visit weightings may not
    'volume'  -> no grouping at all; every file independent (leakiest)
    """
    if level == 'subject':
        return subject_of(stem, subject_regex, schema)
    if level == 'session':
        return session_of(stem, session_regex, schema)
    if level == 'volume':
        return stem
    raise ValueError('unknown grouping level %r' % level)


def describe(stem, schema=DEFAULT_NAME_SCHEMA, group_by='subject',
             subject_regex=None, subgroup_regex=None, session_regex=None,
             age_regex=None):
    """One-row summary used by the parse previews.

    Takes the same regex overrides as the accessors, so the preview shows what
    will actually be used rather than what the schema alone would give. A preview
    that disagrees with the real parse is worse than no preview.
    """
    p = parse_stem(stem, schema)
    tok, num = age_of(stem, age_regex, schema)
    return {'stem': stem, 'ok': p['_ok'], 'n_tokens': p['_n'],
            'subject': subject_of(stem, subject_regex, schema),
            'session': session_of(stem, session_regex, schema),
            'group': group_key_of(stem, group_by, subject_regex, session_regex,
                                  schema),
            'subgroup': subgroup_of(stem, subgroup_regex, schema=schema),
            'age': tok, 'age_num': num}


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

def load_name_map(split_path=None, manifest_path=None, split=None,
                  warn=None):
    """{assigned_name_without_ext: original_stem} for ONE split of a dataset.

    `build_sr_dataset.py` renumbers volumes to 0.nii.gz, 1.nii.gz, ... so the
    dataloader's positional pairing cannot mispair anything. That erases the
    original filename, and with it the weighting and age. The manifest carries the
    mapping back.

    **The split matters.** Numbering restarts at 0 in every split, so `train/0`
    and `test/0` are different volumes with the same assigned name. Merging all
    splits into one dict silently resolves `0` to whichever split appeared last in
    the manifest, which mislabels every other split. The split is taken from
    `split` if given, else inferred from the last path component of `split_path`
    ('train' / 'val' / 'test').

    Falls back to merging all splits only when the split cannot be determined, and
    warns in that case, because the result is then only trustworthy if assigned
    names happen to be globally unique (i.e. --keep_names was used).

    Returns {} if no manifest exists, so callers can fall back to on-disk names.
    """
    if manifest_path is None:
        if split_path is None:
            return {}
        manifest_path = os.path.join(
            os.path.dirname(os.path.abspath(split_path)), 'manifest.json')
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path) as f:
        man = json.load(f)
    splits = man.get('splits', {})

    if split is None and split_path:
        split = os.path.basename(os.path.normpath(os.path.abspath(split_path)))

    if split in splits:
        groups = [splits[split]]
    else:
        groups = list(splits.values())
        names = [strip_ext(e.get('assigned_name', ''))
                 for g in groups for e in g]
        if len(names) != len(set(names)) and warn:
            warn('load_name_map: split %r not found in %s, and assigned names '
                 'collide across splits (numbering restarts per split). Labels '
                 'resolved from this map are unreliable.' % (split, manifest_path))

    out = {}
    for entries in groups:
        for e in entries:
            nm = strip_ext(e.get('assigned_name', ''))
            out[nm] = e.get('stem', nm)
    return out
