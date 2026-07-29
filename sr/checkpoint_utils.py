#!/usr/bin/env python3
"""
checkpoint_utils.py
===================
Checkpoint key/shape reconciliation, expressed over plain `{key: shape}` mappings
so it can be unit-tested without torch or a GPU.

Why this exists
---------------
Warm-starting from the released GAMBAS weights is the single highest-value thing
you can do at n~50 volumes, but a naive `load_state_dict` fails or silently
half-loads for three separate reasons:

  1. **`module.` prefixes.** The repo saves `net.module.cpu().state_dict()` when
     GPUs are present, so whether keys carry a `module.` prefix depends on how the
     checkpoint was produced, not on how you are loading it.
  2. **InstanceNorm buffers.** `InstanceNorm3d` with `affine=False,
     track_running_stats=False` has no `running_mean` / `running_var`. Older
     checkpoints may carry them; `base_model.__patch_instance_norm_state_dict`
     exists precisely to strip them. Loading them back is an error.
  3. **Shape mismatches.** Any difference in `ngf`, `input_nc` or `output_nc`
     between the checkpoint and your model produces per-tensor shape conflicts.
     `strict=False` alone hides these: the tensor is simply not loaded, and you
     get a partially random network with no warning.

`match_state_dict` resolves all three and, crucially, **reports coverage**. A
warm-start that loads 4% of the generator is worse than no warm-start, because it
looks like it worked. The caller is expected to refuse to proceed below a
coverage threshold unless explicitly overridden.
"""

BUFFER_SUFFIXES = ('running_mean', 'running_var', 'num_batches_tracked')


def strip_module_prefix(keys):
    """Return (mapping old->new, changed). Removes a uniform leading 'module.'."""
    keys = list(keys)
    if keys and all(k.startswith('module.') for k in keys):
        return {k: k[len('module.'):] for k in keys}, True
    return {k: k for k in keys}, False


def drop_norm_buffers(shapes):
    """Drop InstanceNorm running statistics, which the model does not define."""
    return {k: v for k, v in shapes.items()
            if not any(k.endswith(s) for s in BUFFER_SUFFIXES)}


def match_state_dict(ckpt_shapes, model_shapes):
    """Reconcile a checkpoint against a model.

    Both arguments are `{param_name: shape_tuple}`.

    Returns a dict with:
      rename          {ckpt_key: model_key} for the keys that should be loaded
      loaded          model keys that will be populated
      shape_mismatch  [(model_key, ckpt_shape, model_shape)]
      unexpected      ckpt keys with no counterpart in the model
      missing         model keys left at their initialised values
      dropped_buffers ckpt keys removed as norm buffers
      stripped_prefix whether a uniform 'module.' prefix was removed
      coverage        fraction of model *parameters* (by count of tensors) loaded
      coverage_numel  fraction by element count, when shapes are numeric
    """
    ren, stripped = strip_module_prefix(ckpt_shapes.keys())
    renamed = {ren[k]: v for k, v in ckpt_shapes.items()}

    before = set(renamed)
    renamed = drop_norm_buffers(renamed)
    dropped = sorted(before - set(renamed))

    loaded, shape_mismatch, unexpected = [], [], []
    rename = {}
    for k, shp in renamed.items():
        if k not in model_shapes:
            unexpected.append(k)
        elif tuple(shp) != tuple(model_shapes[k]):
            shape_mismatch.append((k, tuple(shp), tuple(model_shapes[k])))
        else:
            loaded.append(k)
            rename[('module.' + k) if stripped else k] = k

    missing = sorted(set(model_shapes) - set(loaded))

    def numel(shape):
        n = 1
        for d in shape:
            n *= int(d)
        return n

    total_t = len(model_shapes) or 1
    try:
        total_n = sum(numel(s) for s in model_shapes.values()) or 1
        loaded_n = sum(numel(model_shapes[k]) for k in loaded)
        cov_n = loaded_n / total_n
    except (TypeError, ValueError):
        cov_n = float('nan')

    cov_t = len(loaded) / total_t
    # Guard on the stricter of the two. They diverge exactly in the case that
    # matters: a handful of large tensors failing to match reads as high
    # tensor-coverage but leaves most of the actual weights random.
    cov_guard = cov_t if cov_n != cov_n else min(cov_t, cov_n)  # nan-safe

    return {'rename': rename,
            'loaded': sorted(loaded),
            'shape_mismatch': sorted(shape_mismatch),
            'unexpected': sorted(unexpected),
            'missing': missing,
            'dropped_buffers': dropped,
            'stripped_prefix': stripped,
            'coverage': cov_t,
            'coverage_numel': cov_n,
            'coverage_guard': cov_guard}


def format_report(res, tag='G', max_list=5):
    """Human-readable summary of match_state_dict, for the training log."""
    lines = ['warm-start %s: loaded %d/%d tensors (%.1f%% of tensors, %.1f%% of '
             'parameters)'
             % (tag, len(res['loaded']), len(res['loaded']) + len(res['missing']),
                100 * res['coverage'], 100 * res['coverage_numel'])]
    if res['stripped_prefix']:
        lines.append("  removed a uniform 'module.' prefix from checkpoint keys")
    if res['dropped_buffers']:
        lines.append('  dropped %d InstanceNorm buffer(s), e.g. %s'
                     % (len(res['dropped_buffers']),
                        res['dropped_buffers'][:max_list]))
    if res['shape_mismatch']:
        lines.append('  %d SHAPE MISMATCH(es) -- these stay randomly initialised:'
                     % len(res['shape_mismatch']))
        for k, a, b in res['shape_mismatch'][:max_list]:
            lines.append('    %-50s ckpt %s vs model %s' % (k, a, b))
        if len(res['shape_mismatch']) > max_list:
            lines.append('    ... and %d more'
                         % (len(res['shape_mismatch']) - max_list))
        lines.append('  A shape mismatch means the checkpoint was trained with '
                     'different --ngf / --input_nc / --output_nc.')
    if res['unexpected']:
        lines.append('  %d checkpoint key(s) unused, e.g. %s'
                     % (len(res['unexpected']), res['unexpected'][:max_list]))
    if res['missing']:
        lines.append('  %d model key(s) NOT initialised from the checkpoint, e.g. %s'
                     % (len(res['missing']), res['missing'][:max_list]))
    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# Subgroup-balanced sampling weights
# --------------------------------------------------------------------------- #

def balance_weights(labels, power=1.0, cap=None):
    """Per-sample sampling weights that (partially) equalise subgroup frequency.

    `labels` is a list of subgroup labels, one per training volume.

    power = 0.0  natural frequency (no reweighting)
    power = 1.0  full balance: every subgroup contributes equally in expectation
    power = 0.5  square-root balance, the usual compromise -- it removes most of
                 the imbalance while limiting how often any single volume from a
                 small subgroup is revisited

    Why not always full balance: with 14 t2w volumes against 38 t1w, full balance
    revisits each t2w volume ~2.7x as often. Random cropping means each visit is a
    different patch, so the cost is much lower than with whole-image training, but
    it is not zero -- the model sees the same 14 individuals' anatomy more often.
    `cap` optionally clips the max/min weight ratio as a further guard.

    Returns (weights, info) where info reports counts and the effective
    post-weighting share of each subgroup, so you can verify what you asked for.
    """
    from collections import Counter
    counts = Counter(labels)
    n = len(labels)
    if n == 0:
        return [], {'counts': {}, 'expected_share': {}}

    raw = {}
    for lab, c in counts.items():
        raw[lab] = (1.0 / c) ** float(power) if c > 0 else 0.0
    if cap is not None and raw:
        lo = min(v for v in raw.values() if v > 0)
        raw = {k: min(v, lo * float(cap)) for k, v in raw.items()}

    weights = [raw[l] for l in labels]
    total = sum(weights) or 1.0
    share = {}
    for lab in counts:
        share[lab] = sum(w for w, l in zip(weights, labels) if l == lab) / total

    info = {'counts': dict(counts),
            'natural_share': {k: v / n for k, v in counts.items()},
            'expected_share': share,
            'power': float(power),
            'weight_per_label': raw}
    return weights, info


def format_balance(info):
    lines = ['subgroup sampling (--balance_power %.2f):' % info.get('power', 0.0)]
    lines.append('  %-10s %6s %14s %14s' % ('subgroup', 'n', 'natural share',
                                            'sampled share'))
    for lab in sorted(info.get('counts', {})):
        lines.append('  %-10s %6d %13.1f%% %13.1f%%'
                     % (lab, info['counts'][lab],
                        100 * info['natural_share'][lab],
                        100 * info['expected_share'][lab]))
    return '\n'.join(lines)
