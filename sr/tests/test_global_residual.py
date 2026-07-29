"""Verify the --global_residual long skip.

The claim being tested is specific and load-bearing: with `global_residual=True`
and `res_scale` initialised to zero, an untrained GAMBAS must reproduce its
input EXACTLY. That is what makes epoch-0 validation equal the sinc baseline
instead of ~10 dB below it, so if it is even slightly wrong the whole reason for
the change evaporates.

Runs on CPU with a stubbed `mamba_ssm`, so it needs torch but no GPU and no
compiled selective-scan kernel. The stub only has to be shape-preserving --
the test never depends on what Mamba computes, only that the residual path
bypasses it.

    python -m sr.tests.test_global_residual
"""
import sys
import types

import numpy as np


def _stub_mamba():
    """Install a shape-preserving stand-in for mamba_ssm.Mamba."""
    if 'mamba_ssm' in sys.modules:
        return
    import torch.nn as nn

    class _Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kw):
            super().__init__()
            self.proj = nn.Linear(d_model, d_model)

        def forward(self, x):          # (B, L, D) -> (B, L, D)
            return self.proj(x)

    mod = types.ModuleType('mamba_ssm')
    mod.Mamba = _Mamba
    sys.modules['mamba_ssm'] = mod


def main():
    _stub_mamba()
    import torch
    from models.mamba_modules3D import GAMBAS

    torch.manual_seed(0)
    failures = []

    def check(name, ok, detail=''):
        print('  %-58s %s%s' % (name, 'PASS' if ok else 'FAIL',
                                '' if ok else '  <-- ' + detail))
        if not ok:
            failures.append(name)

    # A 32^3 patch gives an 8^3 bottleneck, small enough to run on CPU quickly.
    x = torch.randn(1, 1, 32, 32, 32)

    print('\n1. identity at initialisation (the whole point)')
    net = GAMBAS(input_dim=1, img_size=(32, 32), output_dim=1, global_residual=True)
    net.eval()
    with torch.no_grad():
        y = net(x)
    check('res_scale initialised to exactly 0',
          float(net.res_scale.abs().max()) == 0.0,
          'res_scale=%r' % net.res_scale.data)
    check('output shape matches input', y.shape == x.shape, str(y.shape))
    maxdiff = float((y - x).abs().max())
    check('output IS the input (max|y-x| == 0)', maxdiff == 0.0,
          'max abs diff = %.3e' % maxdiff)

    print('\n2. the residual branch is live, not disconnected')
    with torch.no_grad():
        net.res_scale.fill_(1.0)
        y1 = net(x)
    moved = float((y1 - x).abs().max())
    check('nonzero res_scale changes the output', moved > 1e-6,
          'max abs diff = %.3e' % moved)
    # Gradient must reach res_scale, otherwise it can never leave zero and the
    # network is permanently frozen at the identity -- a silent, total failure.
    net.res_scale.data.zero_()
    net.train()
    loss = (net(x) - torch.randn_like(x)).abs().mean()
    loss.backward()
    g = net.res_scale.grad
    check('gradient reaches res_scale when it is 0',
          g is not None and float(g.abs().max()) > 0,
          'grad = %r' % (None if g is None else g.item()))

    print('\n3. published architecture still available and different')
    torch.manual_seed(0)
    plain = GAMBAS(input_dim=1, img_size=(32, 32), output_dim=1, global_residual=False)
    plain.eval()
    with torch.no_grad():
        yp = plain(x)
    check('global_residual=False has no res_scale',
          not hasattr(plain, 'res_scale'))
    check('global_residual=False does NOT return the input',
          float((yp - x).abs().max()) > 1e-6)
    check('plain output stays in Tanh range [-1, 1]',
          float(yp.abs().max()) <= 1.0 + 1e-6, '%.4f' % float(yp.abs().max()))

    print('\n4. mismatched channel counts are refused, not silently broadcast')
    try:
        GAMBAS(input_dim=3, img_size=(32, 32), output_dim=1, global_residual=True)
        check('input_dim != output_dim raises', False, 'no exception')
    except ValueError:
        check('input_dim != output_dim raises ValueError', True)

    print('\n5. warm-start reconciliation tolerates the new parameter')
    from sr.checkpoint_utils import match_state_dict
    # A checkpoint from the published (non-residual) model lacks res_scale.
    # match_state_dict works on {name: shape} maps and returns a single report.
    old_sd = plain.state_dict()
    new = GAMBAS(input_dim=1, img_size=(32, 32), output_dim=1, global_residual=True)
    shapes = lambda sd: {k: tuple(v.shape) for k, v in sd.items()}   # noqa: E731
    res = match_state_dict(shapes(old_sd), shapes(new.state_dict()))
    check('coverage stays above the 0.5 --init_min_coverage floor',
          res['coverage_guard'] > 0.5,
          'coverage_guard = %.4f' % res['coverage_guard'])
    check('res_scale reported missing (so it stays at 0)',
          'res_scale' in res['missing'], str(res['missing'][:5]))
    check('nothing reported as a shape mismatch',
          not res['shape_mismatch'], str(res['shape_mismatch'][:3]))
    matched = {mk: old_sd[ck] for ck, mk in res['rename'].items()}
    missing = new.load_state_dict(matched, strict=False)
    check('only res_scale is missing after load',
          list(missing.missing_keys) == ['res_scale'],
          str(list(missing.missing_keys)[:5]))
    # And after that load the net must STILL be the identity.
    new.eval()
    with torch.no_grad():
        yn = new(x)
    d = float((yn - x).abs().max())
    check('warm-started net is still exactly the identity', d == 0.0,
          'max abs diff = %.3e' % d)

    print('\n%s' % ('-' * 70))
    if failures:
        print('%d CHECK(S) FAILED: %s' % (len(failures), ', '.join(failures)))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
