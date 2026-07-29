#!/usr/bin/env python3
"""
sr_options.py
=============
Option set for retraining GAMBAS as a 2 mm -> 1 mm super-resolution model.

This subclasses the repo's `TrainOptions` so every existing flag still works and
`models.get_option_setter` still injects the per-model arguments (`--lambda_A`,
`--imageSize`, ...). It only changes defaults that are wrong for this task and
adds the flags the SR training loop needs.

Changed defaults and why
------------------------
  --patch_size 128 128 128 -> unchanged in size but now settable from the CLI
        (the base parser declares it without `nargs`, so it could only ever be
        overridden in Python. Fixed here.)
  --new_resolution (0.45,)*3 -> (1.0,)*3   The paediatric ULF pipeline worked at
        0.45 mm. Our target grid is 1 mm. Only used when --resample is on.
  --niter/--niter_decay 500/500 -> 100/100  1000 epochs over a few hundred
        volumes is enormous for a paired, well-posed SR task; the L1 term
        converges fast. Raise it if your validation curve is still improving.
  --lambda_A 100 -> 100 (unchanged) but see --lambda_adv: for SR you usually
        want the adversarial term small, since a strong GAN term invents
        plausible-but-wrong cortical detail. 0 gives you a pure L1 regressor,
        which maximises PSNR and is the right baseline to beat.
  --save_epoch_freq 100 -> 10, so a preempted cluster job loses less.

Added flags
-----------
  --val_freq              epochs between validation passes
  --val_patch_size        patch used for sliding-window validation
  --val_stride            stride for the sliding window
  --val_max_volumes       cap validation cost
  --val_metric            which metric selects `best_net_G.pth`
  --sr_augment            use sr_transforms.SRAugmentation instead of the
                          label-destroying default Augmentation
  --input_noise_std       input-only noise jitter (see sr_transforms)
  --min_fg_frac           foreground fraction required of a training crop
  --amp                   mixed precision
  --grad_accum            gradient accumulation steps (emulate a larger batch)
  --seed                  full reproducibility
  --tensorboard           write TB logs under <checkpoints_dir>/<name>/tb
"""

import argparse
import os

from options.train_options import TrainOptions


def _expand3(vals, name):
    """Normalise a 1- or 3-element list to exactly 3.

    These options use `nargs='+', type=int` rather than a single string-parsing
    `type=`, so that ALL of these work:

        --patch_size 96                    -> [96, 96, 96]
        --patch_size 128 128 96            -> [128, 128, 96]
        --patch_size $PATCH_SIZE           (unquoted, word-splits into 3)

    That last form is what the sbatch scripts use and what `check_dataset.py` and
    `evaluate_sr.py` already accepted (`nargs=3`). An earlier revision used a
    single-argument `type=` that parsed "128 128 128" as one string, which meant
    the *unquoted* form -- the one every cluster script and the smoke test used --
    failed with "unrecognized arguments: 128 128".
    """
    if vals is None:
        return None
    vals = list(vals)
    if len(vals) == 1:
        vals = vals * 3
    if len(vals) != 3:
        raise SystemExit('%s: expected 1 or 3 values, got %d (%s)'
                         % (name, len(vals), vals))
    return [int(v) for v in vals]


class SROptions(TrainOptions):

    def initialize(self, parser):
        parser = TrainOptions.initialize(self, parser)

        # ---- fix flags the base parser made un-overridable ------------------
        # The base parser declares --patch_size / --new_resolution without a
        # type or nargs, so `--patch_size 128 128 128` fails. Re-register them.
        for action in list(parser._actions):
            if action.dest in ('patch_size', 'new_resolution'):
                parser._remove_action(action)
                for opt_str in action.option_strings:
                    parser._option_string_actions.pop(opt_str, None)

        parser.add_argument('--patch_size', type=int, nargs='+',
                            default=[128, 128, 128],
                            help='Training patch size: one value for a cube, or '
                                 'three. Must be divisible by 4 (two stride-2 '
                                 'encoders and two transposed-conv decoders).')
        parser.add_argument('--new_resolution', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                            help='Only used when --resample is true')

        # ---- SR-appropriate defaults ---------------------------------------
        parser.set_defaults(
            model='gambas',
            netG='gambas',
            which_direction='AtoB',      # images/ = 2 mm input, labels/ = 1 mm target
            resample=False,              # the dataset builder already put both on
                                         # the same 1 mm grid; resampling again
                                         # would only blur the target
            niter=100,
            niter_decay=100,
            save_epoch_freq=10,
            save_latest_freq=500,
            print_freq=20,
            batch_size=1,
            workers=8,
            name='sr_2mm_to_1mm',
            # Pure L1 by default. At ~50 training volumes a discriminator can
            # effectively memorise the set, GAN instability is hardest to diagnose
            # at small n, and L1 maximises PSNR -- so this is the honest baseline
            # any adversarial variant has to beat. It also skips the
            # discriminator entirely (see models/gambas_model.py), freeing the
            # memory that a 128^3 patch needs. Raise it once you have the L1
            # number and can compare hf_energy against the target's.
            lambda_adv=0.0,
        )

        # ---- warm start ------------------------------------------------------
        parser.add_argument('--init_from', type=str, default='',
                            help='Path to a .pth to initialise the GENERATOR from, '
                                 'e.g. the released GAMBAS latest_net_G.pth. At '
                                 'n~50 volumes this is likely the largest single '
                                 'win available. Keys are reconciled for module. '
                                 'prefixes, InstanceNorm buffers and shape '
                                 'mismatches, and coverage is reported. Ignored '
                                 'when --continue_train resumes an existing run.')
        parser.add_argument('--init_from_D', type=str, default='',
                            help='Optional discriminator initialisation. The public '
                                 'GAMBAS release ships G only, so normally leave '
                                 'this empty.')
        parser.add_argument('--init_min_coverage', type=float, default=0.5,
                            help='Refuse to start if less than this fraction of '
                                 'generator tensors could be initialised from '
                                 '--init_from. Guards against silently warm-starting '
                                 'from an incompatible checkpoint, which looks like '
                                 'it worked and is worse than not doing it.')
        parser.add_argument('--init_allow_partial', action='store_true',
                            help='Proceed despite low coverage from --init_from')

        # ---- subgroup balancing ---------------------------------------------
        parser.add_argument('--balance_subgroups', action='store_true',
                            help='Sample training patches so the t1w/t2w imbalance '
                                 'is (partially) equalised. Labels come from the '
                                 'original filenames via the dataset manifest.')
        parser.add_argument('--balance_power', type=float, default=0.5,
                            help='0 = natural frequency, 1 = full balance, '
                                 '0.5 = sqrt balance (default). Full balance on a '
                                 '38/14 split revisits each minority volume ~2.7x '
                                 'as often; random cropping makes each visit a '
                                 'different patch, but it is still the same 14 '
                                 'individuals.')
        parser.add_argument('--balance_cap', type=float, default=None,
                            help='Optional cap on the max/min sampling weight ratio')
        parser.add_argument('--name_schema', type=str,
                            default='id,session,age,weighting',
                            help='Filename field order, used to read the subgroup '
                                 'label for --balance_subgroups')

        # ---- validation -----------------------------------------------------
        parser.add_argument('--val_freq', type=int, default=5,
                            help='Run validation every N epochs (0 disables)')
        parser.add_argument('--val_patch_size', type=int, nargs='+', default=None,
                            help='Sliding-window patch for validation: one value '
                                 'or three (default: same as --patch_size)')
        parser.add_argument('--val_stride', type=int, nargs='+', default=None,
                            help='Sliding-window stride: one value or three '
                                 '(default: half the patch)')
        parser.add_argument('--val_max_volumes', type=int, default=8,
                            help='Validate on at most this many volumes')
        parser.add_argument('--val_metric', type=str, default='psnr',
                            choices=['psnr', 'ssim', 'l1'],
                            help='Metric that decides best_net_G.pth')
        parser.add_argument('--save_val_predictions', action='store_true',
                            help='Write validation predictions as NIfTI each val pass')

        # ---- augmentation ---------------------------------------------------
        parser.add_argument('--sr_augment', action='store_true', default=True,
                            help='Use sr_transforms.SRAugmentation (recommended). '
                                 'The repo default blurs and re-noises the LABEL.')
        parser.add_argument('--legacy_augment', dest='sr_augment', action='store_false',
                            help='Use utils.NiftiDataset.Augmentation instead')
        parser.add_argument('--no_augment', action='store_true',
                            help='Disable augmentation entirely')
        parser.add_argument('--aug_prob', type=float, default=0.8)
        parser.add_argument('--input_noise_std', type=float, default=0.01,
                            help='Input-only Rician noise jitter, fraction of range')
        parser.add_argument('--gamma_jitter', type=float, default=0.1,
                            help='Paired gamma jitter half-width (0 disables)')
        parser.add_argument('--min_fg_frac', type=float, default=0.10,
                            help='Minimum tissue fraction in a training crop')

        # ---- on-the-fly degradation randomisation ---------------------------
        parser.add_argument('--randomize_degradation', action='store_true',
                            help='Re-simulate the 2 mm input from the 1 mm target '
                                 'per training sample, with a randomly drawn '
                                 'apodisation and SNR, instead of using the '
                                 'precomputed lowres-<method>/ volume. TRAINING '
                                 'ONLY -- validation and test always use the '
                                 'precomputed fixed condition, so the metric you '
                                 'select and report on stays interpretable.')
        parser.add_argument('--apod_range', type=float, nargs=2, default=[0.0, 0.3],
                            help='Tukey taper fraction range. 0 = rectangular '
                                 'truncation, 1 = full Hann. The default is '
                                 'deliberately narrow: the endpoints differ by '
                                 '~87%% in mean gradient magnitude, much wider '
                                 'than plausible vendor variation.')
        parser.add_argument('--snr_range', type=float, nargs=2, default=[20.0, 40.0],
                            help='Target SNR range at source resolution. '
                                 '0 0 disables noise.')
        parser.add_argument('--degradation_modes', nargs='+', default=['kspace'],
                            choices=['kspace', 'kspace_hann', 'slab', 'gaussian'],
                            help="Families to sample from. Default ('kspace') is "
                                 'the rectangular family with continuous '
                                 'apodisation, which already spans kspace_hann via '
                                 '--apod_range. Do not add "gaussian" (unphysical, '
                                 'ablation only); add "slab" only for genuinely 2D '
                                 'multi-slice protocols.')
        parser.add_argument('--degradation_p', type=float, default=1.0,
                            help='Probability of re-simulating a given sample. '
                                 '<1 mixes the precomputed fixed condition back in.')
        parser.add_argument('--target_spacing', type=float, nargs=3,
                            default=[2.0, 2.0, 2.0],
                            help='Simulated acquisition voxel size, for '
                                 '--randomize_degradation. Must match what '
                                 'simulate_lowres.py used.')
        parser.add_argument('--source_spacing', type=float, nargs=3,
                            default=[1.0, 1.0, 1.0],
                            help='Source voxel size, for --randomize_degradation')

        # ---- optimisation ---------------------------------------------------
        parser.add_argument('--amp', action='store_true',
                            help='bf16/fp16 autocast. Test that mamba_ssm is happy '
                                 'with it on your GPU before trusting long runs.')
        parser.add_argument('--amp_dtype', type=str, default='bfloat16',
                            choices=['bfloat16', 'float16'])
        parser.add_argument('--grad_accum', type=int, default=1,
                            help='Accumulate this many steps before an optimiser step')
        parser.add_argument('--iters_per_epoch', type=int, default=0,
                            help='If >0, sample this many patches per epoch instead '
                                 'of one per volume. Useful when you have few '
                                 'volumes but want many crops per epoch.')
        parser.add_argument('--seed', type=int, default=1234)
        parser.add_argument('--tensorboard', action='store_true', default=True)
        parser.add_argument('--no_tensorboard', dest='tensorboard', action='store_false')

        self.isTrain = True
        return parser

    def parse(self):
        opt = super(SROptions, self).parse()

        # Normalise the 1-or-3 forms to exactly 3 before anything downstream
        # indexes them. Done here rather than in a `type=` callable so that the
        # unquoted `--patch_size $PATCH_SIZE` form used by every sbatch script
        # works, as well as `--patch_size 96`.
        opt.patch_size = _expand3(opt.patch_size, '--patch_size')
        opt.val_patch_size = _expand3(opt.val_patch_size, '--val_patch_size')
        opt.val_stride = _expand3(opt.val_stride, '--val_stride')

        if opt.val_patch_size is None:
            opt.val_patch_size = list(opt.patch_size)
        if opt.val_stride is None:
            opt.val_stride = [max(1, p // 2) for p in opt.val_patch_size]
        for p in opt.patch_size:
            if p % 4:
                raise SystemExit(
                    'patch_size %s: every dimension must be divisible by 4, because '
                    'the GAMBAS generator has two stride-2 encoder layers and two '
                    'transposed-conv decoders. Got %d.' % (opt.patch_size, p))
        if not os.path.isdir(os.path.join(opt.data_path, 'images')):
            print('WARNING: %s/images does not exist. Did you run '
                  'sr/build_sr_dataset.py and point --data_path at '
                  '<out_root>/train?' % opt.data_path)
        return opt
