# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, LayerNorm
import torch.nn.functional as F
from .path_generate import generate3d, gilbert3d, generate_gilbert_indices_3D

from mamba_ssm import Mamba


logger = logging.getLogger(__name__)

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,dim2=None):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        #use_dropout= use_dropo
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class BottleneckCNN(nn.Module):
    def __init__(self):
        super(BottleneckCNN, self).__init__()
        use_bias = True
        norm_layer = nn.InstanceNorm3d
        padding_type = 'replicate'
        
        # Residual CNN
        model = [ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                             use_bias=use_bias)]
        setattr(self, "residual_cnn", nn.Sequential(*model))

    def forward(self, x):
        x = self.residual_cnn(x)
        return x

class MambaLayer(nn.Module):
    """ Mamba layer for state-space sequence modeling

    Args:
        dim (int): Model dimension.
        d_state (int): SSM state expansion factor.
        d_conv (int): Local convolution width.
        expand (int): Block expansion factor.
    
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2): # Before it was d_state=16, d_conv=4, expand=2
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba1 = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba2 = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

        self.conv1d = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=1)

        # The generalised-Hilbert scan order used to be precomputed here for a
        # HARDCODED 32x32x32 grid:
        #
        #     self.generator = gilbert3d(32, 32, 32)
        #     self.gilbert_indices = generate_gilbert_indices_3D(32, 32, 32, ...)
        #
        # The bottleneck sits after two stride-2 encoders, so its spatial size is
        # patch_size / 4. That made 32^3 correct for a 128^3 patch and WRONG for
        # anything else: a 96^3 patch gives a 24^3 = 13824-element sequence while
        # the gather indices still ran to 32767, producing
        #
        #     ScatterGatherKernel.cu:144 ... Assertion `idx_dim >= 0 &&
        #     idx_dim < index_size && "index out of bounds"` failed
        #
        # reported asynchronously at whatever op ran next (usually torch.flip),
        # which makes it look unrelated to indexing.
        #
        # `gilbert3d` already supports arbitrary grid sizes, so the fix is to
        # build the scan lazily from the runtime shape and cache it per shape.
        # One entry per distinct bottleneck size; training uses exactly one.
        self._scan_cache = {}

    def _scan_indices(self, D, H, W, device):
        """(gilbert, degilbert, gilbert_reversed, degilbert_reversed) for D,H,W.

        Cached per (D, H, W, device). Buffers are NOT registered as module state:
        they are derived deterministically from the shape, so keeping them out of
        the state_dict means checkpoints stay portable across patch sizes.
        """
        key = (D, H, W, str(device))
        if key not in self._scan_cache:
            gen = gilbert3d(D, H, W)
            g = generate_gilbert_indices_3D(D, H, W, gen)
            g = g.expand(-1, self.dim, -1).permute(0, 2, 1).to(device)
            n = D * H * W
            if int(g.max()) >= n or int(g.min()) < 0 or g.shape[1] != n:
                raise RuntimeError(
                    'gilbert scan for %dx%dx%d produced indices in [%d, %d] with '
                    'length %d, expected [0, %d) with length %d. The bottleneck '
                    'grid is patch_size/4 per axis.'
                    % (D, H, W, int(g.min()), int(g.max()), g.shape[1], n, n))
            gr = torch.flip(g, dims=[2])
            self._scan_cache[key] = (g, torch.argsort(g), gr, torch.argsort(gr))
        return self._scan_cache[key]


    def forward(self, x):
        B, C, D, H, W = x.shape

        # Check model dimension
        assert C == self.dim

        # Bidirectional mamba.
        #
        # The scan order is derived from the ACTUAL spatial shape rather than a
        # hardcoded 32^3, so any patch size works. Was `device = 'cuda:0'`, which
        # also broke multi-GPU DataParallel: replicas on cuda:1+ received index
        # tensors pinned to cuda:0. Take the device from the input instead.
        x1 = x.view(B, C, -1).permute(0, 2, 1)
        gilbert, degilbert, _, degilbert_r = self._scan_indices(D, H, W, x.device)

        # Expand along batch: the cached index tensor has batch extent 1.
        if gilbert.shape[0] != B:
            gilbert = gilbert.expand(B, -1, -1)
            degilbert = degilbert.expand(B, -1, -1)
            degilbert_r = degilbert_r.expand(B, -1, -1)

        x1 = torch.gather(x1, 1, gilbert)
        x2 = torch.flip(x1, dims=[1])

        # Pass forwad and reverse through mamba
        norm_out1 = self.norm(x1)
        mamba_out1 = self.mamba1(norm_out1)
        norm_out2 = self.norm(x2)
        mamba_out2 = self.mamba2(norm_out2)

        out1 = torch.gather(mamba_out1, 1, degilbert).permute(0, 2, 1).view(B, C, D, H, W)
        out2 = torch.gather(mamba_out2, 1, degilbert_r).permute(0, 2, 1).view(B, C, D, H, W)

        # out1 = mamba_out1.permute(0, 2, 1).view(B, C, D, H, W)
        # out2 = mamba_out2.permute(0, 2, 1).view(B, C, D, H, W)

        concatenated = torch.cat((out1, out2), dim=1)
        output = self.conv1d(concatenated)

        return output

class ccMambaWithCNN(nn.Module):
    """ Channel-compressed Mamba (ccMamba) block with residual CNN block

    Args:
        config (dict): Model configuration.
        in_channels (int): Number of input channels.
        d_state (int): SSM state expansion factor.
        d_conv (int): Local convolution width.
        expand (int): Block expansion factor.
        ngf (int): Number of generator filters.
        norm_layer (nn.Module): Normalization layer.
        use_dropout (bool): Use dropout.
        use_bias (bool): Use bias.
        img_size (int): Image size.
    
    """
    def __init__(self, in_channels, d_state=16, d_conv=4, expand=2, ngf=64, norm_layer=nn.BatchNorm2d, use_bias=True):
        super().__init__()
        # Mamba block
        self.mamba_layer = MambaLayer(
            dim=in_channels, d_state=d_state, d_conv=d_conv, expand=expand
        )

        ngf = 64
        padding_type = 'replicate'
        use_bias = True
        norm_layer = nn.InstanceNorm3d

        # Channel compression block
        self.cc = channel_compression(ngf*8, ngf*4)

        # Residual CNN block
        model = [ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, 
                             use_bias=use_bias)]
        setattr(self, "residual_cnn", nn.Sequential(*model))

    def forward(self, x):
        # Pass input through Mamba block
        mamba_out = self.mamba_layer(x)
        x = torch.cat([x, mamba_out], dim=1)

        # Pass Mamba block output through channel compression
        x = self.cc(x)
        
        # Pass channel compressed output through residual CNN block
        x = self.residual_cnn(x)

        return x

########Generator############
class GAMBAS(nn.Module):
    def __init__(self, input_dim, img_size=224, output_dim=3,
                 global_residual=False):
        super(GAMBAS, self).__init__()
        # self.config = config
        output_nc = output_dim
        ngf = 64

        # ------------------------------------------------------------------ #
        # Optional global (long) skip: y = x + s * G(x).
        #
        # As published, GAMBAS has no path from input to output -- the encoder
        # strides the volume down 4x, nine bottleneck blocks run at that
        # resolution, and the decoder reconstructs everything from scratch.
        # That is the right inductive bias for its original task (64 mT ULF
        # T2w -> 3T-like), where the input and target differ enormously and
        # the output really must be synthesised.
        #
        # It is the wrong bias for 2 mm -> 1 mm on the same scanner, where the
        # sinc-interpolated input is already ~38 dB from the target. There the
        # network's job is to add the missing high-frequency band, not to
        # regenerate the anatomy it was just handed -- and a 4x-strided
        # bottleneck with no skip discards precisely the fine detail it needs
        # to preserve, so it has to spend capacity relearning the identity.
        #
        # `s` is a learnable scalar initialised to ZERO, so an untrained (or
        # freshly warm-started) network outputs its input exactly and
        # validation starts at the sinc baseline rather than 10 dB below it.
        # Every optimiser step is then measured as improvement on that
        # baseline instead of a crawl back up to it.
        #
        # init_weights() only touches modules whose class name contains
        # Conv/Linear/BatchNorm3d, so this bare Parameter survives init_net().
        self.global_residual = bool(global_residual)
        if self.global_residual:
            if input_dim != output_dim:
                raise ValueError(
                    'global_residual needs input_dim == output_dim, got %d and %d'
                    % (input_dim, output_dim))
            self.res_scale = nn.Parameter(torch.zeros(1))
        use_bias = True
        norm_layer = nn.InstanceNorm3d
        padding_type = "replication"
        mult = 4

        ############################################################################################
        # Layer1-Encoder1
        model = [nn.ReplicationPad3d(3),
                 nn.Conv3d(input_dim, ngf, kernel_size=7, padding=0, 
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
      
        setattr(self, "encoder_1", nn.Sequential(*model))
        ############################################################################################
        # Layer2-Encoder2
        n_downsampling = 2
        model = []
        i = 0
        mult = 2**i
        model = [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, 
                 stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]

        setattr(self, "encoder_2", nn.Sequential(*model))
        ############################################################################################
        # Layer3-Encoder3
        model = []
        i = 1
        mult = 2**i
        model = [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, 
                 stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        
        setattr(self, "encoder_3", nn.Sequential(*model))
        ############################################################################################
        # Bottlenck Layers
        mult = 4
        img_size = 256 
        input_dim = 256 # Adjust this according to new input dimension

        # ccMamba block with residual CNN block
        self.bottleneck_1 = ccMambaWithCNN(input_dim)
        # self.bottleneck_1 = BottleneckCNN()
        
        self.bottleneck_2 = BottleneckCNN()
        self.bottleneck_3 = BottleneckCNN()
        self.bottleneck_4 = BottleneckCNN()

        # ccMamba block with residual CNN block
        self.bottleneck_5 = ccMambaWithCNN(input_dim)
        # self.bottleneck_5 = BottleneckCNN()
        
        self.bottleneck_6 = BottleneckCNN()
        self.bottleneck_7 = BottleneckCNN()
        self.bottleneck_8 = BottleneckCNN()

        # ccMamba block with residual CNN block
        # self.bottleneck_9 = BottleneckCNN()
        self.bottleneck_9 = ccMambaWithCNN(input_dim)

        ############################################################################################
        # Layer13-Decoder1
        n_downsampling = 2
        i = 0
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2), 
                                    kernel_size=3, stride=2, 
                                    padding=1, output_padding=1, 
                                    bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)]
        setattr(self, "decoder_1", nn.Sequential(*model))
        ############################################################################################
        # Layer14-Decoder2
        i = 1
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, "decoder_2", nn.Sequential(*model))
        ############################################################################################
        # Layer15-Decoder3
        model = []
        model = [nn.ReplicationPad3d(3)]
        model += [nn.Conv3d(ngf, output_dim, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        setattr(self, "decoder_3", nn.Sequential(*model))

    def forward(self, x):
        inp = x
        # Encoder
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)

        # Episodic bottleneck
        x = self.bottleneck_1(x3)
        x = self.bottleneck_2(x)
        x = self.bottleneck_3(x)
        x = self.bottleneck_4(x)
        x = self.bottleneck_5(x)
        x = self.bottleneck_6(x)
        x = self.bottleneck_7(x)
        x = self.bottleneck_8(x)
        x = self.bottleneck_9(x)

        # Decoder
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        if self.global_residual:
            # Deliberately NOT clamped to [-1, 1] here: clamping would zero the
            # gradient wherever the sum overshoots, and every consumer already
            # clips (validate/evaluate do np.clip((pred + 1) / 2, 0, 1)).
            x = inp + self.res_scale * x
        return x


class channel_compression(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(channel_compression, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=True),
            nn.InstanceNorm3d(out_channels))
        else:
          self.skip = None

        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=True),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=True),
            nn.InstanceNorm3d(out_channels))

    def forward(self, x):
        out = self.block(x)
        out += (x if self.skip is None else self.skip(x))
        out = F.relu(out)
        return out