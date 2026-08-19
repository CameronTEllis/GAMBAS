import torch
from collections import OrderedDict
from torch.autograd import Variable
# import util.util as util
from .base_model import BaseModel
from . import networks3D
from torchvision import models
import random
# from generative.losses import PerceptualLoss

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


class GambasModel(BaseModel):
    def name(self):
        return "GambasModel"

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_f', type=float, default=0.9, help='momentum term for f')
            parser.add_argument('--lambda_A', type=float, default=100.0, help='lambda for sobel l1 loss')
            parser.add_argument('--lambda_adv', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--l1_edge_weight', type=float, default=0.0,
                                help='Edge-weighted L1: weight the per-voxel L1 by '
                                     'the TARGET gradient magnitude so edges (incl. '
                                     'the GM/WM boundary) get more loss. The loss is '
                                     'renormalised to mean weight 1, so total loss '
                                     'scale and lambda_A are unchanged -- it only '
                                     'REDISTRIBUTES effort onto edges. 0 = plain L1 '
                                     '(default, identical to before); ~1-4 emphasises '
                                     'edges. Cheap first test of a boundary-aware loss.')
            parser.add_argument('--imageSize', type=int, default=256, help='size of largest axis from input 3D volume (if all equal, then this is the size of all axes)')
            # parser.add_argument('--lambda_perc', type=float, default=1.0, help='weight for perceptual loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D', 'G_GAN', 'G_L1']
        # self.loss_names = ['D', 'G_GAN', 'G_L1', 'G_perc']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names = ['real_A', 'fake_B', 'real_B']

        self.visual_names = visual_names
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']


        # load/define networks
        
        self.netG = networks3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, 
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      **{'img_size': (opt.imageSize, opt.imageSize),
                                         # getattr: the non-SR entry points
                                         # (train.py, test.py) never define it.
                                         'global_residual': getattr(opt, 'global_residual', False),
                                         }
                                      )
        if self.isTrain:
            self.lambda_f = opt.lambda_f
            use_sigmoid = opt.no_lsgan
            self.netD = networks3D.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, 
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        # NOTE: this block used to call self.load_network(...) -- singular -- which
        # is not defined on BaseModel (or anywhere in the repo), so ANY use of
        # --continue_train with --model gambas raised AttributeError before
        # training could start. It is also redundant: BaseModel.setup(), which
        # runs immediately after initialize(), already calls load_networks() (plural,
        # and correct) under exactly the same condition. ea_gan_model.py and
        # resvit_model.py have these same two lines commented out for this reason;
        # gambas_model.py was missed. Left here as a comment to match its siblings.
        #
        # if not self.isTrain or opt.continue_train:
        #     self.load_network(self.netG, "G", opt.which_epoch)
        #     if self.isTrain:
        #         self.load_network(self.netD, "D", opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks3D.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # self.criterionPerc = PerceptualLoss(spatial_dims=3, network_type="alex").to(self.device)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


        # print("---------- Networks initialized -------------")
        # networks.print_network(self.netG)
        # if self.isTrain:
        #     networks.print_network(self.netD)
        # print("-----------------------------------------------")

    def set_input(self, input):
        AtoB = self.opt.which_direction == "AtoB"
        self.real_A = input[0 if AtoB else 1].to(self.device)
        self.real_B = input[1 if AtoB else 0].to(self.device)

    def forward(self):
        self.fake_B = self.netG(self.real_A)
        # self.fake_sobel = networks3D.sobelLayer(self.fake_B)
        # self.real_sobel = networks3D.sobelLayer(self.real_B).detach() 

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        # fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B, self.fake_sobel), 1))
        self.pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        # real_AB = torch.cat((self.real_A, self.real_B, self.real_sobel), 1)
        self.pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    @staticmethod
    def _edge_weighted_l1(fake, real, alpha):
        """mean( w * |fake - real| ) with w = 1 + alpha * edge(real), mean(w)=1.

        `edge(real)` is the target's gradient magnitude (forward differences along
        D,H,W), normalised per sample by 4x its mean so that voxels whose gradient
        exceeds ~4x the average -- i.e. tissue boundaries -- approach full weight
        while flat regions stay near zero. Computed under no_grad: it is a weight on
        the target, not a differentiable term. Renormalising to mean weight 1 keeps
        the total loss magnitude (and thus lambda_A) identical to plain L1, so the
        only change versus the baseline is WHERE the L1 effort is spent.
        """
        with torch.no_grad():
            dz = torch.zeros_like(real)
            dy = torch.zeros_like(real)
            dx = torch.zeros_like(real)
            dz[:, :, 1:, :, :] = real[:, :, 1:, :, :] - real[:, :, :-1, :, :]
            dy[:, :, :, 1:, :] = real[:, :, :, 1:, :] - real[:, :, :, :-1, :]
            dx[:, :, :, :, 1:] = real[:, :, :, :, 1:] - real[:, :, :, :, :-1]
            g = torch.sqrt(dz * dz + dy * dy + dx * dx + 1e-12)
            b = g.shape[0]
            gmean = g.view(b, -1).mean(dim=1).clamp_min(1e-8).view(b, 1, 1, 1, 1)
            edge = (g / (4.0 * gmean)).clamp(0.0, 1.0)
            w = 1.0 + float(alpha) * edge
            w = w / w.mean()
        return (w * (fake - real).abs()).mean()

    def backward_G(self):
        # First, G(A) should fake the discriminator.
        # --lambda_adv was declared in modify_commandline_options but never
        # applied here, so setting it had no effect (resvit_model.py does apply
        # it). Applied now, which makes --lambda_adv 0.0 a pure L1 regressor --
        # the right baseline for a super-resolution task, where a strong
        # adversarial term invents plausible-but-wrong anatomical detail.
        lambda_adv = getattr(self.opt, 'lambda_adv', 1.0)
        if lambda_adv == 0:
            # Skip the discriminator forward pass too, not just its gradient.
            self.loss_G_GAN = torch.zeros((), device=self.device)
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            # fake_AB = torch.cat((self.real_A, self.fake_B, self.fake_sobel), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True) * lambda_adv
        # Second, G(A) = B
        edge_w = getattr(self.opt, 'l1_edge_weight', 0.0)
        if edge_w and edge_w > 0:
            # Edge-weighted L1: put more of the (fixed) L1 budget on high-gradient
            # regions of the TARGET -- i.e. tissue boundaries such as GM/WM -- which
            # is where the recoverable high-frequency structure lives and where a
            # plain, spatially-uniform L1 under-invests. Weights are renormalised to
            # mean 1, so this only REDISTRIBUTES effort; total loss scale and
            # lambda_A are unchanged, keeping it comparable to the plain-L1 run.
            self.loss_G_L1 = self._edge_weighted_l1(
                self.fake_B, self.real_B, edge_w) * self.opt.lambda_A
        else:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        # Third, perceptual loss: (G(A)) = (B)
        # self.loss_G_perc = self.criterionPerc(self.fake_B, self.real_B) * self.opt.lambda_perc

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_perc
        
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D -- skipped entirely when the adversarial term is switched off, which
        # saves the discriminator's activations and roughly a third of the step
        # time. Without this, --lambda_adv 0 still pays for a discriminator whose
        # gradients never reach the generator.
        if getattr(self.opt, 'lambda_adv', 1.0) == 0:
            self.loss_D = torch.zeros((), device=self.device)
            return
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
    
    # def update_sobel_lambda(self, epochNum):
    #     self.sobelLambda = self.opt.lambda_sobel/150*epochNum
    #     print('update sobel lambda: %f' % (self.sobelLambda))



