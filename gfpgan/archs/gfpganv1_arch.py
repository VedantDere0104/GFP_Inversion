import math
import random
import torch
from basicsr.archs.stylegan2_arch import (ConvLayer, EqualConv2d, EqualLinear, ResBlock, ScaledLeakyReLU,
                                          StyleGAN2Generator)
from basicsr.ops.fused_act import FusedLeakyReLU
from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn
from torch.nn import functional as F
import sys
#/content/drive/MyDrive/gfp_inversion/GFPGAN/gfpgan/archs/models
from gfpgan.archs.models.psp import pSp
from argparse import Namespace

class StyleGAN2GeneratorSFT(StyleGAN2Generator):
    """StyleGAN2 Generator with SFT modulation (Spatial Feature Transform).

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel magnitude. A cross production will be
            applied to extent 1D resample kernel to 2D resample kernel. Default: (1, 3, 3, 1).
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
        narrow (float): The narrow ratio for channels. Default: 1.
        sft_half (bool): Whether to apply SFT on half of the input channels. Default: False.
    """

    def __init__(self,
                 out_size,
                 num_style_feat=512,
                 num_mlp=8,
                 channel_multiplier=2,
                 resample_kernel=(1, 3, 3, 1),
                 lr_mlp=0.01,
                 narrow=1,
                 sft_half=False):
        super(StyleGAN2GeneratorSFT, self).__init__(
            out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            resample_kernel=resample_kernel,
            lr_mlp=lr_mlp,
            narrow=narrow)
        self.sft_half = sft_half

    def forward(self,
                styles,
                conditions,
                input_is_latent=False,
                noise=None,
                randomize_noise=True,
                truncation=1,
                truncation_latent=None,
                inject_index=None,
                return_latents=False):
        """Forward function for StyleGAN2GeneratorSFT.

        Args:
            styles (list[Tensor]): Sample codes of styles.
            conditions (list[Tensor]): SFT conditions to generators.
            input_is_latent (bool): Whether input is latent style. Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
            truncation (float): The truncation ratio. Default: 1.
            truncation_latent (Tensor | None): The truncation latent tensor. Default: None.
            inject_index (int | None): The injection index for mixing noise. Default: None.
            return_latents (bool): Whether to return style latents. Default: False.
        """
        # style codes -> latents with Style MLP layer
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]
        # style truncation
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_truncation
        # get style latents with injection
        if len(styles) == 1:
            inject_index = self.num_latent

            if styles[0].ndim < 3:
                # repeat latent code for all the layers
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:  # used for encoder with different latent code for each layer
                latent = styles[0]
        elif len(styles) == 2:  # mixing noises
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)

        # main generation
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2],
                                                        noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)

            # the conditions may have fewer levels
            if i < len(conditions):
                # SFT part to combine the conditions
                if self.sft_half:  # only apply SFT to half of the channels
                    out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
                    out_sft = out_sft * conditions[i - 1] + conditions[i]
                    out = torch.cat([out_same, out_sft], dim=1)
                else:  # apply SFT to all the channels
                    out = out * conditions[i - 1] + conditions[i]

            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)  # feature back to the rgb space
            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None


class ConvUpLayer(nn.Module):
    """Convolutional upsampling layer. It uses bilinear upsampler + Conv.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input. Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output. Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
        activate (bool): Whether use activateion. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 bias_init_val=0,
                 activate=True):
        super(ConvUpLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # self.scale is used to scale the convolution weights, which is related to the common initializations.
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

        if bias and not activate:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

        # activation
        if activate:
            if bias:
                self.activation = FusedLeakyReLU(out_channels)
            else:
                self.activation = ScaledLeakyReLU(0.2)
        else:
            self.activation = None

    def forward(self, x):
        # bilinear upsample
        out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # conv
        out = F.conv2d(
            out,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        # activation
        if self.activation is not None:
            out = self.activation(out)
        return out


class ResUpBlock(nn.Module):
    """Residual block with upsampling.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
    """

    def __init__(self, in_channels, out_channels):
        super(ResUpBlock, self).__init__()

        self.conv1 = ConvLayer(in_channels, in_channels, 3, bias=True, activate=True)
        self.conv2 = ConvUpLayer(in_channels, out_channels, 3, stride=1, padding=1, bias=True, activate=True)
        self.skip = ConvUpLayer(in_channels, out_channels, 1, bias=False, activate=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        return out


@ARCH_REGISTRY.register()
class GFPGANv1(nn.Module):
    """The GFPGAN architecture: Unet + StyleGAN2 decoder with SFT.

    Ref: GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel magnitude. A cross production will be
            applied to extent 1D resample kernel to 2D resample kernel. Default: (1, 3, 3, 1).
        decoder_load_path (str): The path to the pre-trained decoder model (usually, the StyleGAN2). Default: None.
        fix_decoder (bool): Whether to fix the decoder. Default: True.

        num_mlp (int): Layer number of MLP style layers. Default: 8.
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
        input_is_latent (bool): Whether input is latent style. Default: False.
        different_w (bool): Whether to use different latent w for different layers. Default: False.
        narrow (float): The narrow ratio for channels. Default: 1.
        sft_half (bool): Whether to apply SFT on half of the input channels. Default: False.
    """

    def __init__(
            self,
            out_size,
            num_style_feat=512,
            channel_multiplier=1,
            resample_kernel=(1, 3, 3, 1),
            decoder_load_path=None,
            fix_decoder=True,
            # for stylegan decoder
            num_mlp=8,
            lr_mlp=0.01,
            input_is_latent=False,
            different_w=False,
            narrow=1,
            sft_half=False):

        super(GFPGANv1, self).__init__()

        model_path = "/content/drive/MyDrive/gfp_inversion/GFPGAN/experiments/pretrained_models/e4e_ffhq_encode.pt"
        ckpt = torch.load(model_path, map_location='cpu')
        print('Model Loaded')
        opts_ = ckpt['opts']
        opts_['is_train'] = True
        opts_['checkpoint_path'] = model_path
        opts_['distortion_scale'] = 0.15
        opts_['aug_rate'] = 0.8
        opts_= Namespace(**opts_)
        self.net = pSp(opts_)
        self.net.cuda()
        
        print('Model successfully loaded!')

    def get_latents(self , net, x, is_cars=False):
        codes = net.encoder(x)
        if net.opts.start_from_latent_avg:
            if codes.ndim == 2:
                codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
            else:
                codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
        if codes.shape[1] == 18 and is_cars:
            codes = codes[:, :16, :]
        return codes

    def forward(self, x, return_latents=False, return_rgb=True, randomize_noise=True):

      latent_codes = self.get_latents(self.net, x)
    
      # calculate the distortion map
      with torch.no_grad():
        imgs, _ = self.net.decoder([latent_codes[0].unsqueeze(0).cuda()],None, input_is_latent=True, randomize_noise=False, return_latents=True)
      #x = torch.nn.functional.interpolate(x, size=(256,256) , mode='bilinear')
      #res = x -  torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')

      # ADA
      #img_edit = torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')
      #res_align  = self.net.grid_align(torch.cat((res, img_edit  ), 1))

      # consultation fusion
      #conditions = self.net.residue(res_align)

      #result_image, _ = self.net.decoder([latent_codes],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)

      return imgs , _


@ARCH_REGISTRY.register()
class FacialComponentDiscriminator(nn.Module):
    """Facial component (eyes, mouth, noise) discriminator used in GFPGAN.
    """

    def __init__(self):
        super(FacialComponentDiscriminator, self).__init__()
        # It now uses a VGG-style architectrue with fixed model size
        self.conv1 = ConvLayer(3, 64, 3, downsample=False, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv2 = ConvLayer(64, 128, 3, downsample=True, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv3 = ConvLayer(128, 128, 3, downsample=False, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv4 = ConvLayer(128, 256, 3, downsample=True, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv5 = ConvLayer(256, 256, 3, downsample=False, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.final_conv = ConvLayer(256, 1, 3, bias=True, activate=False)

    def forward(self, x, return_feats=False):
        """Forward function for FacialComponentDiscriminator.

        Args:
            x (Tensor): Input images.
            return_feats (bool): Whether to return intermediate features. Default: False.
        """
        feat = self.conv1(x)
        feat = self.conv3(self.conv2(feat))
        rlt_feats = []
        if return_feats:
            rlt_feats.append(feat.clone())
        feat = self.conv5(self.conv4(feat))
        if return_feats:
            rlt_feats.append(feat.clone())
        out = self.final_conv(feat)

        if return_feats:
            return out, rlt_feats
        else:
            return out, None
