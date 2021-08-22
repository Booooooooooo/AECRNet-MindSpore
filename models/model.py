import numpy as np
from scipy.stats import truncnorm

import mindspore
from mindspore import Parameter, Tensor
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from losses.loss import Loss
# from mindspore.common.tensor import Tensor

from models.DCN import DeformConv2d
ops_print = ops.Print()
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, pad_mode='pad', padding=(kernel_size // 2), has_bias=bias)

class PALayer(nn.Cell):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.SequentialCell([
            nn.Conv2d(channel, channel // 8, 1, padding=0, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // 8, 1, 1, padding=0, has_bias=True),
            nn.Sigmoid()
        ])

    def construct(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Cell):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        #### Done:self.avg_pool = nn.AdaptiveAvgPool2d(1) # 指定输出map的size
        self.ca = nn.SequentialCell([
            nn.Conv2d(channel, channel // 8, 1, padding=0, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // 8, channel, 1, padding=0, has_bias=True),
            nn.Sigmoid()
        ])

    def construct(self, x):
        avg_pool = nn.AvgPool2d((x.shape[-2], x.shape[-1]))
        y = avg_pool(x)
        y = self.ca(y)
        return x * y

class DehazeBlock(nn.Cell):
    def __init__(self, conv, dim, kernel_size):
        super(DehazeBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU()
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def construct(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

class Mix(nn.Cell):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = Parameter(Tensor(m, mindspore.float32), requires_grad=True)
        self.w = w
        self.exp = ops.Exp()
        self.mix_block = nn.Sigmoid()

    def construct(self, fea1, fea2):
        # mix_factor = 1 / (1 + self.exp(-self.w))
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out, self.w

class Dehaze(nn.Cell):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(Dehaze, self).__init__()

        self.down1 = nn.SequentialCell([nn.Pad(paddings=((0,0), (0, 0), (3,3), (3,3)), mode='REFLECT'),
                                        nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                        nn.ReLU()])
        # self.pad = P.MirrorPad(mode="REFLECT")
        # self.down1 = nn.SequentialCell([nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
        #                                 nn.ReLU()])
        self.down2 = nn.SequentialCell([nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, pad_mode='pad', padding=1),
                                        nn.ReLU()])
        self.down3 = nn.SequentialCell([nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, pad_mode='pad', padding=1),
                                        nn.ReLU()])

        self.block = DehazeBlock(default_conv, ngf*4, 3)

        self.up1 = nn.SequentialCell([nn.Conv2dTranspose(ngf*4, ngf*2, kernel_size=3, stride=2, pad_mode='pad', padding=1),
                                      # nn.Pad(paddings=((0, 0), (0, 0), (0,1), (0,1))), ## output_padding=1
                                      nn.ReLU()])
        self.up2 = nn.SequentialCell([nn.Conv2dTranspose(ngf*2, ngf, kernel_size=3, stride=2, pad_mode='pad', padding=1),
                                      nn.Pad(paddings=((0, 0), (0, 0), (0,1),(0,1))),
                                      nn.ReLU()])
        self.up3 = nn.SequentialCell([nn.Pad(paddings=((0,0), (0,0), (3,3), (3,3)), mode='REFLECT'),
                                      nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                      nn.Tanh()])

        self.dcn_block = DeformConv2d(256, 256)

        #### deconv可以考虑省略
        # self.deconv = FastDeconv(3, 3, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.Conv2d(3, 3, 3, stride=1, padding=1, pad_mode='pad')

        m1 = -1
        m2 = -0.6
        self.mix4 = Mix(m=m1)
        self.mix5 = Mix(m=m2)
        print(f'Mix setting: m1={m1} m2={m2}')

    def construct(self, x):
        #### deconv可以考虑省略
        x_deconv = self.deconv(x)

        x_down1 = self.down1(x_deconv)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)

        x1 = self.block(x_down3)
        x2 = self.block(x1)

        x3 = self.block(x2)
        x4 = self.block(x3)

        x5 = self.block(x4)
        x6 = self.block(x5)

        ###TODO:dcn not implemented
        x_dcn1 = self.dcn_block(x6)
        x_dcn2 = self.dcn_block(x_dcn1)

        x_out_mix, m4 = self.mix4(x_down3, x_dcn2)
        # x_out_mix, m4 = self.mix4(x_down3, x6)
        x_up1 = self.up1(x_out_mix)
        x_up1_mix, m5 = self.mix5(x_down2, x_up1)
        x_up2 = self.up2(x_up1_mix)
        out = self.up3(x_up2)

        return out, [x2, x4, x6], m4, m5

class DehazeWithLossCell(nn.Cell):
    def __init__(self, net):
        super(DehazeWithLossCell, self).__init__()
        self.net = net
        self.loss = Loss()
        self.psnr = nn.PSNR()
        self.ssim = nn.SSIM()

    def construct(self, input, clear):
        # ops_print(input)
        output, _, m4, m5 = self.net(input)
        output = output[:, :, :clear.shape[-2], :clear.shape[-1]]

        loss = self.loss(output, clear, input, clear)
        psnr = self.psnr(output, clear)
        ssim = self.ssim(output, clear)
        # losses = [loss, psnr, ssim]

        # ops_print('In Model:', loss, psnr.mean(), ssim.mean())
        # return loss, psnr, ssim
        return loss





