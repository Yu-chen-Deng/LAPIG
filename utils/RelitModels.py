# Author: Bingyao Huang (https://github.com/BingyaoHuang/CompenNeSt-plusplus)
# Modified by: Yuchen Deng

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pytorch_tps


class ShadingNetSPAA(nn.Module):
    # Extended from CompenNet
    def __init__(self, use_rough=True):
        super(ShadingNetSPAA, self).__init__()
        self.use_rough = use_rough
        self.name = (
            self.__class__.__name__
            if self.use_rough
            else self.__class__.__name__ + "_no_rough"
        )
        self.relu = nn.ReLU()

        # backbone branch
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)

        # surface image feature extraction branch
        num_chan = 6 if self.use_rough else 3
        self.conv1_s = nn.Conv2d(num_chan, 32, 3, 2, 1)
        self.conv2_s = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3_s = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4_s = nn.Conv2d(128, 256, 3, 1, 1)

        # transposed conv
        self.transConv1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.transConv2 = nn.ConvTranspose2d(64, 32, 2, 2, 0)
        self.conv6 = nn.Conv2d(32, 3, 3, 1, 1)

        # skip layers
        self.skipConv1 = nn.Sequential(
            nn.Conv2d(3, 3, 1, 1, 0),
            self.relu,
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu,
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu,
        )

        self.skipConv2 = nn.Conv2d(32, 64, 1, 1, 0)
        self.skipConv3 = nn.Conv2d(64, 128, 3, 1, 1)

        # stores biases of surface feature branch (net simplification)
        self.register_buffer("res1_s", None)
        self.register_buffer("res2_s", None)
        self.register_buffer("res3_s", None)
        self.register_buffer("res4_s", None)

        # initialization function, first checks the module type,
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    # simplify trained model by trimming surface branch to biases
    def simplify(self, s):
        self.res1_s = self.relu(self.conv1_s(s))
        self.res2_s = self.relu(self.conv2_s(self.res1_s))
        self.res3_s = self.relu(self.conv3_s(self.res2_s))
        self.res4_s = self.relu(self.conv4_s(self.res3_s))

        self.res1_s = self.res1_s.squeeze()
        self.res2_s = self.res2_s.squeeze()
        self.res3_s = self.res3_s.squeeze()
        self.res4_s = self.res4_s.squeeze()

    # x is the input uncompensated image, s is a 1x3x256x256 surface image
    def forward(self, x, *argv):
        s = torch.cat(argv, 1)

        # surface feature extraction
        res1_s = self.relu(self.conv1_s(s)) if self.res1_s is None else self.res1_s
        res2_s = self.relu(self.conv2_s(res1_s)) if self.res2_s is None else self.res2_s
        res3_s = self.relu(self.conv3_s(res2_s)) if self.res3_s is None else self.res3_s
        res4_s = self.relu(self.conv4_s(res3_s)) if self.res4_s is None else self.res4_s

        # backbone
        # res1 = self.skipConv1(x)
        res1 = self.skipConv1(argv[0])
        x = self.relu(self.conv1(x) + res1_s)
        res2 = self.skipConv2(x)
        x = self.relu(self.conv2(x) + res2_s)
        res3 = self.skipConv3(x)
        x = self.relu(self.conv3(x) + res3_s)
        x = self.relu(self.conv4(x) + res4_s)
        x = self.relu(self.conv5(x) + res3)
        x = self.relu(self.transConv1(x) + res2)
        x = self.relu(self.transConv2(x))
        x = torch.clamp(self.relu(self.conv6(x) + res1), max=1)

        return x


class PSA_relit_PCNet(nn.Module):
    # Project-and-Capture Network for SPAA
    def __init__(
        self,
        mask=torch.ones((1, 3, 256, 256)),
        shading_net=None,
        fix_shading_net=False,
        use_mask=True,
        use_rough=True,
    ):
        super(PSA_relit_PCNet, self).__init__()
        self.name = self.__class__.__name__
        self.use_mask = use_mask
        self.use_rough = use_rough

        if not self.use_mask:
            self.name += "_no_mask"
        if not self.use_rough:
            self.name += "_no_rough"

        # initialize from existing models or create new models
        # self.warping_net = copy.deepcopy(warping_net.module) if warping_net is not None else WarpingNet()
        self.shading_net = (
            copy.deepcopy(shading_net.module)
            if shading_net is not None
            else ShadingNetSPAA()
        )

        if self.use_mask:
            self.register_buffer("mask", mask.clone())

        # fix procam net
        for param in self.shading_net.parameters():
            param.requires_grad = not fix_shading_net

    # simplify trained model to a single sampling grid for faster testing
    def simplify(self, s):
        # self.warping_net.simplify(s)
        # self.shading_net.simplify(self.warping_net(s))
        self.shading_net.simplify(s)

    # s is Bx3x256x256 surface image
    def forward(self, x, s):
        # geometric correction using WarpingNet
        # x = self.warping_net(x)

        if self.use_mask:
            x = x * self.mask
        if self.use_rough:
            x = self.shading_net(x, s, x * s)
        else:
            x = self.shading_net(x, s)

        return x


class ShadingNetSPAA_clamp(nn.Module):
    # Extended from CompenNet
    def __init__(self, use_rough=True):
        super(ShadingNetSPAA_clamp, self).__init__()
        self.use_rough = use_rough
        self.name = (
            self.__class__.__name__
            if self.use_rough
            else self.__class__.__name__ + "_no_rough"
        )
        self.relu = nn.ReLU()

        # backbone branch
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)

        # surface image feature extraction branch
        num_chan = 6 if self.use_rough else 3
        self.conv1_s = nn.Conv2d(num_chan, 32, 3, 2, 1)
        self.conv2_s = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3_s = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4_s = nn.Conv2d(128, 256, 3, 1, 1)

        # transposed conv
        self.transConv1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.transConv2 = nn.ConvTranspose2d(64, 32, 2, 2, 0)
        self.conv6 = nn.Conv2d(32, 3, 3, 1, 1)

        # skip layers
        self.skipConv1 = nn.Sequential(
            nn.Conv2d(3, 3, 1, 1, 0),
            self.relu,
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu,
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu,
        )

        self.skipConv2 = nn.Conv2d(32, 64, 1, 1, 0)
        self.skipConv3 = nn.Conv2d(64, 128, 3, 1, 1)

        # stores biases of surface feature branch (net simplification)
        self.register_buffer("res1_s", None)
        self.register_buffer("res2_s", None)
        self.register_buffer("res3_s", None)
        self.register_buffer("res4_s", None)

        # initialization function, first checks the module type,
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    # simplify trained model by trimming surface branch to biases
    def simplify(self, s):
        self.res1_s = self.relu(self.conv1_s(s))
        self.res2_s = self.relu(self.conv2_s(self.res1_s))
        self.res3_s = self.relu(self.conv3_s(self.res2_s))
        self.res4_s = self.relu(self.conv4_s(self.res3_s))

        self.res1_s = self.res1_s.squeeze()
        self.res2_s = self.res2_s.squeeze()
        self.res3_s = self.res3_s.squeeze()
        self.res4_s = self.res4_s.squeeze()

    # x is the input uncompensated image, s is a 1x3x256x256 surface image
    def forward(self, x, *argv):
        s = torch.cat([argv[0], argv[1]], 1)

        # surface feature extraction
        res1_s = self.relu(self.conv1_s(s)) if self.res1_s is None else self.res1_s
        res2_s = self.relu(self.conv2_s(res1_s)) if self.res2_s is None else self.res2_s
        res3_s = self.relu(self.conv3_s(res2_s)) if self.res3_s is None else self.res3_s
        res4_s = self.relu(self.conv4_s(res3_s)) if self.res4_s is None else self.res4_s

        # backbone
        # res1 = self.skipConv1(x)
        res1 = self.skipConv1(argv[0])
        x = self.relu(self.conv1(x) + res1_s)
        res2 = self.skipConv2(x)
        x = self.relu(self.conv2(x) + res2_s)
        res3 = self.skipConv3(x)
        x = self.relu(self.conv3(x) + res3_s)
        x = self.relu(self.conv4(x) + res4_s)
        x = self.relu(self.conv5(x) + res3)
        x = self.relu(self.transConv1(x) + res2)
        x = self.relu(self.transConv2(x))
        x = torch.clamp(self.relu(self.conv6(x) + res1), min=argv[2], max=argv[3])

        return x


class PSA_relit_PCNet(nn.Module):
    # Project-and-Capture Network for SPAA
    def __init__(
        self,
        mask=torch.ones((1, 3, 256, 256)),
        clamp_min=0,
        clamp_max=1,
        shading_net=None,
        fix_shading_net=False,
        use_mask=True,
        use_rough=True,
    ):
        super(PSA_relit_PCNet, self).__init__()
        self.name = self.__class__.__name__
        self.use_mask = use_mask
        self.use_rough = use_rough
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        if not self.use_mask:
            self.name += "_no_mask"
        if not self.use_rough:
            self.name += "_no_rough"

        # initialize from existing models or create new models
        # self.warping_net = copy.deepcopy(warping_net.module) if warping_net is not None else WarpingNet()
        self.shading_net = (
            copy.deepcopy(shading_net.module)
            if shading_net is not None
            else ShadingNetSPAA_clamp()
        )

        if self.use_mask:
            self.register_buffer("mask", mask.clone())

        # fix procam net
        for param in self.shading_net.parameters():
            param.requires_grad = not fix_shading_net

    # simplify trained model to a single sampling grid for faster testing
    def simplify(self, s):
        # self.warping_net.simplify(s)
        # self.shading_net.simplify(self.warping_net(s))
        self.shading_net.simplify(s)

    # s is Bx3x256x256 surface image
    def forward(self, x, s):
        # geometric correction using WarpingNet
        # x = self.warping_net(x)

        if self.use_mask:
            x = x * self.mask
        if self.use_rough:
            x = self.shading_net(x, s, x * s, self.clamp_min, self.clamp_max)
        else:
            x = self.shading_net(x, s)

        return x


class CompenNeSt(nn.Module):
    def __init__(self):
        super(CompenNeSt, self).__init__()
        self.name = self.__class__.__name__
        self.relu = nn.ReLU()

        self.simplified = False

        # siamese encoder
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)

        # transposed conv
        self.transConv1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.transConv2 = nn.ConvTranspose2d(64, 32, 2, 2, 0)

        # output layer
        self.conv6 = nn.Conv2d(32, 3, 3, 1, 1)

        # skip layers (see s3 in forward)
        # s1
        self.skipConv11 = nn.Conv2d(3, 3, 1, 1, 0)
        self.skipConv12 = nn.Conv2d(3, 3, 3, 1, 1)
        self.skipConv13 = nn.Conv2d(3, 3, 3, 1, 1)

        # s2
        self.skipConv21 = nn.Conv2d(32, 64, 1, 1, 0)
        self.skipConv22 = nn.Conv2d(64, 64, 3, 1, 1)

        # stores biases of surface feature branch (net simplification)
        self.register_buffer("res1_s_pre", None)
        self.register_buffer("res2_s_pre", None)
        self.register_buffer("res3_s_pre", None)

        # initialization function, first checks the module type,
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    # simplify trained model by trimming surface branch to biases
    def simplify(self, s):
        res1_s = self.relu(self.skipConv11(s))
        res1_s = self.relu(self.skipConv12(res1_s))
        res1_s = self.skipConv13(res1_s)
        self.res1_s_pre = res1_s

        s = self.relu(self.conv1(s))

        res2_s = self.skipConv21(s)
        res2_s = self.relu(res2_s)
        res2_s = self.skipConv22(res2_s)
        self.res2_s_pre = res2_s

        s = self.relu(self.conv2(s))
        s = self.relu(self.conv3(s))
        self.res3_s_pre = s

        self.res1_s_pre = self.res1_s_pre.squeeze()
        self.res2_s_pre = self.res2_s_pre.squeeze()
        self.res3_s_pre = self.res3_s_pre.squeeze()

        self.simplified = True

    # x is the input uncompensated image, s is a 1x2sx2s56x2s56 surface image
    def forward(self, output_x, s):
        # surface feature extraction

        # alternate between surface and image branch
        if self.simplified:
            res1_s = self.res1_s_pre
            res2_s = self.res2_s_pre
            res3_s = self.res3_s_pre
        else:
            res1_s = self.relu(self.skipConv11(s))
            res1_s = self.relu(self.skipConv12(res1_s))
            res1_s = self.skipConv13(res1_s)

            s = self.relu(self.conv1(s))

            res2_s = self.skipConv21(s)
            res2_s = self.relu(res2_s)
            res2_s = self.skipConv22(res2_s)

            s = self.relu(self.conv2(s))
            res3_s = self.relu(self.conv3(s))

        res1 = self.relu(self.skipConv11(output_x))
        res1 = self.relu(self.skipConv12(res1))
        res1 = self.skipConv13(res1)
        res1 -= res1_s

        x = self.relu(self.conv1(output_x))

        res2 = self.skipConv21(x)
        res2 = self.relu(res2)
        res2 = self.skipConv22(res2)
        res2 -= res2_s

        x = self.relu(self.conv2(x))

        x = self.relu(self.conv3(x))
        x = x - res3_s  # s3

        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        x = self.relu(self.transConv1(x) + res2)
        x = self.relu(self.transConv2(x))
        x = self.relu(self.conv6(x) + res1)

        x = torch.clamp(x, max=1)

        return x


class PSA_relit_CompenNeStPlusplus(nn.Module):
    def __init__(self, warping_net=None, compen_net=None):
        super(PSA_relit_CompenNeStPlusplus, self).__init__()
        self.name = self.__class__.__name__

        # initialize from existing models or create new models
        # self.warping_net = copy.deepcopy(warping_net.module) if warping_net is not None else WarpingNet()
        self.compen_nest = (
            copy.deepcopy(compen_net.module) if compen_net is not None else CompenNeSt()
        )

    # simplify trained model to a single sampling grid for faster testing
    def simplify(self, s):
        # self.warping_net.simplify(s)
        # self.compen_nest.simplify(self.warping_net(s))
        self.compen_nest.simplify(s)

    # s is Bx3x256x256 surface image
    def forward(self, x, s):
        # geometric correction using WarpingNet (both x and s)
        # x = self.warping_net(x)
        # s = self.warping_net(s)

        # photometric compensation using CompenNet
        x = self.compen_nest(x, s)

        return x


class CompenNeSt_clamp(nn.Module):
    def __init__(self):
        super(CompenNeSt_clamp, self).__init__()
        self.name = self.__class__.__name__
        self.relu = nn.ReLU()

        self.simplified = False

        # siamese encoder
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)

        # transposed conv
        self.transConv1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.transConv2 = nn.ConvTranspose2d(64, 32, 2, 2, 0)

        # output layer
        self.conv6 = nn.Conv2d(32, 3, 3, 1, 1)

        # skip layers (see s3 in forward)
        # s1
        self.skipConv11 = nn.Conv2d(3, 3, 1, 1, 0)
        self.skipConv12 = nn.Conv2d(3, 3, 3, 1, 1)
        self.skipConv13 = nn.Conv2d(3, 3, 3, 1, 1)

        # s2
        self.skipConv21 = nn.Conv2d(32, 64, 1, 1, 0)
        self.skipConv22 = nn.Conv2d(64, 64, 3, 1, 1)

        # stores biases of surface feature branch (net simplification)
        self.register_buffer("res1_s_pre", None)
        self.register_buffer("res2_s_pre", None)
        self.register_buffer("res3_s_pre", None)

        # initialization function, first checks the module type,
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    # simplify trained model by trimming surface branch to biases
    def simplify(self, s):
        res1_s = self.relu(self.skipConv11(s))
        res1_s = self.relu(self.skipConv12(res1_s))
        res1_s = self.skipConv13(res1_s)
        self.res1_s_pre = res1_s

        s = self.relu(self.conv1(s))

        res2_s = self.skipConv21(s)
        res2_s = self.relu(res2_s)
        res2_s = self.skipConv22(res2_s)
        self.res2_s_pre = res2_s

        s = self.relu(self.conv2(s))
        s = self.relu(self.conv3(s))
        self.res3_s_pre = s

        self.res1_s_pre = self.res1_s_pre.squeeze()
        self.res2_s_pre = self.res2_s_pre.squeeze()
        self.res3_s_pre = self.res3_s_pre.squeeze()

        self.simplified = True

    # x is the input uncompensated image, s is a 1x2sx2s56x2s56 surface image
    def forward(self, output_x, s, clamp_min, clamp_max):
        # surface feature extraction

        # alternate between surface and image branch
        if self.simplified:
            res1_s = self.res1_s_pre
            res2_s = self.res2_s_pre
            res3_s = self.res3_s_pre
        else:
            res1_s = self.relu(self.skipConv11(s))
            res1_s = self.relu(self.skipConv12(res1_s))
            res1_s = self.skipConv13(res1_s)

            s = self.relu(self.conv1(s))

            res2_s = self.skipConv21(s)
            res2_s = self.relu(res2_s)
            res2_s = self.skipConv22(res2_s)

            s = self.relu(self.conv2(s))
            res3_s = self.relu(self.conv3(s))

        res1 = self.relu(self.skipConv11(output_x))
        res1 = self.relu(self.skipConv12(res1))
        res1 = self.skipConv13(res1)
        res1 -= res1_s

        x = self.relu(self.conv1(output_x))

        res2 = self.skipConv21(x)
        res2 = self.relu(res2)
        res2 = self.skipConv22(res2)
        res2 -= res2_s

        x = self.relu(self.conv2(x))

        x = self.relu(self.conv3(x))
        x = x - res3_s  # s3

        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        x = self.relu(self.transConv1(x) + res2)
        x = self.relu(self.transConv2(x))
        x = self.relu(self.conv6(x) + res1)

        x = torch.clamp(x, min=clamp_min, max=clamp_max)

        return x


class PSA_relit_CompenNeStPlusplus_clamp(nn.Module):
    def __init__(self, warping_net=None, clamp_min=0, clamp_max=1, compen_net=None):
        super(PSA_relit_CompenNeStPlusplus_clamp, self).__init__()
        self.name = self.__class__.__name__
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # initialize from existing models or create new models
        # self.warping_net = copy.deepcopy(warping_net.module) if warping_net is not None else WarpingNet()
        self.compen_nest = (
            copy.deepcopy(compen_net.module)
            if compen_net is not None
            else CompenNeSt_clamp()
        )

    # simplify trained model to a single sampling grid for faster testing
    def simplify(self, s):
        # self.warping_net.simplify(s)
        # self.compen_nest.simplify(self.warping_net(s))
        self.compen_nest.simplify(s)

    # s is Bx3x256x256 surface image
    def forward(self, x, s):
        # geometric correction using WarpingNet (both x and s)
        # x = self.warping_net(x)
        # s = self.warping_net(s)

        # photometric compensation using CompenNet
        x = self.compen_nest(x, s, self.clamp_min, self.clamp_max)

        return x
