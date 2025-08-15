import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import DeformConv2dPack

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, gc=32, bias=True):
        super(DenseBlock, self).__init__()

        self.conv1 = DeformConv2dPack(channel_in, gc, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, stride=1, padding=1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, stride=1, padding=1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, stride=1, padding=1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = x.contiguous()
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5

class SEBlock(nn.Module):
    def __init__(self, channel_in, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel_in, channel_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_in // reduction, channel_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEDenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, gc=32, bias=True):
        super(SEDenseBlock, self).__init__()
        self.se = SEBlock(channel_in)
        self.dense_block = DenseBlock(channel_in, channel_out, gc, bias)

    def forward(self, x):
        x = self.se(x)
        x = self.dense_block(x)
        return x

class DAINN(nn.Module):
    def __init__(self, split_len1, split_len2, clamp=1.0):
        super(DAINN, self).__init__()
        self.split_len1 = split_len1
        self.split_len2 = split_len2
        self.clamp = clamp

        self.F =SEDenseBlock(self.split_len2, self.split_len1)
        self.G = SEDenseBlock(self.split_len1, self.split_len2)
        self.H = SEDenseBlock(self.split_len1, self.split_len2)
            
    def forward(self, x, rev=False):

        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape

    def get_weight(self, rev):
        w_shape = self.w_shape
        if not rev:
            weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
        else:
            weight = torch.inverse(self.weight.double()).float() \
                .view(w_shape[0], w_shape[1], 1, 1)
        return weight

    def forward(self, input, rev=False):
        weight = self.get_weight(rev)
        if not rev:
            z = F.conv2d(input, weight)
            return z
        else:
            z = F.conv2d(input, weight)
            return z


class DAIM(nn.Module):
    def __init__(self, opt):
        super(DAIM, self).__init__()
        self.operations = nn.ModuleList()
        #
        for _ in range(opt['network']['InvBlock']['block_num']):
            #
            if opt['network']['InvBlock']['downscaling']['use_conv1x1']:
                a = InvertibleConv1x1(opt['network']['InvBlock']['split1_img'] + \
                                      opt['network']['InvBlock']['split2_repeat'])
                self.operations.append(a)
            #
            b = DAINN(opt['network']['InvBlock']['split1_img'], opt['network']['InvBlock']['split2_repeat'])
            self.operations.append(b)

    def forward(self, x, rev=False):
        if not rev:
            #
            for op in self.operations:
                x = op.forward(x, rev)
            #
        else:
            for op in reversed(self.operations):
                x = op.forward(x, rev)
        return x
