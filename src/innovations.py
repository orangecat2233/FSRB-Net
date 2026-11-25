from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, dilation=1, activation=True):
        super().__init__()
        if padding is None:
            padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class FourierConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.pre = ConvBNAct(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.wr = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.wi = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.stride = stride
        self.out = ConvBNAct(out_channels, out_channels, 3 if stride > 1 else 1, stride)

    def forward(self, x):
        x = self.pre(x)
        dtype = x.dtype
        z = torch.fft.rfft2(x.float(), norm="ortho")
        z = torch.complex(z.real * self.wr - z.imag * self.wi, z.real * self.wi + z.imag * self.wr)
        y = torch.fft.irfft2(z, s=x.shape[-2:], norm="ortho").to(dtype)
        return self.out(y)


class FourierSR(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, num_blocks=8, sparsity_threshold=0.01):
        super().__init__()
        if in_channels % num_blocks:
            raise ValueError("in_channels must be divisible by num_blocks")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sparsity_threshold = float(sparsity_threshold)
        self.num_blocks = int(num_blocks)
        self.block_size = in_channels // num_blocks
        self.scale = 0.02
        self.w = nn.Parameter(self.scale * torch.randn(num_blocks, self.block_size, self.block_size, 2))
        self.w1 = nn.Parameter(self.scale * torch.randn(2, num_blocks, self.block_size, 1, 1))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, num_blocks, self.block_size, 1, 1))
        self.b = nn.Parameter(self.scale * torch.randn(2, num_blocks, self.block_size))
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        residual = x
        dtype = x.dtype
        x = x.float()
        batch, channels, height, width = x.shape
        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        x = x.reshape(batch, self.num_blocks, self.block_size, x.shape[2], x.shape[3])
        weight = torch.view_as_complex(self.w.float().contiguous())
        x = torch.einsum("bkihw,kio->bkohw", x, weight)
        w1, w2, bias = self.w1.float(), self.w2.float(), self.b.float()
        real = F.relu(x.real * w1[0].unsqueeze(0) - x.imag * w1[1].unsqueeze(0) + bias[0, :, :, None, None])
        imag = F.relu(x.imag * w2[0].unsqueeze(0) + x.real * w2[1].unsqueeze(0) + bias[1, :, :, None, None])
        x = torch.stack((real, imag), dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x).reshape(batch, channels, real.shape[-2], real.shape[-1])
        x = torch.fft.irfft2(x, s=(height, width), dim=(2, 3), norm="ortho").to(dtype)
        return self.conv_1x1(x + residual)


class CoordAtt(nn.Module):
    def __init__(self, channels=512, reduction=32):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.reduce = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.BatchNorm2d(hidden), nn.Hardswish())
        self.height = nn.Conv2d(hidden, channels, 1)
        self.width = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        height, width = x.shape[-2:]
        h = x.mean(dim=3, keepdim=True)
        w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        z = self.reduce(torch.cat((h, w), dim=2))
        h, w = torch.split(z, (height, width), dim=2)
        return x * self.height(h).sigmoid() * self.width(w.transpose(2, 3)).sigmoid()
