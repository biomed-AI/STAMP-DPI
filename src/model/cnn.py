# -*- encoding: utf-8 -*-
'''
@Time       : 2020/07/16 15:48:49
@Author     : Yize Jiang
@Email      : yize.jiang@galixir.com
@File       : cnn.py
@Project    : X-DPI
@Description: 动态CNN模块
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv5x5(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.elu(out)
        return out


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5,
                     stride=stride, padding=2, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


class DynamicCNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels=8,
        cnn_channels=32,
        cnn_layers=4,
        block=ResidualBlock,
    ):
        super(DynamicCNNEncoder, self).__init__()
        self.r = 10
        self.descriptor = None
        self.in_channels = in_channels
        self.conv = conv3x3(1, in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.elu = nn.ELU(inplace=False)
        self.layer1 = self.make_layer(block, cnn_channels, cnn_layers)
        self.layer2 = self.make_layer(block, cnn_channels, cnn_layers)
        self.linear_first_seq = torch.nn.Linear(cnn_channels, 32)
        self.linear_second_seq = torch.nn.Linear(32, self.r)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def softmax(self, input, axis=1):
        """
        Softmax applied to axis=n
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied

        Returns:
            softmaxed tensors
        """
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def prepare_features(self, sequences):
        features = self.dicts[sequences]
        return features

    def forward(self, pic):
        pic = self.conv(pic)
        pic = self.bn(pic)
        pic = self.elu(pic)
        pic = self.layer1(pic)
        pic = self.layer2(pic)
        seq_embed = torch.mean(pic, 2)
        seq_embed = seq_embed.permute(0, 2, 1)
        seq_att = F.tanh(self.linear_first_seq(seq_embed))
        seq_att = self.linear_second_seq(seq_att)
        seq_att = self.softmax(seq_att, 1)
        seq_att = seq_att.transpose(1, 2)
        seq_embed = seq_att @ seq_embed
        seq_embed = torch.sum(seq_embed, 1) / self.r
        return seq_embed


if __name__ == "__main__":
    model = DynamicCNNEncoder(cnn_channels=16)
    inputs = torch.rand(64, 1, 100, 100)
    outputs = model(inputs)
    print(outputs.shape)
