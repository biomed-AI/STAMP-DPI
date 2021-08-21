# -*- encoding: utf-8 -*-
'''
@Time       : 2020/07/18 13:53:49
@Author     : Yize Jiang
@Email      : yize.jiang@galixir.com
@File       : gnn.py
@Project    : X-DPI
@Description: 图神经网络
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=True, normalize_embedding=False, dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant(self.bias.data, 0.0)

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y


class ResGCN(nn.Module):
    def __init__(self, args, in_dim, out_dim):
        super(ResGCN, self).__init__()
        self.res_conv1 = GraphConv(in_dim, out_dim)
        self.res_conv2 = GraphConv(out_dim, out_dim)
        self.res_conv3 = GraphConv(out_dim, out_dim)

    def forward(self, inputs, adj):
        outputs = self.res_conv1(inputs, adj)
        outputs = self.res_conv2(outputs, adj)
        outputs = self.res_conv3(outputs, adj)
        return outputs


class SimpleGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleGCN, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, inputs, adj):
        return torch.bmm(adj, self.fc(inputs))


class MultiGCN(nn.Module):
    def __init__(self, args, in_dim, out_dim):
        super(MultiGCN, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim, bias=True)
        self.fc2 = nn.Linear(out_dim, out_dim, bias=True)
        self.fc3 = nn.Linear(out_dim, out_dim, bias=True)

    def forward(self, inputs, adj):
        outputs = torch.bmm(adj, self.fc1(inputs))
        outputs = torch.bmm(adj, self.fc2(outputs))
        outputs = torch.bmm(adj, self.fc3(outputs))
        return outputs
