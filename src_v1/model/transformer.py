# -*- encoding: utf-8 -*-
'''
@Time       : 2020/07/16 15:35:33
@Author     : Yize Jiang
@Email      : yize.jiang@galixir.com
@File       : transformer.py
@Project    : X-DPI
@Description: TransformerCPI模型结构
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = math.sqrt(hid_dim // n_heads)

    def forward(self, query, key, value, mask=None):
        # query = key = value [batch size, sent_len, hid_dim]
        bsz = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q, K, V = [batch_size, sent_len, hid_dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        # K, V = [batch_size, n_heads, sent_len_K, hid_dim / n_heads]
        # Q = [batch_size, n_heads, sent_len_Q, hid_dim / n_heads]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch_size, n_heads, sent_len_Q, sent_len_K]

        # if mask is not None:
        #     energy = energy.masked_fill(mask == 0, float('-inf'))
        #     attention = self.do(F.softmax(energy, dim=-1))
        #     attention = attention.masked_fill(mask == 0, 0)
        # else:
        #     attention = self.do(F.softmax(energy, dim=-1))
        energy = energy.masked_fill(mask == 0, float('-inf'))
        attention = self.do(F.softmax(energy, dim=-1))
        # attention = attention.masked_fill(mask == 0, 0)
        # attention = attention / torch.sum(attention, dim=-1, keepdim=True)
        
        # attention = [batch_size, n_heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)
        # x = [batch_size, n_heads, sent_len_Q, hid_dim / n_heads]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch_size, sent_len_Q, n_heads, hid_dim / n_heads]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = [batch_size, sent_len_Q, hid_dim]
        x = self.fc(x)
        # x = [batch_size, sent_len_Q, hid_dim]

        return x


class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        # self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.scale = math.sqrt(0.5)
        self.convs = nn.ModuleList([
            nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2) for _ in range(self.n_layers)
        ])  # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        # pos = torch.arange(0, protein.shape[1]).unsqueeze(0).repeat(protein.shape[0], 1).to(self.device)
        # protein = protein + self.pos_embedding(pos)
        # protein = [batch_size, protein_len, protein_dim]

        conv_input = self.fc(protein)
        # conv_input=[batch_size, protein_len, hid_dim]

        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        # conv_input = [batch_size, hid_dim, protein_len]

        for i, conv in enumerate(self.convs):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            # conved = [batch_size, 2 * hid_dim, protein_len]
            # pass through GLU activation function
            conved = F.glu(conved, dim=1)
            # conved = [batch_size, hid_dim, protein_len]
            # apply residual connection / high way
            conved = (conved + conv_input) * self.scale
            # conved = [batch_size, hid_dim, protein_len]
            # set conv_input to conved for next loop iteration
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        # conved = [batch_size, protein_len, hid_dim]
        conved = self.ln(conved)
        return conved


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch_size, sent_len, hid_dim]

        x = x.permute(0, 2, 1)
        # x = [batch_size, hid_dim, sent_len]
        x = self.do(F.relu(self.fc_1(x)))
        # x = [batch_size, pf_dim, sent_len]
        x = self.fc_2(x)
        # x = [batch_size, hid_dim, sent_len]
        x = x.permute(0, 2, 1)
        # x = [batch_size, sent_len, hid_dim]
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout)
        self.ea = SelfAttention(hid_dim, n_heads, dropout)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound_len, atom_dim]
        # src = [batch_size, protein_len, hid_dim] # encoder output
        # trg_mask = [batch_size, compound_len]
        # src_mask = [batch_size, protein_len]
        # trg = torch.mul(trg, trg_mask.squeeze())
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        trg = self.ln(trg + self.do(self.pf(trg)))
        return trg


class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, dropout):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.dropout = dropout
        self.sa = SelfAttention(hid_dim, n_heads, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout)
            for _ in range(n_layers)
        ])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim, 256)
        self.gn = nn.GroupNorm(8, 256)

    def forward(self, trg, src, trg_mask=None, src_mask=None, line=None):
        # trg = [batch_size, compound_len, atom_dim]
        # src = [batch_size, protein_len, hid_dim] # encoder output

        trg = self.ft(trg)
        # trg = [batch_size, compound_len, hid_dim]

        for i, layer in enumerate(self.layers):
            trg = layer(trg, src, trg_mask, src_mask)
        # trg = [batch_size, compound_len, hid_dim]
        """Use norm to determine which atom is significant. """
        norm = torch.norm(trg, dim=2)
        # norm = [batch_size, compound_len]
        # if trg_mask is not None:
        #     trg_mask = trg_mask.squeeze()
        #     norm = norm.masked_fill(trg_mask == 0, float('-inf'))
        trg_mask = trg_mask.squeeze()
        norm = norm.masked_fill(trg_mask == 0, float('-inf'))
        norm = F.softmax(norm, dim=1)
        # norm = norm.masked_fill(trg_mask == 0, 0)
        # norm = norm / torch.sum(norm, dim=-1, keepdim=True)
        # norm = [batch_size, compound_len]

        # trg = torch.squeeze(trg,dim=0)
        # norm = torch.squeeze(norm,dim=0)

        # sum = torch.zeros((trg.shape[0], self.hid_dim)).type_as(norm)
        # for i in range(norm.shape[0]):
        #     for j in range(norm.shape[1]):
        #         v = trg[i, j, ]
        #         v = v * norm[i, j]
        #         sum[i, ] += v

        sum = torch.bmm(norm.unsqueeze(1), trg).squeeze(1)
        # sum = [batch_size, hid_dim]

        outputs = F.relu(self.fc(sum))
        return outputs
