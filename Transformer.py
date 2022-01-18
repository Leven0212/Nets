# !/usr/bin/env python
# -*- coding:utf-8 -*-　
# Author: leven
# File: Transformer.py
# Date: 2021/12/25 0025
import math
import torch.functional as F
import torch
import copy
import torch.nn as nn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * torch.sqrt(self.d_model)


class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(10000.0)) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MutiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        # h表示head的数量
        super(MutiHeadAttention, self).__init__()
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)
        query, key, value = [f(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2) for f, x in
                             zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h, self.d_k)
        return self.linears[-1](x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, ):
        super(Transformer, self).__init__()
