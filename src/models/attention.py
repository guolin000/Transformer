# src/models/attention.py  (或替换原 MultiHeadAttention 类)
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, q, k, v, mask=None):
        """
        q: (B, Lq, d_model)
        k: (B, Lk, d_model)
        v: (B, Lv, d_model)
        mask: broadcastable bool mask where True indicates positions to mask.
              expected shapes: (B,1,1,Lk) or (1,1,Lq,Lk) or (B,1,Lq,Lk)
        """
        B, Lq, _ = q.size()
        Lk = k.size(1)
        Lv = v.size(1)

        # linear -> (B, L, h, d_k) -> transpose -> (B, h, L, d_k)
        Q = self.W_q(q).view(B, Lq, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, Lk, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, Lv, self.num_heads, self.d_k).transpose(1, 2)

        # scores: (B, h, Lq, Lk)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            # Ensure mask is bool and on same device
            mask = mask.to(scores.device)
            mask = mask.bool()

            # Mask shape can be (B,1,1,Lk), (1,1,Lq,Lk), (B,1,Lq,Lk) etc.
            # Make sure it's broadcastable to (B, self.num_heads, Lq, Lk)
            # If mask has no head dim, it will broadcast over heads automatically.
            # Use masked_fill with mask==True -> fill with -inf for masked positions
            scores = scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, h, Lq, d_k)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.fc(out)
        return out
