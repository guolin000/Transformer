# src/models/decoder.py
# 多层 Decoder 堆叠

import torch.nn as nn
from .decoder_block import DecoderBlock


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, memory_mask)
        return x
