# src/models/transformer.py
# 支持双词表版本的 Transformer（英文 Encoder + 中文 Decoder）
import math

import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,       # 英文词表大小
                 tgt_vocab_size,       # 中文词表大小
                 d_model,
                 num_heads,
                 num_encoder_layers,
                 num_decoder_layers,
                 d_ff,
                 max_len=5000,
                 dropout=0.1):
        super(Transformer, self).__init__()

        self.d_model = d_model

        # === Embedding 层 ===
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # === 位置编码 ===
        self.pos_enc = PositionalEncoding(d_model, max_len)

        # === 编码器和解码器 ===
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)

        # === 输出层（映射到中文词表）===
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # === Dropout 层 ===
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: [batch_size, src_seq_len]
        tgt: [batch_size, tgt_seq_len]
        """

        # ====== Embedding + Position Encoding ======
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        src_emb = self.dropout(self.pos_enc(src_emb))
        tgt_emb = self.dropout(self.pos_enc(tgt_emb))

        # ====== Encoder ======
        enc_out = self.encoder(src_emb, src_mask)

        # ====== Decoder ======
        dec_out = self.decoder(tgt_emb, enc_out, tgt_mask, src_mask)

        # ====== 输出层 ======
        out = self.fc_out(dec_out)  # [batch_size, tgt_seq_len, tgt_vocab_size]

        # ✅ NaN 检查
        if torch.isnan(out).any():
            print("⚠️ forward 输出含 NaN，停止！")
            print("src_emb mean:", src_emb.mean().item(), "dec_out mean:", dec_out.mean().item())

        return out
