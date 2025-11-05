# src/utils/mask.py
import torch

def create_padding_mask(seq):
    """
    seq: [batch, seq_len] (LongTensor)，pad token assumed to be 0
    return: bool mask of shape [batch, 1, 1, seq_len], True where pad
    """
    return (seq == 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, seq_len) dtype=bool

def create_subsequent_mask(size, device=None):
    """
    size: target seq length (int)
    return: bool mask of shape [1, 1, size, size], True where future positions (to be masked)
    """
    # upper triangular with diagonal=1 -> True above diagonal
    subsequent = torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
    return subsequent.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)

def create_tgt_mask(tgt_seq, device=None):
    """
    Combine padding mask and subsequent mask for target.
    tgt_seq: [batch, tgt_len]
    returns mask of shape (batch, 1, tgt_len, tgt_len), dtype=bool
    """
    if device is None:
        device = tgt_seq.device
    tgt_pad_mask = (tgt_seq == 0).unsqueeze(1).unsqueeze(2).to(device)  # (B,1,1,tgt_len)
    subsequent_mask = create_subsequent_mask(tgt_seq.size(1), device=device)  # (1,1,tgt_len,tgt_len)
    # broadcast OR -> shape (B,1,tgt_len,tgt_len)
    tgt_mask = tgt_pad_mask | subsequent_mask
    return tgt_mask
