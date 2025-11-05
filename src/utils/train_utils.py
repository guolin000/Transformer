# src/utils/train_utils.py
import torch
import torch.nn as nn
from src.utils.mask import create_padding_mask, create_tgt_mask,create_subsequent_mask



def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = create_padding_mask(src).to(device)  # (B,1,1,src_len)
        tgt_mask = create_tgt_mask(tgt_input, device=device)  # (B,1,tgt_len,tgt_len)

        logits = model(src, tgt_input, src_mask, tgt_mask)

        # print("logits shape:", logits.shape)
        # print("tgt_output max/min:", tgt_output.max().item(), tgt_output.min().item())

        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        if torch.isnan(logits).any():
            print("⚠️ logits 出现 NaN！")
            break
        if torch.isnan(loss):
            print("⚠️ loss 出现 NaN！")
            break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item()
        # print("梯度范数:", grad_norm)

        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = create_padding_mask(src)
        tgt_mask = create_subsequent_mask(tgt_input.size(1)).to(device)

        logits = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        total_loss += loss.item()
    return total_loss / len(dataloader)
