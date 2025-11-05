# src/main_train.py
import os
import time
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.config import get_args
from src.data.vocab import build_vocab
from src.utils.dataset import TranslationDataset, collate_fn
from src.models.transformer import Transformer
from src.utils.train_utils import train_epoch, evaluate


# =====================
# 设置随机种子
# =====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================
# 保存训练结果
# =====================
def save_training_results(results_dir, train_losses, val_losses, times_per_epoch):
    os.makedirs(results_dir, exist_ok=True)

    epochs = list(range(1, len(train_losses) + 1))
    df = pd.DataFrame({
        "epoch": epochs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "epoch_time_s": times_per_epoch
    })
    csv_path = os.path.join(results_dir, "training_log.csv")
    xlsx_path = os.path.join(results_dir, "training_log.xlsx")
    json_path = os.path.join(results_dir, "training_log.json")

    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as e:
        print("保存 Excel 失败", e)
    df.to_json(json_path, orient="records", force_ascii=False)

    # =====================
    # 绘制训练曲线图
    # =====================
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    img_path = os.path.join(results_dir, "training_curve.png")
    plt.savefig(img_path)
    plt.close()

    return {
        "csv": csv_path,
        "excel": xlsx_path,
        "json": json_path,
        "image": img_path
    }


# =====================
# 参数统计函数
# =====================
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"* 模型参数总数: {total_params:,}")
    print(f"* 可训练参数数: {trainable_params:,}")
    return total_params, trainable_params


def main():
    print("CUDA available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        try:
            print("current device:", torch.cuda.current_device())
            print("device name:", torch.cuda.get_device_name(0))
        except Exception:
            pass

    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("==> 构建词表（基于训练集）...")
    en_stoi, _ = build_vocab(args.data_path, "en", vocab_size=args.vocab_size)
    zh_stoi, _ = build_vocab(args.data_path, "zh", vocab_size=args.vocab_size)

    print("==> 加载训练/验证数据集...")
    train_dataset = TranslationDataset(args.data_path, en_stoi, zh_stoi)
    val_dataset = TranslationDataset(args.val_data_path, en_stoi, zh_stoi)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print("==> 初始化模型...")
    model = Transformer(
        src_vocab_size=len(en_stoi),
        tgt_vocab_size=len(zh_stoi),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len
    ).to(device)

    # =====================
    # 参数统计
    # =====================
    count_parameters(model)

    # =====================
    # 优化器 AdamW
    # =====================
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)

    # =====================
    # 学习率调度器
    # Transformer 经典：warmup + decay
    # =====================
    def lr_lambda(step):
        warmup = 4000
        if step == 0:
            step = 1
        return (args.d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # =====================
    # 损失函数
    # =====================
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    best_loss = float('inf')
    os.makedirs("checkpoints", exist_ok=True)
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    train_losses, val_losses, times_per_epoch = [], [], []

    print("==> 开始训练...")
    total_steps = 0

    for epoch in range(args.num_epochs):
        start_time = time.time()

        model.train()
        total_loss = 0.0
        for i, batch in enumerate(train_loader):
            src, tgt_input, tgt_output = [x.to(device) for x in batch]

            optimizer.zero_grad()
            output = model(src, tgt_input)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
            loss.backward()

            # =====================
            # 梯度裁剪
            # =====================
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()  # 每步更新学习率
            total_steps += 1

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time
        times_per_epoch.append(round(epoch_time, 3))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"epoch [{epoch+1}/{args.num_epochs}] | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | time: {epoch_time:.2f}s | lr: {scheduler.get_last_lr()[0]:.6f}")

        # =====================
        # 保存最优模型
        # =====================
        if val_loss < best_loss:
            best_loss = val_loss
            best_path = os.path.join("checkpoints", "best_model.pt")
            torch.save(model.state_dict(), best_path)
            with open(os.path.join(results_dir, "best_info.txt"), "w", encoding="utf-8") as f:
                f.write(f"best_epoch: {epoch+1}\n")
                f.write(f"best_val_loss: {best_loss:.6f}\n")
                f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    saved = save_training_results(results_dir, train_losses, val_losses, times_per_epoch)
    print("\n==>训练完成，结果文件：")
    for k, v in saved.items():
        print(f"  {k}: {v}")
    print("最佳模型已保存到 checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
