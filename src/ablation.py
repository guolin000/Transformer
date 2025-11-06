# src/ablation.py
import os
import time
import random
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import get_args
from src.utils.logger import get_logger
from src.data.vocab import build_vocab
from src.data.dataset import TranslationDataset, collate_fn
from src.models.transformer import Transformer
from src.utils.train_utils import train_epoch, evaluate

logger = get_logger(log_dir="../results", log_filename="ablation.log")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_experiment(args, exp_name, modify_model_fn=None):
    """
    通用实验函数
    modify_model_fn: 可选，传入一个函数用来修改 Transformer 配置（例如去掉位置编码）
    """
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"[{exp_name}] Using device: {device}")

    # 构建词表
    en_stoi, _ = build_vocab(args.data_path, "en", vocab_size=args.vocab_size)
    zh_stoi, _ = build_vocab(args.data_path, "zh", vocab_size=args.vocab_size)

    # 数据集
    train_dataset = TranslationDataset(args.data_path, en_stoi, zh_stoi)
    val_dataset = TranslationDataset(args.val_data_path, en_stoi, zh_stoi)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 初始化模型
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

    # 修改模型（消融实验）
    if modify_model_fn:
        modify_model_fn(model)

    optimizer = AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)

    def lr_lambda(step):
        d_model = args.d_model
        warmup_steps = getattr(args, "warmup_steps", 4000)
        step += 1
        return min(step ** -0.5, step * warmup_steps ** -1.5) * (d_model ** -0.5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    train_losses, val_losses = [], []
    times_per_epoch = []

    best_loss = float('inf')
    results_dir = f"../results/ablation_{exp_name}"
    os.makedirs(results_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - start_time

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        times_per_epoch.append(epoch_time)

        logger.info(f"[{exp_name}] epoch {epoch+1} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | time: {epoch_time:.2f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pt"))

    # 保存曲线
    epochs = range(1, args.num_epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Curve - {exp_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_curve.png"))
    plt.close()

    # 保存表格
    df = pd.DataFrame({
        "epoch": epochs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "epoch_time_s": times_per_epoch
    })
    df.to_csv(os.path.join(results_dir, "training_log.csv"), index=False)

    return train_losses, val_losses


if __name__ == "__main__":
    args = get_args()

    experiments = []

    # ======================
    # 消融实验1：去掉位置编码
    # ======================
    def remove_pos_encoding(model):
        model.pos_enc = torch.nn.Identity()

    experiments.append(("no_pos_enc", remove_pos_encoding))

    # ======================
    # 消融实验2：减小 Encoder/Decoder 层数
    # ======================
    def reduce_layers(model):
        device = next(model.parameters()).device  # 获取当前模型所在设备

        # 假设原本是 2 层，可以改成 1 层
        model.encoder = type(model.encoder)(
            num_layers=1,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            dropout=args.dropout
        ).to(device)

        model.decoder = type(model.decoder)(
            num_layers=1,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            dropout=args.dropout
        ).to(device)

    experiments.append(("reduce_layers", reduce_layers))

    # ======================
    # 运行实验
    # ======================
    all_results = {}
    for exp_name, fn in experiments:
        logger.info(f"=== 运行消融实验: {exp_name} ===")
        train_losses, val_losses = run_experiment(args, exp_name, modify_model_fn=fn)
        all_results[exp_name] = (train_losses, val_losses)

    # ======================
    # 对比绘图
    # ======================
    plt.figure(figsize=(8, 5))
    for exp_name, (train_losses, val_losses) in all_results.items():
        plt.plot(range(1, len(val_losses)+1), val_losses, label=f"{exp_name} Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Ablation Study - Validation Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../results/ablation_comparison.png")
    plt.close()
