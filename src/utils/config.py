# src/utils/config.py  或者你项目的 config 文件
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Transformer- Config")

    # 数据参数
    parser.add_argument('--data_path', type=str, default='data/ted_zh_en_train.txt', help='训练语料路径')
    parser.add_argument('--val_data_path', type=str, default='data/ted_zh_en_val.txt', help='验证语料路径')
    parser.add_argument('--vocab_size', type=int, default=5000, help='词表大小')
    parser.add_argument('--max_len', type=int, default=512, help='最大序列长度')

    # 模型参数
    parser.add_argument('--d_model', type=int, default=128, help='词向量维度')
    parser.add_argument('--num_heads', type=int, default=4, help='多头注意力头数')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Encoder层数')
    parser.add_argument('--num_decoder_layers', type=int, default=2, help='Decoder层数')
    parser.add_argument('--d_ff', type=int, default=512, help='FeedForward隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--num_epochs', type=int, default=40, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda:0', help='训练设备')

    args = parser.parse_args([])
    return args
