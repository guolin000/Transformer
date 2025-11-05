"""
prepare_iwslt_zh_en.py
---------------------------------
整理本地 IWSLT2017 en-zh 数据集，生成手工 Transformer 可用的训练/验证集
并输出所有样本文件
---------------------------------
"""

import os
import random
import re

# =========================
# 配置
# =========================
DATA_DIR = "en-zh"  # 解压后的 en-zh 文件夹
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALL_FILE = os.path.join(SCRIPT_DIR, "ted_zh_en_all.txt")
TRAIN_FILE = os.path.join(SCRIPT_DIR, "ted_zh_en_train.txt")
VAL_FILE = os.path.join(SCRIPT_DIR, "ted_zh_en_val.txt")
NUM_SAMPLES = 20000
VAL_RATIO = 0.1
SEED = 42
MIN_LEN = 2
MAX_LEN = 100

# =========================
# 工具函数
# =========================
def clean_line(line):
    line = line.strip()
    # 去掉 XML 标签
    line = re.sub(r"<[^>]+>", "", line)
    return line

def read_parallel_file(en_path, zh_path):
    en_lines, zh_lines = [], []
    with open(en_path, encoding="utf-8") as f_en, open(zh_path, encoding="utf-8") as f_zh:
        for en_line, zh_line in zip(f_en, f_zh):
            en_line, zh_line = clean_line(en_line), clean_line(zh_line)
            if not en_line or not zh_line:
                continue
            if MIN_LEN < len(en_line.split()) < MAX_LEN and MIN_LEN < len(zh_line.split()) < MAX_LEN:
                en_lines.append(en_line)
                zh_lines.append(zh_line)
    assert len(en_lines) == len(zh_lines)
    print(f"读取并过滤 {en_path} + {zh_path} 句对数量: {len(en_lines)}")
    return list(zip(en_lines, zh_lines))

def collect_training_data(data_dir):
    train_en = os.path.join(data_dir, "train.tags.en-zh.en")
    train_zh = os.path.join(data_dir, "train.tags.en-zh.zh")
    return read_parallel_file(train_en, train_zh)

def collect_validation_data(data_dir):
    val_pairs = []
    for year in range(2010, 2016):
        en_file = os.path.join(data_dir, f"IWSLT17.TED.dev{year}.en-zh.en.xml")
        zh_file = os.path.join(data_dir, f"IWSLT17.TED.dev{year}.en-zh.zh.xml")
        if os.path.exists(en_file) and os.path.exists(zh_file):
            val_pairs += read_parallel_file(en_file, zh_file)
    return val_pairs

def save_pairs(pairs, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for en, zh in pairs:
            f.write(f"en: {en}\n")
            f.write(f"zh: {zh}\n")
    print(f"✅ 保存: {file_path} ({len(pairs)} 句对)")

def save_train_val(pairs, train_file, val_file, val_ratio=0.1, seed=42):
    random.seed(seed)
    random.shuffle(pairs)
    val_size = int(len(pairs) * val_ratio)
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]

    save_pairs(train_pairs, train_file)
    save_pairs(val_pairs, val_file)

# =========================
# 主函数
# =========================
def prepare_iwslt_zh_en_local(data_dir=DATA_DIR, num_samples=NUM_SAMPLES):
    train_pairs = collect_training_data(data_dir)
    val_pairs = collect_validation_data(data_dir)
    all_pairs = train_pairs + val_pairs
    print(f"🎯 总句对数量: {len(all_pairs)}")

    # 使用随机种子抽样
    if len(all_pairs) > num_samples:
        random.seed(SEED)
        all_pairs = random.sample(all_pairs, num_samples)
        print(f"🎯 抽样 {len(all_pairs)} 个句对")

    # 保存所有样本
    save_pairs(all_pairs, ALL_FILE)
    # 保存训练/验证集
    save_train_val(all_pairs, TRAIN_FILE, VAL_FILE, val_ratio=VAL_RATIO, seed=SEED)

    # 示例预览
    print("🌏 示例预览：")
    with open(TRAIN_FILE, encoding="utf-8") as f:
        for _ in range(6):
            print(next(f).strip())

if __name__ == "__main__":
    prepare_iwslt_zh_en_local()
