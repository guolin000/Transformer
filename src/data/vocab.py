# src/data/vocab.py
import re
from collections import Counter

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD, UNK, BOS, EOS = SPECIAL_TOKENS

def tokenize(text, lang="en"):
    """中英文分词：英文按词，中文按字"""
    if lang == "en":
        return re.findall(r"[\w']+|[.,!?;]", text.lower())
    else:
        return list(text.replace(" ", ""))  # 中文去空格按字切分


def build_vocab(file_path, lang="en", min_freq=2, vocab_size=5000):
    counter = Counter()
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith(f"{lang}:"):
                sent = line[len(f"{lang}:"):].strip()
                tokens = tokenize(sent, lang)
                counter.update(tokens)
    # 按频率排序取前 vocab_size
    vocab = SPECIAL_TOKENS + [tok for tok, freq in counter.most_common(vocab_size - len(SPECIAL_TOKENS))]
    stoi = {tok: i for i, tok in enumerate(vocab)}
    itos = {i: tok for tok, i in stoi.items()}
    return stoi, itos
