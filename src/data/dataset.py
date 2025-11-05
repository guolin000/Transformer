# src/utils/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from data.vocab import tokenize, PAD, BOS, EOS

class TranslationDataset(Dataset):
    def __init__(self, path, src_vocab, tgt_vocab):
        self.pairs = []
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        src, tgt = None, None
        for line in lines:
            if line.startswith("en:"):
                src = line[3:].strip()
            elif line.startswith("zh:"):
                tgt = line[3:].strip()
            if src and tgt:
                self.pairs.append((src, tgt))
                src, tgt = None, None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        src_tokens = [BOS] + tokenize(src_text, "en") + [EOS]
        tgt_tokens = [BOS] + tokenize(tgt_text, "zh") + [EOS]

        src_ids = [self.src_vocab.get(tok, self.src_vocab["<unk>"]) for tok in src_tokens]
        tgt_ids = [self.tgt_vocab.get(tok, self.tgt_vocab["<unk>"]) for tok in tgt_tokens]

        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    pad_idx = 0
    max_src = max(len(s) for s in srcs)
    max_tgt = max(len(t) for t in tgts)

    src_batch = torch.full((len(batch), max_src), pad_idx)
    tgt_batch = torch.full((len(batch), max_tgt), pad_idx)

    for i, (src, tgt) in enumerate(zip(srcs, tgts)):
        src_batch[i, :len(src)] = src
        tgt_batch[i, :len(tgt)] = tgt

    return src_batch, tgt_batch
