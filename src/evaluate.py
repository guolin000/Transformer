# src/evaluate.py
import torch
import math
from src.models.transformer import Transformer
from src.data.vocab import build_vocab, tokenize
from src.utils.mask import create_padding_mask, create_tgt_mask  # create_tgt_mask: pad | subsequent
from src.utils.config import get_args

@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    """
    model: Transformer
    src: [1, src_len] token ids (long)
    src_mask: padding mask for src (B,1,1,src_len)
    start_symbol: index of <bos> in target vocab (int)
    end_symbol: index of <eos> in target vocab (int)
    """
    # encoder input must be embedding + posenc (same as training)
    src_emb = model.src_embedding(src) * math.sqrt(model.d_model)
    src_emb = model.dropout(model.pos_enc(src_emb))
    memory = model.encoder(src_emb, src_mask)  # memory: (1, src_len, d_model)

    ys = torch.full((1, 1), start_symbol, dtype=src.dtype, device=device)  # (1,1)
    for _ in range(max_len - 1):
        # build tgt_mask combining padding and subsequent mask
        tgt_mask = create_tgt_mask(ys, device=device)  # (B,1,tgt_len,tgt_len)

        # embed entire ys sequence each step (consistent with training)
        tgt_emb = model.tgt_embedding(ys) * math.sqrt(model.d_model)
        tgt_emb = model.dropout(model.pos_enc(tgt_emb))

        out = model.decoder(tgt_emb, memory, tgt_mask, src_mask)  # (1, tgt_len, d_model)
        logits = model.fc_out(out[:, -1, :])  # take last time step -> (1, vocab)
        next_word = torch.argmax(logits, dim=-1).item()

        ys = torch.cat([ys, torch.tensor([[next_word]], dtype=ys.dtype, device=device)], dim=1)

        if next_word == end_symbol:
            break

    return ys


def translate_sentence(sentence, model, en_stoi, zh_stoi, zh_itos, device, max_len=50):
    tokens = tokenize(sentence, "en")
    # add BOS/EOS here if your tokenize doesn't; but since greedy_decode uses start_symbol, we don't add bos here
    src_indices = [en_stoi.get(tok, en_stoi.get("<unk>", 0)) for tok in tokens]
    src = torch.tensor([src_indices], dtype=torch.long, device=device)  # shape (1, src_len)
    src_mask = create_padding_mask(src).to(device)

    start_symbol = zh_stoi.get("<bos>", None)
    end_symbol = zh_stoi.get("<eos>", None)
    if start_symbol is None or end_symbol is None:
        raise KeyError("zh_stoi must contain <bos> and <eos> tokens")

    tgt_tokens = greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device)
    decoded = [zh_itos[i.item()] for i in tgt_tokens[0]]
    # remove bos/eos if present
    # join tokens — if your zh_itos are characters/words adjust accordingly
    # return without <bos> and <eos>
    if decoded and decoded[0] == "<bos>":
        decoded = decoded[1:]
    if decoded and decoded[-1] == "<eos>":
        decoded = decoded[:-1]
    return "".join(decoded)


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    en_stoi, _ = build_vocab(args.data_path, "en")
    zh_stoi, zh_itos = build_vocab(args.data_path, "zh")

    model = Transformer(src_vocab_size=len(en_stoi),
                        tgt_vocab_size=len(zh_itos),
                        d_model=args.d_model,
                        num_heads=args.num_heads,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_decoder_layers,
                        d_ff=args.d_ff,
                        dropout=args.dropout,
                        max_len=args.max_len).to(device)

    # 推荐使用 weights_only=True 如果可用
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    model.eval()

    print("请输入英文句子（输入 quit 退出）：")
    while True:
        sentence = input(">> ")
        if sentence.strip().lower() == "quit":
            break
        try:
            print("翻译结果：", translate_sentence(sentence, model, en_stoi, zh_stoi, zh_itos, device))
        except Exception as e:
            print("翻译出错：", e)


if __name__ == "__main__":
    main()
