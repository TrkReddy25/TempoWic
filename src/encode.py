"""
Encode every unique tweet use with the RoBERTa encoder and cache the
contextual vector of the target word at several layers.

A "use" of a target word w is a tweet containing w, following DURel's notion of
a word use.  The vector of the target token is the mean over its sub-word
pieces (RoBERTa byte-level BPE frequently splits Twitter tokens).
"""
import json, os, argparse
import numpy as np
import torch
from transformers import AutoModel, PreTrainedTokenizerFast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def use_key(u):
    return f"{u['text_start']}|{u['text_end']}|{u['text']}"


def collect_uses(pairs_path):
    uses = {}
    with open(pairs_path) as f:
        for line in f:
            p = json.loads(line)
            for u in (p["use1"], p["use2"]):
                uses.setdefault(use_key(u), u)
    return uses


def target_indices(offsets, start, end):
    """Sub-word token positions overlapping the target character span."""
    idx = [i for i, (a, b) in enumerate(offsets)
           if not (a == b == 0) and a < end and b > start]
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/claude/models/roberta-base-spacy")
    ap.add_argument("--pairs", default=os.path.join(RES, "pairs.jsonl"))
    ap.add_argument("--out", default=os.path.join(RES, "use_vectors.npz"))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    torch.set_num_threads(2)
    tok = PreTrainedTokenizerFast.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, output_hidden_states=True).eval()

    uses = collect_uses(args.pairs)
    keys = sorted(uses)
    print(f"{len(keys)} unique uses to encode")

    vecs = {L: np.zeros((len(keys), 768), dtype=np.float32) for L in LAYERS}
    sent_vecs = np.zeros((len(keys), 768), dtype=np.float32)   # mean-pooled last layer

    for s in range(0, len(keys), args.batch_size):
        batch = keys[s:s + args.batch_size]
        texts = [uses[k]["text"] for k in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_len, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        with torch.no_grad():
            out = model(**enc)
        hs = out.hidden_states
        for bi, k in enumerate(batch):
            u = uses[k]
            idx = target_indices(offs[bi].tolist(), u["text_start"], u["text_end"])
            if not idx:                       # target truncated away -> use CLS
                idx = [0]
            for L in LAYERS:
                vecs[L][s + bi] = hs[L][bi, idx].mean(0).numpy()
            mask = enc["attention_mask"][bi].bool()
            sent_vecs[s + bi] = hs[12][bi][mask].mean(0).numpy()
        if (s // args.batch_size) % 25 == 0:
            print(f"  {s}/{len(keys)}", flush=True)

    np.savez_compressed(args.out, keys=np.array(keys, dtype=object),
                        sent=sent_vecs, **{f"L{L}": vecs[L] for L in LAYERS})
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
