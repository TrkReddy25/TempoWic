"""
Fine-tune a RoBERTa cross-encoder on the TempoWiC training split.

Input format:  <s> tweet1 (target marked with * ... *) </s></s> tweet2 </s>
Label:         1 = the two uses have the same meaning, 0 = different meaning.
Model selection: best macro-F1 on the TempoWiC validation split.
"""
import json, os, argparse, random, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def mark(use):
    t, a, b = use["text"], use["text_start"], use["text_end"]
    return t[:a] + "* " + t[a:b] + " *" + t[b:]


class PairData(Dataset):
    def __init__(self, pairs, tok, max_len):
        self.p, self.tok, self.max_len = pairs, tok, max_len

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        p = self.p[i]
        enc = self.tok(mark(p["use1"]), mark(p["use2"]), truncation="longest_first",
                       max_length=self.max_len, padding="max_length", return_tensors="pt")
        item = {k: v[0] for k, v in enc.items()}
        item["labels"] = torch.tensor(p["gold"] if p["gold"] is not None else 0)
        return item


def run_eval(model, loader):
    model.eval()
    probs, golds = [], []
    with torch.no_grad():
        for b in loader:
            y = b.pop("labels")
            logits = model(**b).logits
            probs.append(torch.softmax(logits, -1)[:, 1].numpy())
            golds.append(y.numpy())
    probs = np.concatenate(probs)
    golds = np.concatenate(golds)
    pred = (probs >= 0.5).astype(int)
    return {"macro_f1": float(f1_score(golds, pred, average="macro")),
            "f1_same": float(f1_score(golds, pred, pos_label=1)),
            "f1_diff": float(f1_score(golds, pred, pos_label=0)),
            "accuracy": float(accuracy_score(golds, pred))}, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/claude/models/roberta-base-spacy")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="/home/claude/models/tempowic-ft")
    args = ap.parse_args()

    torch.set_num_threads(2)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    pairs = [json.loads(l) for l in open(os.path.join(RES, "pairs.jsonl"))]
    tr = [p for p in pairs if p["split"] == "train"]
    va = [p for p in pairs if p["split"] == "validation"]
    te = [p for p in pairs if p["split"] == "test"]

    tok = PreTrainedTokenizerFast.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    dl_tr = DataLoader(PairData(tr, tok, args.max_len), batch_size=args.batch_size, shuffle=True)
    dl_va = DataLoader(PairData(va, tok, args.max_len), batch_size=32)
    dl_te = DataLoader(PairData(te, tok, args.max_len), batch_size=32)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(dl_tr) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    best = {"macro_f1": -1}
    history = []
    for ep in range(args.epochs):
        model.train(); t0 = time.time()
        for i, b in enumerate(dl_tr):
            loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if i % 20 == 0:
                print(f"  seed{args.seed} ep{ep} step {i}/{len(dl_tr)} loss {loss.item():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        vm, _ = run_eval(model, dl_va)
        history.append({"epoch": ep, "validation": vm})
        print(f"seed{args.seed} epoch {ep}: val macro-F1 {vm['macro_f1']:.4f}", flush=True)
        if vm["macro_f1"] > best["macro_f1"]:
            best = {**vm, "epoch": ep}
            os.makedirs(args.out_dir, exist_ok=True)
            model.save_pretrained(args.out_dir); tok.save_pretrained(args.out_dir)

    model = AutoModelForSequenceClassification.from_pretrained(args.out_dir)
    tm, te_probs = run_eval(model, dl_te)
    vm, va_probs = run_eval(model, dl_va)
    out = {"seed": args.seed, "args": vars(args), "best_epoch": best["epoch"],
           "validation": vm, "test": tm, "history": history}
    with open(os.path.join(RES, f"finetune_seed{args.seed}.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(os.path.join(RES, f"finetune_probs_seed{args.seed}.npz"),
                        test_id=np.array([p["pair_id"] for p in te], dtype=object), test_prob=te_probs,
                        val_id=np.array([p["pair_id"] for p in va], dtype=object), val_prob=va_probs)
    print(json.dumps({"seed": args.seed, "validation": vm, "test": tm}, indent=2))


if __name__ == "__main__":
    main()
