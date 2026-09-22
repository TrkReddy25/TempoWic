"""Score every use pair (COMPARE, EARLIER, LATER) with the fine-tuned cross-encoder."""
import json, os, argparse
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune import PairData

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

ap = argparse.ArgumentParser()
ap.add_argument("--model_dir", default="/home/claude/models/tempowic-ft")
ap.add_argument("--max_len", type=int, default=128)
ap.add_argument("--batch_size", type=int, default=32)
a = ap.parse_args()

torch.set_num_threads(2)
pairs = [json.loads(l) for l in open(os.path.join(RES, "pairs.jsonl"))]
tok = PreTrainedTokenizerFast.from_pretrained(a.model_dir)
model = AutoModelForSequenceClassification.from_pretrained(a.model_dir).eval()
dl = DataLoader(PairData(pairs, tok, a.max_len), batch_size=a.batch_size)

probs = []
with torch.no_grad():
    for i, b in enumerate(dl):
        b.pop("labels")
        probs.append(torch.softmax(model(**b).logits, -1)[:, 1].numpy())
        if i % 25 == 0:
            print(f"  {i*a.batch_size}/{len(pairs)}", flush=True)
probs = np.concatenate(probs)
np.savez_compressed(os.path.join(RES, "finetune_all_probs.npz"),
                    pair_id=np.array([p["pair_id"] for p in pairs], dtype=object), prob=probs)
print("saved finetune_all_probs.npz")
