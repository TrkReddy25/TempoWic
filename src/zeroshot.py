"""
Zero-shot (unsupervised) relatedness scoring of use pairs.

Scores
  cos_L<k>    cosine similarity of the target-word vector at encoder layer k
  cos_sent    cosine similarity of mean-pooled sentence vectors (last layer)
  jaccard     content-word Jaccard overlap of the two tweets (lexical control)
  random      uniform random score (control)

The decision threshold (and the encoder layer) is selected on the TempoWiC
validation split with macro-F1, the official shared-task metric.
"""
import json, os, re, argparse
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
LAYERS = list(range(13))

STOP = set("""a an the and or but if of to in on at for with without from by as is are was were be been
being this that these those it its it's i you he she they we me him her them my your his their our not
no so do does did doing have has had just about into over after before up down out than then there here
what which who whom when where why how all any both each few more most other some such only own same too
very can will would should could may might must s t don now""".split())
TOKEN = re.compile(r"[A-Za-z']+")


def content_tokens(text):
    return {w.lower() for w in TOKEN.findall(text) if w.lower() not in STOP and len(w) > 2}


def cos_rows(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (A * B).sum(1)


def best_threshold(scores, gold):
    """Threshold maximising macro-F1 (predict 'same meaning' when score >= t)."""
    cand = np.unique(np.round(scores, 4))
    cand = np.concatenate([cand, (cand[:-1] + cand[1:]) / 2]) if len(cand) > 1 else cand
    best, bt = -1, float(np.median(scores))
    for t in cand:
        f = f1_score(gold, (scores >= t).astype(int), average="macro")
        if f > best:
            best, bt = f, float(t)
    return bt, best


def evaluate(scores, gold, t):
    pred = (scores >= t).astype(int)
    return {"macro_f1": float(f1_score(gold, pred, average="macro")),
            "f1_same": float(f1_score(gold, pred, pos_label=1)),
            "f1_diff": float(f1_score(gold, pred, pos_label=0)),
            "accuracy": float(accuracy_score(gold, pred))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)

    pairs = [json.loads(l) for l in open(os.path.join(RES, "pairs.jsonl"))]
    z = np.load(os.path.join(RES, "use_vectors.npz"), allow_pickle=True)
    keys = list(z["keys"])
    kidx = {k: i for i, k in enumerate(keys)}

    def uk(u):
        return f"{u['text_start']}|{u['text_end']}|{u['text']}"

    i1 = np.array([kidx[uk(p["use1"])] for p in pairs])
    i2 = np.array([kidx[uk(p["use2"])] for p in pairs])

    scores = {}
    for L in LAYERS:
        V = z[f"L{L}"]
        scores[f"cos_L{L}"] = cos_rows(V[i1], V[i2])
    S = z["sent"]
    scores["cos_sent"] = cos_rows(S[i1], S[i2])
    scores["jaccard"] = np.array([
        (lambda a, b: len(a & b) / len(a | b) if a | b else 0.0)(
            content_tokens(p["use1"]["text"]), content_tokens(p["use2"]["text"]))
        for p in pairs])
    scores["random"] = rng.rand(len(pairs))

    split = np.array([p["split"] for p in pairs])
    gold = np.array([-1 if p["gold"] is None else p["gold"] for p in pairs])
    val = split == "validation"
    tr = split == "train"
    te = split == "test"

    results = {}
    for name, sc in scores.items():
        t, vf = best_threshold(sc[val], gold[val])
        results[name] = {"threshold": t,
                         "validation": evaluate(sc[val], gold[val], t),
                         "train": evaluate(sc[tr], gold[tr], t),
                         "test": evaluate(sc[te], gold[te], t)}
    # majority-class baseline
    maj = int(np.bincount(gold[tr]).argmax())
    for nm, m in (("majority", maj),):
        pred_const = np.full(te.sum(), m)
        results[nm] = {"threshold": None,
                       "test": {"macro_f1": float(f1_score(gold[te], pred_const, average="macro")),
                                "f1_same": float(f1_score(gold[te], pred_const, pos_label=1)),
                                "f1_diff": float(f1_score(gold[te], pred_const, pos_label=0)),
                                "accuracy": float(accuracy_score(gold[te], pred_const))}}

    best_layer = max((k for k in results if k.startswith("cos_L")),
                     key=lambda k: results[k]["validation"]["macro_f1"])
    results["_best_layer"] = best_layer
    with open(os.path.join(RES, "zeroshot_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    np.savez_compressed(os.path.join(RES, "pair_scores_zeroshot.npz"),
                        pair_id=np.array([p["pair_id"] for p in pairs], dtype=object),
                        **scores)

    print(f"{'system':>10} {'val-F1':>7} {'test-F1':>8} {'test-acc':>9}  thr")
    for k in [f"cos_L{L}" for L in LAYERS] + ["cos_sent", "jaccard", "random"]:
        r = results[k]
        print(f"{k:>10} {r['validation']['macro_f1']:7.3f} {r['test']['macro_f1']:8.3f} "
              f"{r['test']['accuracy']:9.3f}  {r['threshold']:.3f}")
    print("majority test macro-F1:", round(results["majority"]["test"]["macro_f1"], 3))
    print("best layer on validation:", best_layer)


if __name__ == "__main__":
    main()
