"""
Logistic regression over contextual-similarity features -- the same family as the
strongest official TempoWiC baseline (logistic regression on TimeLMs similarities).

Features per use pair: cosine of the target-word vectors at every encoder layer,
cosine of the mean-pooled sentence vectors, and lexical (Jaccard) overlap.
Trained on the TempoWiC training split, thresholded at 0.5, evaluated on
validation and test; probabilities are produced for all 10,097 pairs, so the
model can also act as a relatedness "annotator" in the DURel analysis.
"""
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

pairs = [json.loads(l) for l in open(os.path.join(RES, "pairs.jsonl"))]
sc = np.load(os.path.join(RES, "pair_scores_zeroshot.npz"), allow_pickle=True)
feat_names = [f"cos_L{i}" for i in range(13)] + ["cos_sent", "jaccard"]
X = np.column_stack([sc[f] for f in feat_names])
split = np.array([p["split"] for p in pairs])
gold = np.array([-1 if p["gold"] is None else p["gold"] for p in pairs])

tr, va, te = split == "train", split == "validation", split == "test"
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
clf.fit(X[tr], gold[tr])
prob = clf.predict_proba(X)[:, 1]


def ev(mask):
    pred = (prob[mask] >= 0.5).astype(int)
    return {"macro_f1": float(f1_score(gold[mask], pred, average="macro")),
            "f1_same": float(f1_score(gold[mask], pred, pos_label=1)),
            "f1_diff": float(f1_score(gold[mask], pred, pos_label=0)),
            "accuracy": float(accuracy_score(gold[mask], pred))}


out = {"features": feat_names, "train": ev(tr), "validation": ev(va), "test": ev(te),
       "coefficients": dict(zip(feat_names,
                                clf.named_steps["logisticregression"].coef_[0].round(4).tolist()))}
with open(os.path.join(RES, "simlr_results.json"), "w") as f:
    json.dump(out, f, indent=2)
np.savez_compressed(os.path.join(RES, "simlr_probs.npz"),
                    pair_id=np.array([p["pair_id"] for p in pairs], dtype=object), prob=prob)
print(json.dumps({k: out[k] for k in ["validation", "test"]}, indent=2))
print("top features:", sorted(out["coefficients"].items(), key=lambda kv: -abs(kv[1]))[:5])
