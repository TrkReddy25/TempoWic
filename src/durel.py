"""
DURel measures (Schlechtweg et al., 2018) computed on TempoWiC English.

For every target word w and every relatedness model m:
    Mean_e(w)  mean relatedness of w's EARLIER use pairs   (both uses from t1)
    Mean_l(w)  mean relatedness of w's LATER   use pairs   (both uses from t2)
    Mean_c(w)  mean relatedness of w's COMPARE use pairs   (one use per period)

    DELTA_LATER(w)   = Mean_l(w) - Mean_e(w)      innovative (<0) vs reductive (>0)
    COMPARE(w)       = Mean_c(w)                  low = strong change
    DELTA_COMPARE(w) = Mean_c(w) - Mean_e(w)      polysemy-normalised change

Relatedness is mapped onto DURel's 4-point scale (1 = Unrelated ... 4 = Identical):
    fine-tuned classifier : rel = 1 + 3 * P(same meaning)
    cosine similarity     : rel = 1 + 3 * minmax(cos)   (monotone; rankings unchanged)

Gold reference per word: the share of human-labelled COMPARE pairs judged
'different meaning' (= observed meaning shift in TempoWiC).
"""
import json, os, argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def minmax(x):
    lo, hi = np.min(x), np.max(x)
    return (x - lo) / (hi - lo + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="score columns to use; default = auto")
    args = ap.parse_args()

    pairs = [json.loads(l) for l in open(os.path.join(RES, "pairs.jsonl"))]
    df = pd.DataFrame([{ "pair_id": p["pair_id"], "word": p["word"], "split": p["split"],
                         "group": p["group"], "gold": p["gold"]} for p in pairs])

    z = np.load(os.path.join(RES, "pair_scores_zeroshot.npz"), allow_pickle=True)
    assert list(z["pair_id"]) == list(df["pair_id"]), "score/pair misalignment"
    zs = json.load(open(os.path.join(RES, "zeroshot_results.json")))
    best_layer = zs["_best_layer"]

    score_cols = {}
    score_cols[best_layer] = z[best_layer]
    for extra in ["cos_L8", "cos_L12", "cos_sent", "jaccard"]:
        if extra in z.files and extra not in score_cols:
            score_cols[extra] = z[extra]

    # supervised models scored over all pairs, if available
    for tag, fn in (("simlr", "simlr_probs.npz"), ("finetuned", "finetune_all_probs.npz")):
        path = os.path.join(RES, fn)
        if os.path.exists(path):
            f = np.load(path, allow_pickle=True)
            pid2p = dict(zip(list(f["pair_id"]), f["prob"]))
            score_cols[tag] = np.array([pid2p[i] for i in df["pair_id"]])

    for name, sc in score_cols.items():
        df[name] = sc
        df["rel_" + name] = 1 + 3 * (sc if name in ("finetuned", "simlr") else minmax(sc))

    df.to_csv(os.path.join(RES, "pair_scores.csv"), index=False)

    rel_cols = ["rel_" + n for n in score_cols]
    rows = []
    for w, g in df.groupby("word"):
        comp = g[g.group == "COMPARE"]
        row = {"word": w, "split": comp["split"].iloc[0],
               "n_compare": len(comp), "n_earlier": int((g.group == "EARLIER").sum()),
               "n_later": int((g.group == "LATER").sum()),
               "gold_change_rate": float(1 - comp["gold"].mean())}
        for rc in rel_cols:
            me = g.loc[g.group == "EARLIER", rc].mean()
            ml = g.loc[g.group == "LATER", rc].mean()
            mc = g.loc[g.group == "COMPARE", rc].mean()
            m = rc[4:]
            row[f"Mean_e[{m}]"] = me
            row[f"Mean_l[{m}]"] = ml
            row[f"Mean_c[{m}]"] = mc
            row[f"DELTA_LATER[{m}]"] = ml - me
            row[f"COMPARE[{m}]"] = mc
            row[f"DELTA_COMPARE[{m}]"] = mc - me
        rows.append(row)
    words = pd.DataFrame(rows).sort_values(f"DELTA_LATER[{list(score_cols)[-1]}]")
    words.to_csv(os.path.join(RES, "durel_measures.csv"), index=False)

    # how well do the DURel change measures track the human change rate?
    corr = {}
    for m in score_cols:
        c = {}
        for meas in ["COMPARE", "DELTA_COMPARE", "DELTA_LATER"]:
            r, p = spearmanr(words[f"{meas}[{m}]"], words["gold_change_rate"])
            c[meas] = {"spearman": float(r), "p": float(p)}
        corr[m] = c

    # agreement between relatedness models (and gold) on labelled COMPARE pairs
    lab = df[(df.group == "COMPARE") & df.gold.notna()]
    ann = list(score_cols) + ["gold"]
    mat = pd.DataFrame(index=ann, columns=ann, dtype=float)
    for a in ann:
        for b in ann:
            x = lab["gold"] if a == "gold" else lab[a]
            y = lab["gold"] if b == "gold" else lab[b]
            mat.loc[a, b] = spearmanr(x, y).statistic
    mat.to_csv(os.path.join(RES, "agreement_matrix.csv"))

    # each model vs. the average of the other models (DURel Table 3, bottom row)
    avg_row = {}
    for a in score_cols:
        others = [minmax(lab[o].values) for o in score_cols if o != a]
        avg = np.mean(others, axis=0)
        avg_row[a] = float(spearmanr(lab[a].values, avg).statistic)
    out = {"correlation_with_gold_change_rate": corr,
           "agreement_vs_average_of_others": avg_row,
           "n_words": len(words), "models": list(score_cols)}
    with open(os.path.join(RES, "durel_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)

    pd.set_option("display.width", 200)
    m = list(score_cols)[-1]
    print("\n=== word-level change measures (model: %s) ===" % m)
    print(words[["word", "split", "gold_change_rate", f"Mean_e[{m}]", f"Mean_l[{m}]",
                 f"Mean_c[{m}]", f"DELTA_LATER[{m}]", f"DELTA_COMPARE[{m}]"]]
          .round(3).to_string(index=False))
    print("\n=== Spearman of DURel measures with human change rate ===")
    print(json.dumps(corr, indent=2))
    print("\n=== agreement matrix ===")
    print(mat.round(3).to_string())


if __name__ == "__main__":
    main()
