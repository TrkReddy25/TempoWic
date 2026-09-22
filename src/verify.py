"""
Independent verification of the headline numbers.

Re-computes the main quantities from the raw files with a second, deliberately
different implementation (no sklearn for F1, no pandas groupby for the means)
and checks them against what the pipeline wrote into results/.
"""
import json, os, glob
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
ok, fail = [], []


def check(name, a, b, tol=1e-6):
    good = abs(a - b) <= tol
    (ok if good else fail).append(f"{name}: pipeline={a:.6f} recomputed={b:.6f}")
    return good


def macro_f1(gold, pred):
    """Hand-rolled macro-F1."""
    fs = []
    for c in (0, 1):
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(fs))


pairs = [json.loads(l) for l in open(os.path.join(RES, "pairs.jsonl"))]
split = np.array([p["split"] for p in pairs])
group = np.array([p["group"] for p in pairs])
word = np.array([p["word"] for p in pairs])
gold = np.array([-1 if p["gold"] is None else p["gold"] for p in pairs])

# ---- 1. dataset construction -------------------------------------------------
st = json.load(open(os.path.join(RES, "data_stats.json")))
check("n_compare", st["n_compare"], float((group == "COMPARE").sum()), 0)
check("n_earlier", st["n_earlier"], float((group == "EARLIER").sum()), 0)
check("n_words", st["n_words"], float(len(set(word))), 0)
# no COMPARE pair may mix periods incorrectly / no pair may repeat within a group
seen = {}
dups = 0
for p in pairs:
    k = (p["group"], p["word"], tuple(sorted([p["use1"]["text"], p["use2"]["text"]])))
    dups += k in seen
    seen[k] = 1
(ok if dups == 0 else fail).append(f"duplicate use pairs within a group: {dups}")
bad_dates = sum(1 for p in pairs if p["group"] == "EARLIER"
                and p["use1"]["date"][:4] != p["use2"]["date"][:4])
(ok if bad_dates == 0 else fail).append(f"EARLIER pairs spanning two years: {bad_dates}")
bad_c = sum(1 for p in pairs if p["group"] == "COMPARE"
            and p["use1"]["date"][:4] == p["use2"]["date"][:4])
(ok if bad_c == 0 else fail).append(f"COMPARE pairs inside one year: {bad_c}")

# ---- 2. zero-shot metrics ----------------------------------------------------
zs = json.load(open(os.path.join(RES, "zeroshot_results.json")))
sc = np.load(os.path.join(RES, "pair_scores_zeroshot.npz"), allow_pickle=True)
te = split == "test"
for name in [zs["_best_layer"], "cos_L12", "jaccard"]:
    t = zs[name]["threshold"]
    pred = (sc[name][te] >= t).astype(int)
    check(f"zero-shot {name} test macro-F1", zs[name]["test"]["macro_f1"],
          macro_f1(gold[te], pred), 1e-9)

# ---- 3. fine-tuned metrics ---------------------------------------------------
for f in sorted(glob.glob(os.path.join(RES, "finetune_seed*.json"))):
    r = json.load(open(f))
    pr = np.load(f.replace("finetune_seed", "finetune_probs_seed").replace(".json", ".npz"),
                 allow_pickle=True)
    ids = [p["pair_id"] for p in pairs if p["split"] == "test"]
    assert list(pr["test_id"]) == ids
    pred = (pr["test_prob"] >= 0.5).astype(int)
    check(f"fine-tuned seed{r['seed']} test macro-F1", r["test"]["macro_f1"],
          macro_f1(gold[te], pred), 1e-9)

# ---- 4. DURel measures -------------------------------------------------------
import csv
words_rows = list(csv.DictReader(open(os.path.join(RES, "durel_measures.csv"))))
da = json.load(open(os.path.join(RES, "durel_analysis.json")))
# same protocol as make_tables.py / figures.py: best validation macro-F1
val_f1 = {k: zs[k]["validation"]["macro_f1"] for k in zs if k.startswith("cos_") or k == "jaccard"}
sp = os.path.join(RES, "simlr_results.json")
if os.path.exists(sp):
    val_f1["simlr"] = json.load(open(sp))["validation"]["macro_f1"]
ftj = sorted(glob.glob(os.path.join(RES, "finetune_seed*.json")))
if ftj:
    val_f1["finetuned"] = float(np.mean([json.load(open(f))["validation"]["macro_f1"] for f in ftj]))
m = max((k for k in val_f1 if k in da["models"]), key=lambda k: val_f1[k])
rel = {}
with open(os.path.join(RES, "pair_scores.csv")) as f:
    for r in csv.DictReader(f):
        rel.setdefault(r["word"], {}).setdefault(r["group"], []).append(float(r["rel_" + m]))
for row in words_rows[:5] + words_rows[-5:]:
    w_ = row["word"]
    me = float(np.mean(rel[w_]["EARLIER"])); ml = float(np.mean(rel[w_]["LATER"]))
    mc = float(np.mean(rel[w_]["COMPARE"]))
    check(f"{w_} DELTA_LATER[{m}]", float(row[f"DELTA_LATER[{m}]"]), ml - me, 1e-4)
    check(f"{w_} COMPARE[{m}]", float(row[f"COMPARE[{m}]"]), mc, 1e-4)

# gold change rate recomputed straight from pairs.jsonl
for row in words_rows[:5]:
    w_ = row["word"]
    g = np.array([p["gold"] for p in pairs if p["word"] == w_ and p["group"] == "COMPARE"])
    check(f"{w_} gold change rate", float(row["gold_change_rate"]), float(1 - g.mean()), 1e-9)

# correlation of COMPARE with the human change rate
x = np.array([float(r[f"COMPARE[{m}]"]) for r in words_rows])
y = np.array([float(r["gold_change_rate"]) for r in words_rows])
check(f"rho(COMPARE[{m}], change rate)",
      da["correlation_with_gold_change_rate"][m]["COMPARE"]["spearman"],
      float(spearmanr(x, y).statistic), 1e-9)

# ---- 5. numbers quoted in the report ----------------------------------------
nums_path = os.path.join(ROOT, "report", "tables", "numbers.tex")
if os.path.exists(nums_path):
    txt = open(nums_path).read()
    def macro(n):
        import re
        mm = re.search(r"\\newcommand\{\\" + n + r"\}\{(.*?)\}\n", txt, re.S)
        return mm.group(1).replace("{,}", "").replace(",", "") if mm else None
    check("report macro numTotalPairs", float(macro("numTotalPairs")), float(len(pairs)), 0)
    check("report macro zsF", float(macro("zsF")),
          round(100 * zs[zs["_best_layer"]]["test"]["macro_f1"], 1), 0.051)
    if macro("ftF"):
        fts = [json.load(open(p))["test"]["macro_f1"]
               for p in sorted(glob.glob(os.path.join(RES, "finetune_seed*.json")))]
        check("report macro ftF", float(macro("ftF")), round(100 * float(np.mean(fts)), 1), 0.051)
    check("report macro rhoCompare", float(macro("rhoCompare")),
          round(da["correlation_with_gold_change_rate"][m]["COMPARE"]["spearman"], 2), 0.005)

print("=" * 70)
for line in ok:
    print("PASS ", line)
for line in fail:
    print("FAIL ", line)
print("=" * 70)
print(f"{len(ok)} checks passed, {len(fail)} failed")
raise SystemExit(1 if fail else 0)
