"""
Data preparation for the English DURel replication on TempoWiC.

Builds three groups of use pairs per target word, mirroring Schlechtweg et al. (2018):
  EARLIER  : both uses drawn from the earlier time period t1
  LATER    : both uses drawn from the later time period t2
  COMPARE  : one use from t1, one from t2  (== the original TempoWiC pairs)

TempoWiC only ships COMPARE pairs (with gold labels); EARLIER/LATER pairs are
re-sampled from the same pool of tweets, following the DURel sampling protocol.
"""
import json, random, os, argparse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results")

SPLIT_FILES = {
    "train": ("train.data.jl", "train.labels.tsv"),
    "validation": ("validation.data.jl", "validation.labels.tsv"),
    "test": ("test-codalab-10k.data.jl", "test.gold.tsv"),
}


def load_labels(path):
    labs = {}
    with open(os.path.join(DATA, path)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            i, l = line.split("\t")
            labs[i] = int(l)
    return labs


def load_split(split):
    dfile, lfile = SPLIT_FILES[split]
    labels = load_labels(lfile)
    rows = []
    with open(os.path.join(DATA, dfile)) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] not in labels:      # test file contains dummy instances
                continue
            r["label"] = labels[r["id"]]
            r["split"] = split
            rows.append(r)
    return rows


def use_key(t):
    """A 'use' is a tweet; identify it by text + target position."""
    return (t["text"], t["text_start"], t["text_end"])


def build_compare(rows):
    """TempoWiC instances = COMPARE group (one use from t1, one from t2)."""
    out = []
    for r in rows:
        out.append({
            "pair_id": r["id"],
            "word": r["word"],
            "split": r["split"],
            "group": "COMPARE",
            "gold": r["label"],
            "use1": r["tweet1"],
            "use2": r["tweet2"],
        })
    return out


def collect_uses(rows):
    """word -> period -> list of unique uses.  period 0 = earlier year, 1 = later."""
    uses = defaultdict(lambda: {0: {}, 1: {}})
    for r in rows:
        for t, p in ((r["tweet1"], 0), (r["tweet2"], 1)):
            uses[r["word"]][p][use_key(t)] = t
    return {w: {p: list(d.values()) for p, d in v.items()} for w, v in uses.items()}


def sample_within(uses, n_per_group=100, seed=42):
    """Sample EARLIER and LATER use pairs (DURel protocol: no use pair twice)."""
    rng = random.Random(seed)
    pairs = []
    for word, periods in sorted(uses.items()):
        for p, gname in ((0, "EARLIER"), (1, "LATER")):
            pool = periods[p]
            k = len(pool)
            all_pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
            rng.shuffle(all_pairs)
            chosen = all_pairs[:n_per_group]
            for n, (i, j) in enumerate(chosen):
                pairs.append({
                    "pair_id": f"{gname[0].lower()}{n}-{word}",
                    "word": word,
                    "split": "sampled",
                    "group": gname,
                    "gold": None,
                    "use1": pool[i],
                    "use2": pool[j],
                })
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_group", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    all_rows = []
    for split in SPLIT_FILES:
        all_rows += load_split(split)

    compare = build_compare(all_rows)
    uses = collect_uses(all_rows)
    within = sample_within(uses, args.n_per_group, args.seed)
    pairs = compare + within

    with open(os.path.join(OUT, "pairs.jsonl"), "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    # corpus statistics
    stats = {"n_instances": len(all_rows), "n_words": len(uses),
             "n_compare": len(compare), "n_earlier": sum(1 for p in within if p["group"] == "EARLIER"),
             "n_later": sum(1 for p in within if p["group"] == "LATER"),
             "per_split": {}, "per_word": {}}
    for split in SPLIT_FILES:
        rs = [r for r in all_rows if r["split"] == split]
        stats["per_split"][split] = {
            "n_pairs": len(rs),
            "n_words": len({r["word"] for r in rs}),
            "n_same_meaning": sum(r["label"] for r in rs),
            "n_diff_meaning": sum(1 - r["label"] for r in rs),
            "periods": sorted({r["tweet1"]["date"][:4] for r in rs} | {r["tweet2"]["date"][:4] for r in rs}),
        }
    for w, per in sorted(uses.items()):
        rs = [r for r in all_rows if r["word"] == w]
        stats["per_word"][w] = {
            "split": rs[0]["split"],
            "n_compare_pairs": len(rs),
            "n_uses_t1": len(per[0]), "n_uses_t2": len(per[1]),
            "t1": sorted({r["tweet1"]["date"] for r in rs}),
            "t2": sorted({r["tweet2"]["date"] for r in rs}),
            "prop_same_meaning": sum(r["label"] for r in rs) / len(rs),
        }
    with open(os.path.join(OUT, "data_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "per_word"}, indent=2))
    print("total pairs to score:", len(pairs))


if __name__ == "__main__":
    main()
