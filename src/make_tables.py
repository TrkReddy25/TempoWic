"""Generate all LaTeX tables in report/tables/ directly from the result files."""
import json, os, glob
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
TAB = os.path.join(ROOT, "report", "tables")
os.makedirs(TAB, exist_ok=True)


def num(x):
    """Consistent thousands separator for every number printed in a table."""
    return f"{int(x):,}".replace(",", "{,}")


def sn(x, dec=2, signed=False):
    """Number for a table cell: real math minus, padded so columns line up."""
    v = float(x)
    body = f"{abs(v):.{dec}f}"
    if v < 0:
        return "$-$" + body
    return ("$+$" if signed else "\\phantom{$-$}") + body


def w(name, s):
    with open(os.path.join(TAB, name), "w") as f:
        f.write(s)
    print("wrote", name)


# ---------------------------------------------------------------- data stats
st = json.load(open(os.path.join(RES, "data_stats.json")))
rows = []
for sp in ["train", "validation", "test"]:
    d = st["per_split"][sp]
    rows.append(f"{sp.capitalize()} & {d['n_words']} & {num(d['n_pairs'])} & {num(d['n_same_meaning'])} & "
                f"{num(d['n_diff_meaning'])} & {d['periods'][0]}/{d['periods'][1]} \\\\")
tot = st["per_split"]
w("tab_data.tex", r"""\begin{tabular}{lrrrrc}
\toprule
Split & Words & COMPARE pairs & same & different & $t_1$/$t_2$ \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
All & %d & %s & %s & %s & -- \\
\bottomrule
\end{tabular}""" % (st["n_words"], num(st["n_compare"]),
                    num(sum(tot[s]["n_same_meaning"] for s in tot)),
                    num(sum(tot[s]["n_diff_meaning"] for s in tot))))

# ---------------------------------------------------------- pair inventory
w("tab_groups.tex", r"""\begin{tabular}{lrl}
\toprule
Group & Use pairs & Source \\
\midrule
EARLIER ($t_1$,$t_1$) & %s & sampled (100 per word) \\
LATER ($t_2$,$t_2$) & %s & sampled (100 per word) \\
COMPARE ($t_1$,$t_2$) & %s & original TempoWiC pairs (gold-labelled) \\
\midrule
Total & %s & \\
\bottomrule
\end{tabular}""" % (num(st["n_earlier"]), num(st["n_later"]), num(st["n_compare"]),
                    num(st["n_earlier"] + st["n_later"] + st["n_compare"])))

# ------------------------------------------------------------- main results
zs = json.load(open(os.path.join(RES, "zeroshot_results.json")))
best_layer = zs["_best_layer"]
lines = []


def fmt(r, key="test"):
    d = r[key]
    return (f"{100*d['macro_f1']:.1f} & {100*d['f1_same']:.1f} & "
            f"{100*d['f1_diff']:.1f} & {100*d['accuracy']:.1f}")


lines.append(r"\multicolumn{5}{l}{\textit{Controls}} \\")
lines.append(f"Random & {fmt(zs['random'])} \\\\")
lines.append(f"Majority class & {fmt(zs['majority'])} \\\\")
lines.append(f"Lexical overlap (Jaccard) & {fmt(zs['jaccard'])} \\\\")
lines.append(r"\midrule")
lines.append(r"\multicolumn{5}{l}{\textit{Zero-shot contextual similarity}} \\")
lines.append(f"Sentence vectors (mean-pooled) & {fmt(zs['cos_sent'])} \\\\")
lines.append(f"Target vectors, last layer (L12) & {fmt(zs['cos_L12'])} \\\\")
lines.append(f"Target vectors, layer selected on dev ({best_layer[4:]}) & {fmt(zs[best_layer])} \\\\")
best_test_layer = max((k for k in zs if k.startswith("cos_L")),
                      key=lambda k: zs[k]["test"]["macro_f1"])
lines.append(f"\\quad {{\\small (oracle layer {best_test_layer[4:]})}} & {fmt(zs[best_test_layer])} \\\\")

simlr_path = os.path.join(RES, "simlr_results.json")
simlr = json.load(open(simlr_path)) if os.path.exists(simlr_path) else None
if simlr:
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{l}{\textit{Supervised}} \\")
    lines.append(f"Logistic regression over similarities & {fmt(simlr)} \\\\")

ft_files = sorted(glob.glob(os.path.join(RES, "finetune_seed*.json")))
if ft_files:
    fts = [json.load(open(p)) for p in ft_files]
    arr = lambda k, s: np.array([f[s][k] for f in fts])
    if not simlr:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{5}{l}{\textit{Supervised}} \\")
    for f in fts:
        lines.append(f"Fine-tuned cross-encoder (seed {f['seed']}) & {fmt(f)} \\\\")
    lines.append("Fine-tuned cross-encoder (mean$\\pm$sd) & " + " & ".join(
        f"{100*arr(k,'test').mean():.1f}\\small{{$\\pm${100*arr(k,'test').std():.1f}}}"
        for k in ["macro_f1", "f1_same", "f1_diff", "accuracy"]) + r" \\")

w("tab_results.tex", r"""\begin{tabular}{lrrrr}
\toprule
System & macro-F1 & F1$_{same}$ & F1$_{diff}$ & Acc. \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")

# ----------------------------------------------------------------- agreement
mat = pd.read_csv(os.path.join(RES, "agreement_matrix.csv"), index_col=0)
an = list(mat.columns)
pretty = {"cos_sent": "sent.\\ cos", "jaccard": "Jaccard", "finetuned": "fine-tuned CE",
          "simlr": "sim.\\ log.\\ reg.", "gold": "\\textsc{gold}"}
pretty.update({f"cos_L{i}": f"cos L{i}" for i in range(13)})
hdr = " & ".join(pretty.get(a, a) for a in an)
body = []
for i, a in enumerate(an):
    cells = []
    for j, b in enumerate(an):
        cells.append("--" if i == j else ("" if j < i else sn(mat.loc[a, b])))
    body.append(pretty.get(a, a) + " & " + " & ".join(cells) + r" \\")
da = json.load(open(os.path.join(RES, "durel_analysis.json")))
avg = da["agreement_vs_average_of_others"]
body.append(r"\midrule")
body.append("vs.\\ avg.\\ of others & " + " & ".join(
    sn(avg[a]) if a in avg else "--" for a in an) + r" \\")
w("tab_agreement.tex", r"""\begin{tabular}{l""" + "r" * len(an) + r"""}
\toprule
 & """ + hdr + r""" \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}""")

# ------------------------------------------------- DURel measure validation
corr = da["correlation_with_gold_change_rate"]
order = [m for m in [best_layer, "cos_L8", "cos_L12", "simlr", "finetuned", "cos_sent", "jaccard"] if m in corr]
seen = set()
order = [m for m in order if not (m in seen or seen.add(m))]
rows = []
for m in order:
    c = corr[m]
    cells = []
    for meas in ["COMPARE", "DELTA_COMPARE", "DELTA_LATER"]:
        r, p = c[meas]["spearman"], c[meas]["p"]
        if p < 0.01:
            star = "$^{**}$"
        elif p < 0.05:
            star = "$^{*}$\\phantom{$^{*}$}"
        else:
            star = "\\phantom{$^{**}$}"
        cells.append(sn(r) + star)
    rows.append(pretty.get(m, m) + " & " + " & ".join(cells) + r" \\")
w("tab_measure_validation.tex", r"""\begin{tabular}{lrrr}
\toprule
Relatedness model & COMPARE & $\Delta$COMPARE & $\Delta$LATER \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")

# ------------------------------------------------------ per-word measures
words = pd.read_csv(os.path.join(RES, "durel_measures.csv"))
m = best_layer  # protocol: model with best validation macro-F1 (see numbers.tex)
wd = words.sort_values(f"COMPARE[{m}]")


def word_rows(frame):
    return [f"{r['word']} & {r['split'][:2]} & {r['gold_change_rate']:.2f} & "
            f"{r[f'Mean_e[{m}]']:.2f} & {r[f'Mean_l[{m}]']:.2f} & {r[f'Mean_c[{m}]']:.2f} & "
            f"{sn(r[f'DELTA_LATER[{m}]'], signed=True)} & "
            f"{sn(r[f'DELTA_COMPARE[{m}]'], signed=True)} \\\\"
            for _, r in frame.iterrows()]


HEAD = (r"Word & Sp. & chg. & Mean$_e$ & Mean$_l$ & Mean$_c$ & $\Delta$L & $\Delta$C \\")
half = (len(wd) + 1) // 2


def block(frame):
    return (r"\begin{tabular}{@{}llrrrrrr@{}}" + "\n\\toprule\n" + HEAD +
            "\n\\midrule\n" + "\n".join(word_rows(frame)) +
            "\n\\bottomrule\n\\end{tabular}")


w("tab_words.tex",
  r"\begin{minipage}[t]{0.49\linewidth}\centering" + "\n" + block(wd.iloc[:half]) +
  "\n\\end{minipage}\\hfill\n" +
  r"\begin{minipage}[t]{0.49\linewidth}\centering" + "\n" + block(wd.iloc[half:]) +
  "\n\\end{minipage}")
print("model used for word table:", m)

# ------------------------------------------- key numbers cited in the prose
def mac(name, val):
    return "\\newcommand{\\%s}{%s}\n" % (name, val)

nums = ""
nums += mac("numWords", st["n_words"])
nums += mac("numCompare", f"{st['n_compare']:,}".replace(",", "{,}"))
nums += mac("numEarlier", num(st["n_earlier"]))
nums += mac("numLater", num(st["n_later"]))
nums += mac("numTotalPairs", f"{st['n_earlier']+st['n_later']+st['n_compare']:,}".replace(",", "{,}"))
nums += mac("numUses", num(6594))
nums += mac("bestLayer", best_layer[5:])
nums += mac("zsF", f"{100*zs[best_layer]['test']['macro_f1']:.1f}")
nums += mac("zsOracleF", f"{100*zs[best_test_layer]['test']['macro_f1']:.1f}")
nums += mac("zsOracleLayer", best_test_layer[5:])
nums += mac("lastLayerF", f"{100*zs['cos_L12']['test']['macro_f1']:.1f}")
nums += mac("jaccardF", f"{100*zs['jaccard']['test']['macro_f1']:.1f}")
nums += mac("randomF", f"{100*zs['random']['test']['macro_f1']:.1f}")
if ft_files:
    nums += mac("ftF", f"{100*arr('macro_f1','test').mean():.1f}")
    nums += mac("ftFsd", f"{100*arr('macro_f1','test').std():.1f}")
    nums += mac("ftVal", f"{100*arr('macro_f1','validation').mean():.1f}")
    nums += mac("numSeeds", len(fts))
val_f1 = {k: zs[k]["validation"]["macro_f1"] for k in zs if k.startswith("cos_") or k == "jaccard"}
if os.path.exists(os.path.join(RES, "simlr_results.json")):
    val_f1["simlr"] = json.load(open(os.path.join(RES, "simlr_results.json")))["validation"]["macro_f1"]
if ft_files:
    val_f1["finetuned"] = float(np.mean([f["validation"]["macro_f1"] for f in fts]))
mm = max((k for k in val_f1 if k in corr), key=lambda k: val_f1[k])
if simlr:
    nums += mac("simlrF", f"{100*simlr['test']['macro_f1']:.1f}")
nums += mac("mainModelVal", f"{100*val_f1[mm]:.1f}")
nums += mac("mainModel", pretty.get(mm, mm))
for meas, key in (("COMPARE", "rhoCompare"), ("DELTA_COMPARE", "rhoDeltaCompare"),
                  ("DELTA_LATER", "rhoDeltaLater")):
    nums += mac(key, f"{corr[mm][meas]['spearman']:.2f}")
    nums += mac(key + "P", f"{corr[mm][meas]['p']:.4f}")
nums += mac("goldAgree", f"{mat.loc[mm, 'gold']:.2f}" if mm in mat.index else "--")
mpos = int((words[f"DELTA_LATER[{mm}]"] > 0).sum())
nums += mac("nReductive", mpos)
nums += mac("nInnovative", len(words) - mpos)
w("numbers.tex", nums)

# ------------------------------------------ alignment with the original study
n_uses_total = 6594
align_rows = [
    ("Corpus", "DTA, German historical prose", "TempoWiC, English tweets"),
    ("Time periods", "1750--1800 vs.\\ 1850--1900", "year $Y$ vs.\\ $Y{+}1$ (2019--2021)"),
    ("Definition of a use", "sentence, with neighbouring sentences", "one tweet"),
    ("Target words", "22", f"{st['n_words']}"),
    ("Use pairs", "1{,}320", num(st['n_earlier']+st['n_later']+st['n_compare'])),
    ("Pairs per word and group", "20", "100 (EARLIER / LATER)"),
    ("Groups", "EARLIER, LATER, COMPARE", "identical"),
    ("Relatedness scale", "1--4, 0 = cannot decide", "1--4, mapped from model scores"),
    ("Raters$^\\dagger$", "5 human annotators", "5 relatedness models"),
    ("Within-period pairs$^\\dagger$", "sampled from the corpus",
     "sampled from TempoWiC's tweets"),
    ("Agreement analysis", "pairwise Spearman between raters", "identical, plus vs.\\ gold labels"),
    ("Change measures", "$\\Delta$LATER, COMPARE, $\\Delta$COMPARE", "identical"),
    ("Validation of measures$^\\dagger$", "qualitative, historical dictionaries",
     "numerical, vs.\\ human change rate"),
]
w("tab_alignment.tex", r"""\begin{tabular}{@{}lll@{}}
\toprule
Element & \citet{schlechtweg2018durel} & This report \\
\midrule
""" + "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in align_rows) + r"""
\bottomrule
\end{tabular}""")

# ------------------------------------------------- per-word error analysis
import numpy as _np
ps = pd.read_csv(os.path.join(RES, "pair_scores.csv"))
te_rows = ps[(ps.split == "test") & (ps.group == "COMPARE")].copy()
thr = zs[best_layer]["threshold"]
te_rows["pred_zs"] = (te_rows[best_layer] >= thr).astype(int)
have_ft = "finetuned" in te_rows.columns
if have_ft:
    te_rows["pred_ft"] = (te_rows["finetuned"] >= 0.5).astype(int)
er = []
for wd_, g in te_rows.groupby("word"):
    row = {"word": wd_, "n": len(g),
           "change_rate": float(1 - g["gold"].mean()),
           "acc_zs": float((g["pred_zs"] == g["gold"]).mean())}
    if have_ft:
        row["acc_ft"] = float((g["pred_ft"] == g["gold"]).mean())
    row["majority"] = float(max(g["gold"].mean(), 1 - g["gold"].mean()))
    er.append(row)
er = pd.DataFrame(er).sort_values("acc_zs", ascending=False)
er.to_csv(os.path.join(RES, "per_word_test_accuracy.csv"), index=False)


def _acc_rows(frame):
    out = []
    for _, r in frame.iterrows():
        cells = [r["word"], f"{r['change_rate']:.2f}", f"{100*r['majority']:.1f}",
                 f"{100*r['acc_zs']:.1f}"]
        if have_ft:
            cells.append(f"{100*r['acc_ft']:.1f}")
        out.append(" & ".join(cells) + r" \\")
    return out


_hdr = (r"Word & chg. & maj. & cos L%s & FT CE \\" % best_layer[5:]) if have_ft \
    else (r"Word & chg. & maj. & cos L%s \\" % best_layer[5:])
_cols = "lrrrr" if have_ft else "lrrr"
_half = (len(er) + 1) // 2


def _acc_block(frame):
    return (r"\begin{tabular}{@{}%s@{}}" % _cols + "\n\\toprule\n" + _hdr +
            "\n\\midrule\n" + "\n".join(_acc_rows(frame)) +
            "\n\\bottomrule\n\\end{tabular}")


w("tab_worderror.tex",
  r"\begin{minipage}[t]{0.49\linewidth}\centering" + "\n" + _acc_block(er.iloc[:_half]) +
  "\n\\end{minipage}\\hfill\n" +
  r"\begin{minipage}[t]{0.49\linewidth}\centering" + "\n" + _acc_block(er.iloc[_half:]) +
  "\n\\end{minipage}")

from scipy.stats import spearmanr as _sp
r_bal, p_bal = _sp(er["majority"], er["acc_zs"])
nums_extra = ""
nums_extra += mac("hardestWord", er.iloc[-1]["word"])
nums_extra += mac("hardestAcc", f"{100*er.iloc[-1]['acc_zs']:.1f}")
nums_extra += mac("easiestWord", er.iloc[0]["word"])
nums_extra += mac("easiestAcc", f"{100*er.iloc[0]['acc_zs']:.1f}")
nums_extra += mac("nBelowMajority", int((er["acc_zs"] < er["majority"]).sum()))
nums_extra += mac("nTestWords", len(er))
nums_extra += mac("rhoBalanceAcc", f"{r_bal:.2f}")
nums_extra += mac("rhoBalanceAccP", f"{p_bal:.3f}")
with open(os.path.join(TAB, "numbers.tex"), "a") as f:
    f.write(nums_extra)
print("per-word error analysis written")

# ------------------------------------------------ background-section tables
def tex_escape(t):
    for a, b in [("\\", "\\textbackslash "), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"),
                 ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"),
                 ("~", "\\textasciitilde "), ("^", "\\textasciicircum ")]:
        t = t.replace(a, b)
    return t


w("tab_scale.tex", r"""\begin{tabular}{@{}cll@{}}
\toprule
Score & Label & Reading \\
\midrule
4 & Identical & the two uses carry the same meaning \\
3 & Closely related & the same meaning, used in different contexts \\
2 & Distantly related & related meanings, e.g.\ literal vs.\ figurative \\
1 & Unrelated & homonymy: no shared meaning \\
0 & Cannot decide & rater abstains (excluded from the means) \\
\bottomrule
\end{tabular}""")

w("tab_resources.tex", r"""\begin{tabular}{@{}lllll@{}}
\toprule
Resource & Languages & Genre & Unit judged & Size \\
\midrule
DURel \citep{schlechtweg2018durel} & German & historical prose & use pair, 1--4 & 1{,}320 pairs \\
DWUG \citep{schlechtweg2021dwug} & 4 languages & mixed historical & use pair, 1--4 & $\sim$100k judgements \\
SemEval-2020 T1 \citep{schlechtweg2020semeval} & 4 languages & historical corpora & word, binary + rank & 137 words \\
WiC \citep{pilehvar2019wic} & English & dictionary examples & use pair, binary & 7{,}466 pairs \\
TempoWiC \citep{loureiro2022tempowic} & English & tweets & use pair, binary & 3{,}297 pairs \\
\midrule
This report & English & tweets & use pair, 1--4 (model) & %s pairs \\
\bottomrule
\end{tabular}""" % num(st["n_earlier"] + st["n_later"] + st["n_compare"]))

# a real, ASCII-only example instance from the training split
ex_rows = [json.loads(l) for l in open(os.path.join(RES, "pairs.jsonl"))]
ex_rows = [p for p in ex_rows if p["group"] == "COMPARE" and p["split"] == "train"]


BAD = ("&amp;", "&lt;", "&gt;", "http", "@", "#")


def ascii_ok(p, limit=100):
    return all(t["text"].isascii() and len(t["text"]) <= limit
               and not any(b in t["text"] for b in BAD)
               for t in (p["use1"], p["use2"]))


ex_diff = next(p for p in ex_rows if p["gold"] == 0 and ascii_ok(p))
ex_same = next(p for p in ex_rows if p["gold"] == 1 and ascii_ok(p) and p["word"] == ex_diff["word"])
ex_lines = []
for p, tag in ((ex_same, "same meaning"), (ex_diff, "different meaning")):
    for i, u in enumerate((p["use1"], p["use2"]), start=1):
        txt = tex_escape(u["text"])
        tgt = tex_escape(u["text"][u["text_start"]:u["text_end"]])
        txt = txt.replace(tgt, "\\textbf{" + tgt + "}", 1)
        ex_lines.append(f"{u['date']} & {txt} \\\\")
    ex_lines.append(f"\\multicolumn{{2}}{{r}}{{\\textit{{gold label: {tag}}}}} \\\\")
    ex_lines.append(r"\midrule" if tag == "same meaning" else "")
w("tab_example.tex", r"""\begin{tabular}{@{}l p{0.80\linewidth}@{}}
\toprule
\multicolumn{2}{@{}l}{Target word: \textit{%s}} \\
\midrule
""" % ex_diff["word"] + "\n".join(l for l in ex_lines if l) + r"""
\bottomrule
\end{tabular}""")

# hyper-parameters actually used
ft_args = fts[0]["args"] if ft_files else {}
w("tab_hyper.tex", r"""\begin{tabular}{@{}ll@{}}
\toprule
Setting & Value \\
\midrule
Encoder & RoBERTa-base (12 layers, 768 dim, 125M parameters) \\
Maximum sequence length & %s word pieces \\
Optimiser & AdamW, weight decay 0.01 \\
Learning rate & $2\times10^{-5}$, linear decay, 10\%% warm-up \\
Batch size & %s \\
Epochs & %s (best validation epoch kept) \\
Seeds & %s \\
Hardware & 2 CPU cores, no GPU \\
Training time & $\approx$40 min per seed \\
Threshold (unsupervised scores) & tuned on validation for macro-F1 \\
Logistic regression & standardised features, $L_2$, $C=1$ \\
\bottomrule
\end{tabular}""" % (ft_args.get("max_len", 128), ft_args.get("batch_size", 16),
                    ft_args.get("epochs", 4), len(ft_files) if ft_files else 1))

# inventory of uses per word and period
import statistics as _stats
pw = st["per_word"]
t1 = [v["n_uses_t1"] for v in pw.values()]
t2 = [v["n_uses_t2"] for v in pw.values()]
cp = [v["n_compare_pairs"] for v in pw.values()]
w("tab_useinventory.tex", r"""\begin{tabular}{@{}lrrr@{}}
\toprule
Per target word & min & median & max \\
\midrule
Unique uses in $t_1$ & %d & %.0f & %d \\
Unique uses in $t_2$ & %d & %.0f & %d \\
COMPARE pairs (gold) & %d & %.0f & %d \\
EARLIER / LATER pairs sampled & 100 & 100 & 100 \\
\bottomrule
\end{tabular}""" % (min(t1), _stats.median(t1), max(t1),
                    min(t2), _stats.median(t2), max(t2),
                    min(cp), _stats.median(cp), max(cp)))
print("background tables written")
