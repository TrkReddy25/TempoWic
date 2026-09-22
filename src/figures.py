"""Figures for the TempoWiC-DURel report."""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#d1372e", "#8a8983"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#b9b8b2", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "figure.dpi": 200})


def main_model(words):
    """Same protocol as make_tables.py: the relatedness model with the best
    validation macro-F1 among those scored over all pairs."""
    import glob
    zs = json.load(open(os.path.join(RES, "zeroshot_results.json")))
    val = {k: zs[k]["validation"]["macro_f1"] for k in zs
           if k.startswith("cos_") or k == "jaccard"}
    sp = os.path.join(RES, "simlr_results.json")
    if os.path.exists(sp):
        val["simlr"] = json.load(open(sp))["validation"]["macro_f1"]
    ft = sorted(glob.glob(os.path.join(RES, "finetune_seed*.json")))
    if ft:
        val["finetuned"] = float(np.mean([json.load(open(f))["validation"]["macro_f1"] for f in ft]))
    avail = [k for k in val if f"DELTA_LATER[{k}]" in words.columns]
    return max(avail, key=lambda k: val[k])


def fig_delta_later(words, m):
    w = words.sort_values(f"DELTA_LATER[{m}]").reset_index(drop=True)
    v = w[f"DELTA_LATER[{m}]"].values
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 5.6), sharey=True,
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    colors = [BLUE if x < 0 else RED for x in v]
    ax.barh(range(len(w)), v, color=colors, height=0.62)
    ax.axvline(0, color="#57565230", lw=1)
    ax.set_yticks(range(len(w)), w["word"])
    ax.set_xlabel(r"$\Delta$LATER  (Mean$_l$ $-$ Mean$_e$)")
    ax.set_title("A  Direction of change", loc="left", fontsize=9)
    ax.text(0.02, 0.02, "innovative", color=BLUE, transform=ax.transAxes, fontsize=7.5)
    ax.text(0.98, 0.02, "reductive", color=RED, transform=ax.transAxes, fontsize=7.5, ha="right")
    ax.xaxis.grid(True); ax.set_axisbelow(True)

    ax2 = axes[1]
    ax2.barh(range(len(w)), w["gold_change_rate"].values, color=GRAY, height=0.62)
    ax2.set_xlabel("human change rate\n(share of 'different meaning' pairs)")
    ax2.set_title("B  Human judgements", loc="left", fontsize=9)
    ax2.set_xlim(0, 1); ax2.xaxis.grid(True); ax2.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_delta_later.pdf"))
    fig.savefig(os.path.join(FIG, "fig_delta_later.png"))
    plt.close(fig)


def fig_compare_scatter(words, m):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for ax, meas, col in ((axes[0], "COMPARE", BLUE), (axes[1], "DELTA_COMPARE", ORANGE)):
        x = words[f"{meas}[{m}]"].values
        y = words["gold_change_rate"].values
        r, p = spearmanr(x, y)
        ax.scatter(x, y, s=22, color=col, edgecolor="white", linewidth=0.6, zorder=3)
        z = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 20)
        ax.plot(xs, np.polyval(z, xs), color="#57565266", lw=1.2, zorder=2)
        for xi, yi, w_ in zip(x, y, words["word"]):
            ax.annotate(w_, (xi, yi), fontsize=5.4, color="#52514e",
                        xytext=(2.5, 2.5), textcoords="offset points")
        label = {"COMPARE": "COMPARE", "DELTA_COMPARE": r"$\Delta$COMPARE"}[meas]
        ax.set_xlabel(label + "  (" + m.replace("cos_L", "cos L").replace("_", " ") + ")")
        ax.set_ylabel("human change rate")
        ax.set_title(rf"$\rho$ = {r:.2f}   (p = {p:.3g})", loc="left", fontsize=8.5)
        ax.grid(True); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_compare_scatter.pdf"))
    fig.savefig(os.path.join(FIG, "fig_compare_scatter.png"))
    plt.close(fig)


def fig_group_distributions(df, words, m):
    """DURel Fig. 4 analogue: judgement distributions per group for two words."""
    w_change = words.sort_values("gold_change_rate").iloc[-1]["word"]
    w_stable = words.sort_values("gold_change_rate").iloc[0]["word"]
    bins = np.array([1, 1.75, 2.5, 3.25, 4.0])
    labels = ["1\nUnrelated", "2\nDistantly", "3\nClosely", "4\nIdentical"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, word in zip(axes, [w_stable, w_change]):
        sub = df[df.word == word]
        width = 0.26
        for k, (grp, col) in enumerate([("EARLIER", BLUE), ("LATER", ORANGE), ("COMPARE", AQUA)]):
            v = sub.loc[sub.group == grp, "rel_" + m].values
            h, _ = np.histogram(v, bins=bins)
            h = h / max(h.sum(), 1)
            ax.bar(np.arange(4) + (k - 1) * width, h, width=width - 0.03, color=col, label=grp)
        ax.set_xticks(range(4), labels, fontsize=7)
        ax.set_title(f"{word}  (human change rate "
                     f"{float(words.loc[words.word == word, 'gold_change_rate'].iloc[0]):.2f})",
                     loc="left", fontsize=9)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    axes[0].set_ylabel("share of use pairs")
    axes[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_group_distributions.pdf"))
    fig.savefig(os.path.join(FIG, "fig_group_distributions.png"))
    plt.close(fig)


def fig_layers(zs):
    layers = [k for k in zs if k.startswith("cos_L")]
    idx = sorted(int(k[5:]) for k in layers)
    val = [zs[f"cos_L{i}"]["validation"]["macro_f1"] for i in idx]
    test = [zs[f"cos_L{i}"]["test"]["macro_f1"] for i in idx]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(idx, val, color=BLUE, lw=2, marker="o", ms=3.5, label="validation")
    ax.plot(idx, test, color=ORANGE, lw=2, marker="o", ms=3.5, label="test")
    ax.axhline(0.5, color=GRAY, lw=1, ls=":", label="random")
    ax.set_xlabel("encoder layer"); ax.set_ylabel("macro-F1")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_layers.pdf"))
    fig.savefig(os.path.join(FIG, "fig_layers.png"))
    plt.close(fig)


def fig_groups_schematic():
    """DURel Figure 1 analogue: the three groups of use pairs."""
    from matplotlib.patches import FancyBboxPatch
    rng = np.random.RandomState(7)
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")

    for x0, label, note in ((0.4, "$t_1$: EARLIER period", "2019 / 2020"),
                            (5.6, "$t_2$: LATER period", "2020 / 2021")):
        ax.add_patch(FancyBboxPatch((x0, 1.05), 4.0, 2.5, boxstyle="round,pad=0.06",
                                    linewidth=1, edgecolor="#b9b8b2", facecolor="#f6f5f2"))
        ax.text(x0 + 2.0, 3.16, label, ha="center", fontsize=8.5)
        ax.text(x0 + 2.0, 2.86, note, ha="center", fontsize=7.5, color="#52514e")

    pts_e = np.column_stack([rng.uniform(0.9, 4.0, 7), rng.uniform(1.35, 2.55, 7)])
    pts_l = np.column_stack([rng.uniform(6.1, 9.2, 7), rng.uniform(1.35, 2.55, 7)])
    for pts in (pts_e, pts_l):
        ax.scatter(pts[:, 0], pts[:, 1], s=32, color="#575652", zorder=3,
                   edgecolor="white", linewidth=0.8)
    for a, b in [(0, 1), (2, 4)]:
        ax.plot(*zip(pts_e[a], pts_e[b]), color=BLUE, lw=1.8, zorder=2)
        ax.plot(*zip(pts_l[a], pts_l[b]), color=ORANGE, lw=1.8, zorder=2)
    for a, b in [(3, 2), (5, 6)]:
        ax.plot([pts_e[a, 0], pts_l[b, 0]], [pts_e[a, 1], pts_l[b, 1]],
                color=AQUA, lw=1.8, zorder=2)

    ax.text(2.4, 0.55, "EARLIER pairs", color=BLUE, ha="center", fontsize=8.5)
    ax.text(7.6, 0.55, "LATER pairs", color=ORANGE, ha="center", fontsize=8.5)
    ax.text(5.0, 0.20, "COMPARE pairs", color=AQUA, ha="center", fontsize=8.5)
    ax.text(5.0, 3.95, "one dot = one use of the target word; one line = one rated use pair",
            ha="center", fontsize=7.5, color="#52514e")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_groups_schematic.pdf"))
    fig.savefig(os.path.join(FIG, "fig_groups_schematic.png"))
    plt.close(fig)


if __name__ == "__main__":
    words = pd.read_csv(os.path.join(RES, "durel_measures.csv"))
    df = pd.read_csv(os.path.join(RES, "pair_scores.csv"))
    zs = json.load(open(os.path.join(RES, "zeroshot_results.json")))
    m = main_model(words)
    fig_delta_later(words, m)
    fig_compare_scatter(words, m)
    fig_group_distributions(df, words, m)
    fig_layers(zs)
    fig_groups_schematic()
    print("figures written for model:", m)
