"""Plot how the metrics change with the number of candidates N (best-of-N).

python plot_results.py --exp a   (after make_tables.py --exp a)
python plot_results.py --exp b
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order
NAMES = {"bas": "select by BAS", "clap": "select by CLAP", "clap+bas": "select by CLAP + BAS"}


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_ylabel(ylabel, fontsize=8, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.set_xlabel("candidates N", fontsize=8, color=INK2)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)


def panel(ax, scal, col, refs):
    for i, (sel, g) in enumerate(scal.groupby("selector", sort=False)):
        ax.plot(g.k, g[col], color=SERIES[i], lw=2, marker="o", ms=5, label=NAMES[sel])
    for label, (val, color) in refs.items():
        ax.axhline(val, color=color, lw=1.2, ls="--")
        ax.text(1, val, label, color=INK2, fontsize=7, va="bottom", ha="left")


def plot_a():
    sa = pd.read_csv("results/exp_a_scaling.csv")
    ta = pd.read_csv("results/exp_a_table.csv").set_index("method")
    real = [m for m in ta.index if m.startswith("real")][0]
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), facecolor=SURFACE)
    for ax, (col, title) in zip(axes, [("bas_beat", "BAS, librosa beats (selection score)"),
                                       ("bas_mm", "BAS, madmom beats (held out)"),
                                       ("dbas_mm", "BAS minus mismatched-music BAS"),
                                       ("style_acc", "dance-style accuracy (held out)")]):
        refs = {"real dancers": (ta.loc[real, col], INK2),
                "1 sample + retime": (ta.loc["+ retime (1 sample)", col], SERIES[1])}
        panel(ax, sa, col, refs)
        style(ax, title, "")
        if col == "style_acc":
            ax.set_ylim(0.5, 1.0)  # accuracy on its natural scale; the wiggles are not significant
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig("results/exp_a_scaling.png", dpi=130, facecolor=SURFACE)


def plot_b():
    sb = pd.read_csv("results/exp_b_scaling.csv")
    panels = [("clap_sel", "CLAP-fused (selection score)", None),
              ("clap", "CLAP, report's checkpoint (held out)", (0.40, 0.60)),
              ("gtzan_acc", "GTZAN genre accuracy (held out)", (0.0, 1.0)),
              ("bas_mm", "BAS, madmom beats (held out)", None),
              ("dbas_mm", "BAS minus mismatched-music BAS", None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 2.8), facecolor=SURFACE)
    for ax, (col, title, ylim) in zip(axes, panels):
        panel(ax, sb, col, {})
        style(ax, title, "")
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig("results/exp_b_scaling.png", dpi=130, facecolor=SURFACE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["a", "b"], required=True)
    plot_a() if ap.parse_args().exp == "a" else plot_b()
