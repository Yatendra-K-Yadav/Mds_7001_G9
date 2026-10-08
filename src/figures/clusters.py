"""
Figure 3 - three types of gambling council.

Plots the two mechanism axes: how many machines are available, and how hard
each one is worked. Bubble size shows adult population, which is what separates
the metropolitan group from the rest.

    python src/figure_clusters.py
"""

import pandas as pd

from src import theme
from src.config import FIGURES, PROCESSED
from src.theme import CATEGORICAL, INK_SOFT, MUTED

import matplotlib.pyplot as plt

OUTPUT = "clusters.png"

# Cluster ids come from src/clustering.py; these describe what each one is.
LABELS = {0: "Metropolitan", 1: "Regional centres", 2: "Low exposure"}
DESCRIPTIONS = {0: "few machines, worked hard",
                1: "many machines, lower intensity",
                2: "few machines, low intensity"}
LABEL_POSITIONS = {0: (6.4, 121_000), 1: (15.3, 118_000), 2: (6.4, 66_500)}
ANNOTATED = [("BRISBANE", -6, -26), ("GOLD COAST", 0, -24),
             ("BUNDABERG", 0, 18), ("NOOSA", 14, -14)]


def main():
    theme.apply_defaults()
    data = pd.read_csv(PROCESSED / "clustered_councils.csv")
    colours = dict(zip(sorted(data.cluster.unique()), CATEGORICAL))

    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.77, bottom=0.175)
    theme.style_axes(ax)

    sizes = 40 + (data.adults / data.adults.max()) * 720
    for cluster in sorted(data.cluster.unique()):
        rows = data.cluster == cluster
        ax.scatter(data.loc[rows, "egms_per_1000_adults"],
                   data.loc[rows, "loss_per_egm"], s=sizes[rows],
                   c=colours[cluster], edgecolors=theme.SURFACE, linewidths=2,
                   alpha=0.88, zorder=3)

    # Direct labels rather than a legend box: the aqua slot sits below 3:1
    # contrast on this surface, so visible labels are required.
    for cluster, (x, y) in LABEL_POSITIONS.items():
        ax.text(x, y, LABELS[cluster], fontsize=12, color=colours[cluster],
                fontweight="bold", va="center")
        ax.text(x, y - 4600, DESCRIPTIONS[cluster], fontsize=8.8,
                color=MUTED, va="center")
        ax.text(x, y - 8600, f"n = {(data.cluster == cluster).sum()} councils",
                fontsize=8.4, color=MUTED, va="center")

    ax.set_xlim(4.6, 21.2)
    ax.set_ylim(50_000, 133_000)
    for name, dx, dy in ANNOTATED:
        row = data[data.lga_raw == name]
        if len(row):
            ax.annotate(name.title(),
                        (row.egms_per_1000_adults.iloc[0], row.loss_per_egm.iloc[0]),
                        textcoords="offset points", xytext=(dx, dy), ha="center",
                        fontsize=8.4, color=INK_SOFT, zorder=6)

    theme.dollars(ax, thousands=True)
    ax.set_xlabel("Poker machines per 1,000 adults", fontsize=10.5,
                  color=INK_SOFT, labelpad=8)
    ax.set_ylabel("Lost per machine per year (2025 dollars)", fontsize=10.5,
                  color=INK_SOFT, labelpad=8)

    theme.title_block(
        fig, "Three kinds of gambling council - and none of them is about wealth",
        f"k-means clusters of {len(data)} Queensland councils, averaged 2021-2025.  "
        "Bubble size = adult population.",
        note="Clusters are statistically independent of socio-economic disadvantage "
             "(chi-square p = 0.94)", x=0.095)
    theme.source_note(fig, [
        "Source: QLD Office of Liquor and Gaming Regulation; ABS Regional Population, "
        "SEIFA 2021, CPI (Brisbane).",
        "k = 3 selected by silhouette score (optimal in 99 of 100 restarts); "
        "features standardised.  Mount Isa excluded."], x=0.095, y=0.022)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / OUTPUT, dpi=220, facecolor=theme.SURFACE)
    print(f"Wrote {FIGURES / OUTPUT}")


if __name__ == "__main__":
    main()
