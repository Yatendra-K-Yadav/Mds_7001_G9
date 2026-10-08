"""
Shared chart styling.

Colours come from a validated palette: the categorical slots and the ordinal
blue ramp were both checked with a colour-blindness simulator (worst-pair
deltaE 9.2 deutan, 24.0 normal vision) against the light chart surface.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ------------------------------------------------------------------ colours
SURFACE = "#fcfcfb"       # chart background
INK = "#0b0b0b"           # headings
INK_SOFT = "#52514e"      # axis titles, secondary text
MUTED = "#898781"         # captions, tick labels
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"          # categorical slot 1
ORANGE = "#eb6834"        # categorical slot 2
AQUA = "#1baf7a"          # categorical slot 3
CATEGORICAL = [BLUE, ORANGE, AQUA]

# Ordinal ramp, light to dark. The light end still clears 2:1 on the surface.
ORDINAL_3 = ["#86b6ef", "#2a78d6", "#104281"]


def apply_defaults():
    """Set matplotlib rcParams once per figure script."""
    plt.rcParams.update({
        "axes.formatter.use_mathtext": False,   # stops '$' being read as LaTeX
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })


def style_axes(ax):
    """Recessive grid, no top/right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=9.5, length=0)


def dollars(ax, axis="y", thousands=False):
    """Format an axis as currency."""
    fmt = (FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k") if thousands
           else FuncFormatter(lambda v, _: f"${v:,.0f}"))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def title_block(fig, title, subtitle, note=None, x=0.09, y=0.945):
    """Headline, one-line subtitle, optional emphasised note."""
    fig.text(x, y, title, fontsize=15.5, color=INK, fontweight="bold", ha="left")
    fig.text(x, y - 0.057, subtitle, fontsize=9.6, color=MUTED, ha="left")
    if note:
        fig.text(x, y - 0.113, note, fontsize=9.6, color=INK_SOFT,
                 ha="left", style="italic")


def source_note(fig, lines, x=0.09, y=0.03, step=0.03):
    """One or more source lines along the bottom."""
    if isinstance(lines, str):
        lines = [lines]
    for i, line in enumerate(lines):
        fig.text(x, y + (len(lines) - 1 - i) * step, line,
                 fontsize=7.5, color=MUTED, ha="left")
