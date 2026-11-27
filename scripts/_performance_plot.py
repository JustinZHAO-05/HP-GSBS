"""Shared layout and asset export for the README benchmark figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORKLOAD_LABELS = ["Small", "Medium", "Large", "XL", "BBBD0 (EEG)"]
SERIES = {
    "reference": ("Official GSBS", "#4D5C70", "s"),
    "cpu-fast": ("CPU", "#B34D15", "o"),
    "torch-cuda": ("PyTorch", "#087D7D", "D"),
    "triton-cuda": ("Triton", "#6546BD", "^"),
}


def setup(salt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "svg.fonttype": "none",
        "svg.hashsalt": salt,
        "text.color": "#243247",
        "axes.labelcolor": "#243247",
        "xtick.color": "#405065",
        "ytick.color": "#243247",
    })
    fig, ax = plt.subplots(figsize=(9, 3.65))
    fig.subplots_adjust(left=.20, right=.96, bottom=.19, top=.81)
    ax.set_yticks(range(len(WORKLOAD_LABELS)), WORKLOAD_LABELS)
    ax.set_ylim(len(WORKLOAD_LABELS) - .55, -.55)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E1E6EC", linewidth=.8)
    for boundary in range(len(WORKLOAD_LABELS) - 1):
        ax.axhline(boundary + .5, color="#EEF1F5", linewidth=.8, zorder=0)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#C7CED8")
    ax.tick_params(length=0, pad=8)
    return fig, ax


def save(fig, ax, stem, series_count):
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.06), ncol=series_count,
        frameon=False, borderaxespad=0, handletextpad=.35,
    )
    fig.savefig(stem.parent / f"{stem.name}.png", dpi=180)
    svg = stem.parent / f"{stem.name}.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8", newline="\n",
    )
    plt.close(fig)
