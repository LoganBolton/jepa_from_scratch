"""Render the kNN in-domain vs domain-shift comparison for the README."""
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

datasets = ["CIFAR-10", "EuroSAT"]
jepa = [56.5, 81.5]
vit = [69.8, 75.2]

JEPA_C = "#1FA391"   # teal — the model we're showcasing
VIT_C = "#C2C7CE"    # muted grey — the baseline
TEXT = "#2B2F36"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "text.color": TEXT,
    "axes.edgecolor": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
})

fig, ax = plt.subplots(figsize=(7, 4.4), dpi=200)

x = range(len(datasets))
w = 0.36
b1 = ax.bar([i - w / 2 for i in x], jepa, w, label="JEPA", color=JEPA_C, zorder=3)
b2 = ax.bar([i + w / 2 for i in x], vit, w, label="ViT", color=VIT_C, zorder=3)

for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{bar.get_height():.0f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

# highlight JEPA's margin over ViT on the unseen domain
delta = round(jepa[1]) - round(vit[1])
ax.text(1 - w / 2 + 0.05, jepa[1] + 1.9, f"(+{delta})", ha="left", va="bottom",
        fontsize=10, fontweight="bold", color="#16A34A")

ax.set_xticks(list(x))
ax.set_xticklabels(datasets, fontsize=13, fontweight="bold")
ax.set_ylabel("kNN accuracy (%)")
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 20))

# in-domain vs domain-shift framing, kept light
ax.text(0, -12, "trained on", ha="center", fontsize=10, style="italic", color="#7A828C")
ax.text(1, -12, "never seen", ha="center", fontsize=10, style="italic", color="#7A828C")

ax.set_title("JEPA adapts better to a new domain", fontsize=15,
             fontweight="bold", pad=14)

ax.grid(axis="y", color="#E6E8EC", zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.tick_params(length=0)
leg = ax.legend(loc="upper left", fontsize=12, bbox_to_anchor=(1.0, 1.08),
                frameon=True, borderpad=0.4, handletextpad=0.5,
                labelspacing=0.3)
leg.get_frame().set_edgecolor("#D5D9DF")
leg.get_frame().set_facecolor("white")
leg.get_frame().set_linewidth(1.0)
leg.get_frame().set_boxstyle("round,pad=0.2,rounding_size=0.4")

fig.tight_layout()
os.makedirs("assets", exist_ok=True)
out = "assets/knn_comparison.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
