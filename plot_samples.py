"""Sample strips for the two datasets, styled to match the kNN plot."""
import re
import torchvision
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox

TEXT = "#2B2F36"
SUB = "#7A828C"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
})


def first_index_per_class(targets, class_to_idx, wanted):
    """First dataset index for each wanted class name."""
    out = []
    for name in wanted:
        c = class_to_idx[name]
        out.append(next(i for i, t in enumerate(targets) if t == c))
    return out


def prettify(name):
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)


cifar = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=None)
eurosat = torchvision.datasets.EuroSAT(root="./data", download=True, transform=None)

cifar_classes = ["cat", "dog", "ship", "truck"]
eurosat_classes = ["Forest", "Highway", "River", "Residential"]

blocks = [
    ("CIFAR-10", "trained on", cifar, cifar_classes),
    ("EuroSAT", "never seen", eurosat, eurosat_classes),
]

fig = plt.figure(figsize=(7.3, 4.6), dpi=200)
subfigs = fig.subfigures(2, 1, hspace=0.12)

for sf, (name, sub, ds, wanted) in zip(subfigs, blocks):
    idxs = first_index_per_class(ds.targets, ds.class_to_idx, wanted)
    axs = sf.subplots(1, len(wanted))
    sf.subplots_adjust(top=0.80, bottom=0.14, left=0.02, right=0.98, wspace=0.12)
    for ax, i, cls in zip(axs, idxs, wanted):
        img, _ = ds[i]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel(prettify(cls), fontsize=11, labelpad=5)

    # header: bold dataset name + light grey italic "(...)", matching the kNN plot
    name_box = TextArea(name, textprops=dict(fontsize=14, fontweight="bold", color=TEXT))
    sub_box = TextArea(f"({sub})", textprops=dict(fontsize=11, style="italic", color=SUB))
    header = HPacker(children=[name_box, sub_box], align="baseline", pad=0, sep=8)
    ab = AnnotationBbox(header, (0.5, 0.99), xycoords=sf.transSubfigure,
                        frameon=False, box_alignment=(0.5, 1.0))
    sf.add_artist(ab)

fig.savefig("assets/dataset_samples.png", bbox_inches="tight", facecolor="white")
print("wrote assets/dataset_samples.png")
