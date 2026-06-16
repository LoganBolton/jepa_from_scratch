# Load a frozen backbone and evaluate it with kNN + a linear probe.
# Works on both JEPA checkpoints and supervised-baseline checkpoints, so the
# two can be compared frozen-vs-frozen on equal terms.
#   uv run python probe.py checkpoints/ckpt_175.pt        # JEPA target encoder
#   uv run python probe.py checkpoints/baseline_175.pt    # supervised backbone

import argparse
import torch

from vit import ViT
from eval import build_eval_dataloaders, knn_eval, linear_probe, NUM_CLASSES


def load_encoder(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "target_encoder" in ckpt:
        state, kind = ckpt["target_encoder"], "jepa target encoder"
    elif "backbone" in ckpt:
        state, kind = ckpt["backbone"], "supervised backbone"
    else:
        raise KeyError(
            f"{ckpt_path} has no 'target_encoder' or 'backbone' key; "
            f"found {list(ckpt.keys())}"
        )
    encoder = ViT(img_size=64, patch_size=4).to(device)
    encoder.load_state_dict(state)
    encoder.eval()
    return encoder, ckpt.get("epoch"), kind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", help="path to a JEPA checkpoint, e.g. checkpoints/ckpt_175.pt")
    parser.add_argument("--dataset", default="cifar10", choices=list(NUM_CLASSES),
                        help="dataset to probe transfer on (eurosat = domain-shift target)")
    parser.add_argument("--epochs", type=int, default=10, help="linear probe training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="linear probe learning rate")
    parser.add_argument("--no-knn", action="store_true", help="skip the kNN eval")
    parser.add_argument("--labels-per-class", type=int, default=None,
                        help="limit eval labels per class to simulate a low-label regime")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, epoch, kind = load_encoder(args.ckpt, device)
    print(f"loaded {args.ckpt} ({kind}, epoch {epoch})")
    print(f"probing on {args.dataset} ({NUM_CLASSES[args.dataset]} classes)")

    train_loader, test_loader = build_eval_dataloaders(dataset=args.dataset)

    if args.labels_per_class is not None:
        print(f"low-label regime: {args.labels_per_class} labels/class")

    if not args.no_knn:
        knn_acc = knn_eval(encoder, train_loader, test_loader, device,
                           labels_per_class=args.labels_per_class)
        print(f"knn accuracy:          {knn_acc:.4f}")

    probe_acc = linear_probe(
        encoder, train_loader, test_loader, device,
        epochs=args.epochs, lr=args.lr,
        num_classes=NUM_CLASSES[args.dataset],
        labels_per_class=args.labels_per_class,
    )
    print(f"linear probe accuracy: {probe_acc:.4f}")


if __name__ == "__main__":
    main()
