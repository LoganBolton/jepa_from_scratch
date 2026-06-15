# JEPA from scratch — CIFAR-10

Reference points: random baseline **29.2%**, raw-pixel kNN **36.9%**.

## Frozen backbone, full labels

| Model | Backbone | kNN | Linear probe |
|-------|----------|----:|-------------:|
| JEPA (ckpt 175) | frozen | 54.0% | 57.4% |
| Supervised (ckpt 25) | frozen | 69.8% | 70.0% |

## Low-label regime (frozen backbone)

| Labels/class | JEPA kNN | JEPA probe | Supervised kNN | Supervised probe |
|-------------:|---------:|-----------:|---------------:|-----------------:|
| 5 | 16.1% | 20.8% | 44.9% | 64.5% |
| 50 | 37.3% | 37.0% | 67.3% | 68.6% |
| all (500) | 54.0% | 57.4% | 69.8% | 70.0% |

The supervised "backbone" was pretrained on all 50k labels, so probing it with few labels is not a true low-label test — its features are already class-aligned. A fair low-label comparison requires training the supervised model from scratch on the restricted label set.
