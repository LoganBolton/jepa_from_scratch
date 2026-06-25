# JEPA from scratch

I trained I-JEPA from scratch on CIFAR-10, then trained a vanilla ViT as a baseline. After pretraining on CIFAR, I then evaluated both models on EuroSAT, an out of domain dataset of satellite imagery. Both models were evaluated with K nearest neighbors and also with a fine tuned linear probe.

I-JEPA performed significantly worse on CIFAR, but was better on the pretrained ViT on EuroSat. This supports LeCun's that self supervised learning with image embeddings produces superior representations than with supervised learning. 

However, I-JEPA was significantly tricker to train. Originally, my masking code wasn't obscuring enough of the image leading to the model easily being able to predict the embeddings of these patches. I also did some tuning of the training. The ViT on the other hand worked well out of the box. It was super easy to train and didn't require much training compute or hyperparameter optimziation. My takeaway from this is that if you are just training from scratch on a particular, make your life easier and do supervised learning! However if are looking to save on training compute, it's best to finetune a big self supervised model. 

![kNN accuracy: in-domain vs. a new domain](assets/knn_comparison.png)


JEPA (475 epochs)
ViT (25 epochs)
Linear probe trained 100 epochs on frozen features.

| | CIFAR-10 kNN | CIFAR-10 probe | EuroSAT kNN | EuroSAT probe |
|---|---:|---:|---:|---:|
| **JEPA** (frozen, self-supervised) | 56.5% | 69.0% | **81.5%** | **86.9%** |
| **Supervised ViT** (frozen) | **69.8%** | 69.9% | 75.2% | 84.0% |

## Reproduce

```bash
# self-supervised JEPA pretraining (2 GPUs)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run torchrun --nproc_per_node=2 train.py

# supervised ViT baseline
uv run torchrun --nproc_per_node=2 train.py --baseline

# frozen-encoder eval (works on JEPA and ViT)
uv run python probe.py checkpoints/{checkpoint_name}.pt   --dataset cifar10 --epochs 100
```
