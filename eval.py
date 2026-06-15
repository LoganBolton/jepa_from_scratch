import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as T
import torchvision.datasets.cifar

def build_eval_dataloaders(batch_size=256):
    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)
    test_set  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

@torch.no_grad()
def extract_features(encoder, loader, device="cuda"):
    encoder.eval()
    feats, labels = [], []
    for imgs, y in loader:
        out = encoder(imgs.to(device)) # (B, num_patches, D)
        out = out.mean(dim=1) # (B, D)
        feats.append(out.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)

@torch.no_grad()
def cache_features(encoder, loader, device="cuda"):
    """Run the frozen encoder once and keep mean-pooled features on device."""
    feats, labels = extract_features(encoder, loader, device)
    return feats.to(device), labels.to(device)


def linear_probe(encoder, train_loader, test_loader, device="cuda",
                 epochs=100, lr=1e-3, weight_decay=0.0, num_classes=10):
    """Freeze the encoder, train a single linear layer on its features.

    This is the standard SSL eval protocol and the fair comparison against a
    supervised backbone. Features are cached once since the encoder is frozen
    and the eval transforms are deterministic.
    """
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    train_f, train_y = cache_features(encoder, train_loader, device)
    test_f, test_y = cache_features(encoder, test_loader, device)

    # normalize features using train statistics
    mean = train_f.mean(dim=0, keepdim=True)
    std = train_f.std(dim=0, keepdim=True) + 1e-6
    train_f = (train_f - mean) / std
    test_f = (test_f - mean) / std

    head = nn.Linear(train_f.size(1), num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    n = train_f.size(0)
    batch_size = 1024
    best_acc = 0.0
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = head(train_f[idx])
            loss = loss_fn(logits, train_y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            preds = head(test_f).argmax(dim=1)
            acc = (preds == test_y).float().mean().item()
        best_acc = max(best_acc, acc)

    return best_acc


def knn_eval(encoder, train_loader, test_loader, device="cuda", k=20):
    train_f, train_y = extract_features(encoder, train_loader, device)
    train_f = F.normalize(train_f, dim=1)
    test_f, test_y = extract_features(encoder, test_loader, device)
    test_f = F.normalize(test_f, dim=1)
    
    train_f, train_y = train_f.to(device), train_y.to(device)
    test_f, test_y   = test_f.to(device), test_y.to(device)

    
    chunk_size = 256
    correct = 0
    for i in range(0, test_f.size(0), chunk_size):
        f = test_f[i: i + chunk_size]
        y = test_y[i: i + chunk_size]
        
        sims = f @ train_f.T
        idx = sims.topk(k, dim=1).indices
        preds = train_y[idx].mode(dim=1).values # majority vote
        correct += (preds==y).sum().item()
    acc = correct / test_f.size(0)
    return acc
    