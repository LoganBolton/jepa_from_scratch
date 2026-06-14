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
    