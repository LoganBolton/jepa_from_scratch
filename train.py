import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import copy
import math

from vit import *
from predictor import *
from utils import *
from mask import MaskData

EPOCHS = 10
LR = 1e-4
BATCH_SIZE = 64
NUM_TARGET_BLOCKS = 4
m = 0.996

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.ToTensor(),
])
dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset, BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True
)

context_encoder = ViT(img_size=32, patch_size=4).to(device)
prediction_encoder = Predictor(
    encoder_dim=512, num_patches=context_encoder.num_patches
).to(device)

target_encoder = copy.deepcopy(context_encoder)
for p in target_encoder.parameters():
    p.requires_grad_(False)

optimizer = torch.optim.AdamW(
    list(context_encoder.parameters()) + list(prediction_encoder.parameters()),
    lr=LR,
)
loss_fn = nn.MSELoss()
masker = MaskData(NUM_TARGET_BLOCKS, grid_size=int(context_encoder.num_patches ** 0.5))

for epoch in range(EPOCHS):
    for step, (imgs, _) in enumerate(loader):
        imgs = imgs.to(device)
        B = imgs.shape[0]

        context_indices, target_blocks = masker.get_indices()
        context_indices = batchify(context_indices, B).to(device)

        context_embed = context_encoder(imgs, context_indices)
        with torch.no_grad():
            full_reps = target_encoder(imgs)

        loss = 0
        for target_block in target_blocks:
            target_idx = batchify(target_block, B).to(device)
            pred_embed = prediction_encoder(context_embed, context_indices, target_idx)
            target_embed = gather_tokens(full_reps, target_idx)
            loss += loss_fn(pred_embed, target_embed)

        loss /= NUM_TARGET_BLOCKS
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_update(target_encoder, context_encoder, m)

        if step % 50 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
