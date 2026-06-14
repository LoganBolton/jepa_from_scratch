# uv run torchrun --nproc_per_node=2 train.py

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as T
import wandb
from transformers import get_cosine_schedule_with_warmup

from vit import *
from predictor import *
from utils import *
from mask import MaskData
from eval import knn_eval, build_eval_dataloaders

EPOCHS = 1_000
LR = 1e-4
BATCH_SIZE = 256
NUM_TARGET_BLOCKS = 4
EMA_START = 0.996
EMA_END = 1.0
WARMUP_EPOCHS = 10
SAVE_EVERY = 20


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return dist.get_rank(), local_rank, dist.get_world_size(), device


def build_loader(rank, world_size, is_main):
    transform = T.Compose(
        [T.Resize((64, 64)), 
         T.RandomHorizontalFlip(),
         T.ToTensor()]
        )
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    dist.barrier()
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=2, drop_last=True, pin_memory=True,
    )
    return loader, sampler


def build_models(device, local_rank):
    context_encoder = ViT(img_size=64, patch_size=4).to(device)
    prediction_encoder = Predictor(
        encoder_dim=512, num_patches=context_encoder.num_patches
    ).to(device)

    target_encoder = copy.deepcopy(context_encoder)
    for p in target_encoder.parameters():
        p.requires_grad_(False)

    context_encoder = DDP(context_encoder, device_ids=[local_rank])
    prediction_encoder = DDP(prediction_encoder, device_ids=[local_rank])
    return context_encoder, prediction_encoder, target_encoder


def jepa_loss(imgs, context_encoder, prediction_encoder, target_encoder, masker, loss_fn, device):
    B = imgs.shape[0]
    context_idx, target_blocks = masker.get_indices()
    context_idx = batchify(context_idx, B).to(device)

    target_flat = [i for block in target_blocks for i in block]
    target_idx = batchify(target_flat, B).to(device)

    context_embed = context_encoder(imgs, context_idx)
    with torch.no_grad():
        full_reps = target_encoder(imgs)

    pred_embed = prediction_encoder(context_embed, context_idx, target_idx)
    target_embed = gather_tokens(full_reps, target_idx)
    target_embed = F.layer_norm(target_embed, (target_embed.size(-1),))

    loss = loss_fn(pred_embed, target_embed)
    rep_std = full_reps.std(dim=0).mean().item()
    return loss, rep_std


def main():
    rank, local_rank, world_size, device = setup_ddp()
    is_main = rank == 0

    if is_main:
        wandb.init(project="jepa-from-scratch", config={
            "epochs": EPOCHS, "lr": LR, "batch_size_per_gpu": BATCH_SIZE,
            "world_size": world_size, "num_target_blocks": NUM_TARGET_BLOCKS,
            "ema_start": EMA_START, "ema_end": EMA_END,
        })

    loader, sampler = build_loader(rank, world_size, is_main)
    context_encoder, prediction_encoder, target_encoder = build_models(device, local_rank)
    eval_train_loader, eval_test_loader = build_eval_dataloaders()

    params = list(context_encoder.parameters()) + list(prediction_encoder.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=LR
    )
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_EPOCHS*len(loader), num_training_steps=EPOCHS*len(loader))

    loss_fn = nn.MSELoss()
    masker = MaskData(NUM_TARGET_BLOCKS, int(context_encoder.module.num_patches ** 0.5))

    total_steps = EPOCHS * len(loader)
    
    global_step = 0

    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        for step, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)

            loss, rep_std = jepa_loss(
                imgs, context_encoder, prediction_encoder, target_encoder,
                masker, loss_fn, device,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()

            m = EMA_START + (EMA_END - EMA_START) * (global_step / total_steps)
            ema_update(target_encoder, context_encoder.module, m)
            global_step += 1

            # if is_main:
            #     wandb.log({"loss": loss.item(), "rep_std": rep_std, "ema_m": m, "epoch": epoch})
                # if step % 50 == 0:
                #     print(f"epoch {epoch} step {step} loss {loss.item():.4f} "
                #           f"rep_std {rep_std:.4f} m {m:.5f}")
        knn_acc = None
        if epoch % SAVE_EVERY == 0:
            if is_main:
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "target_encoder": target_encoder.state_dict(),
                    "context_encoder": context_encoder.module.state_dict(),  # .module unwraps DDP
                    "predictor": prediction_encoder.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, f"checkpoints/ckpt_{epoch}.pt")
                knn_acc = knn_eval(target_encoder, eval_train_loader, eval_test_loader, device)
            dist.barrier()

        if is_main:
            log = {"loss": loss.item(), 
                    "rep_std": rep_std, 
                    "ema_m": m,
                    "epoch": epoch, 
                    "LR": scheduler.get_last_lr()[0]}
            if knn_acc is not None:
                log["knn_acc"] = knn_acc
            wandb.log(log, step=epoch)
            
            print(f"epoch {epoch} loss {loss.item():.4f} "
                    f"rep_std {rep_std:.4f}")

    if is_main:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
