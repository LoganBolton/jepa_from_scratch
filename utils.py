import torch

def gather_tokens(x, mask_indices):
    B = mask_indices.shape[0] # (B, n_keep)
    D = x.shape[-1] # (1, N, D)
    
    x = x.expand(B, -1, -1 ) # (1, N, D) -> (B, N, D)
    mask_indices = mask_indices.unsqueeze(-1).expand(-1,-1,D) # (B, n_keep) -> (B, n_keep, D)
    
    # drop patches
    x = torch.gather(x, 1, mask_indices)
    return x


def batchify(idx, B):
    # plain list of flat indices -> (B, n) shared across the batch
    return torch.tensor(idx).unsqueeze(0).expand(B, -1)


def ema_update(target_encoder, context_encoder, m):
    for tp, cp in zip(target_encoder.parameters(), context_encoder.parameters()):
        tp.data = m * tp.data + (1-m) * cp.data