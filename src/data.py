"""
CIFAR-10 loading, stratified sub-sampling and GPU-side augmentation.

The whole dataset (50k + 10k images, uint8) is only ~180 MB, so it is kept
resident on the GPU. This removes the CPU DataLoader from the measurement
(Windows DataLoader workers add large, noisy overheads) and makes the time /
energy we measure attributable to the model itself.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_cifar10(root: str, device: str = "cuda"):
    tr = CIFAR10(root, train=True, download=True)
    te = CIFAR10(root, train=False, download=True)
    to_t = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW uint8
    return {
        "x_train": to_t(tr.data).to(device),
        "y_train": torch.tensor(tr.targets, dtype=torch.long, device=device),
        "x_test": to_t(te.data).to(device),
        "y_test": torch.tensor(te.targets, dtype=torch.long, device=device),
    }


def stratified_subset(y: torch.Tensor, fraction: float, seed: int) -> torch.Tensor:
    """Indices of a class-balanced random subset. Same (fraction, seed) -> same subset for every model."""
    if fraction >= 1.0:
        return torch.arange(len(y), device=y.device)
    rng = np.random.default_rng(seed)
    y_np = y.cpu().numpy()
    idx = []
    for c in range(10):
        cls = np.where(y_np == c)[0]
        k = int(round(len(cls) * fraction))
        idx.append(rng.choice(cls, size=k, replace=False))
    idx = np.sort(np.concatenate(idx))
    return torch.from_numpy(idx).to(y.device)


def normalize(x_uint8: torch.Tensor) -> torch.Tensor:
    x = x_uint8.float().div_(255.0)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random crop (pad 4) + random horizontal flip, vectorised on the GPU."""
    B = x.shape[0]
    dev = x.device
    xp = F.pad(x, (4, 4, 4, 4))
    dy = torch.randint(0, 9, (B,), device=dev, generator=gen)
    dx = torch.randint(0, 9, (B,), device=dev, generator=gen)
    ar = torch.arange(32, device=dev)
    rows = (dy[:, None] + ar)[:, None, :, None]                 # B,1,32,1
    cols = (dx[:, None] + ar)[:, None, None, :]                 # B,1,1,32
    bidx = torch.arange(B, device=dev)[:, None, None, None]
    cidx = torch.arange(x.shape[1], device=dev)[None, :, None, None]
    out = xp[bidx, cidx, rows, cols]
    flip = torch.rand(B, device=dev, generator=gen) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)
