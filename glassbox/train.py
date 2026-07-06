"""A minimal, readable training loop for the GPT model.

Nothing exotic: AdamW, an optional cosine-ish constant LR, gradient steps over
random batches, and a periodic held-out loss estimate. The function returns the
history of train/val losses so a caller (the demo script) can plot a loss curve.

We keep the training deterministic where PyTorch allows (fixed seeds + a seeded
batch generator) so results are reproducible across runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .data import TextData
from .model import GPT


@dataclass
class TrainConfig:
    max_iters: int = 2000
    batch_size: int = 32
    eval_interval: int = 100
    eval_iters: int = 20          # batches averaged for each loss estimate
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    seed: int = 1337
    device: str = "cpu"
    log: bool = True
    history: list = field(default_factory=list)


def set_seed(seed: int) -> None:
    """Seed every RNG we touch so runs are reproducible."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def estimate_loss(model: GPT, data: TextData, cfg: TrainConfig,
                  generator: torch.Generator) -> dict[str, float]:
    """Average the loss over a few batches of train and val, with dropout off."""
    model.eval()
    out: dict[str, float] = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            xb, yb = data.get_batch(split, cfg.batch_size, generator=generator)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model: GPT, data: TextData, cfg: TrainConfig
          ) -> list[tuple[int, float, float]]:
    """Train `model` on `data`. Returns a list of (iter, train_loss, val_loss)."""
    set_seed(cfg.seed)
    model.to(cfg.device)
    model.train()

    # AdamW = Adam with decoupled weight decay. Weight decay is a mild pull of
    # weights toward zero that regularises the model; we (conventionally) do NOT
    # decay biases or LayerNorm gains, only the matmul weights.
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    optim = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate, betas=(0.9, 0.99),
    )

    # A dedicated, seeded generator for batch sampling -> reproducible batches.
    gen = torch.Generator().manual_seed(cfg.seed)

    history: list[tuple[int, float, float]] = []
    for it in range(cfg.max_iters + 1):
        # Periodically estimate and record held-out loss.
        if it % cfg.eval_interval == 0 or it == cfg.max_iters:
            losses = estimate_loss(model, data, cfg, gen)
            history.append((it, losses["train"], losses["val"]))
            if cfg.log:
                print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")

        if it == cfg.max_iters:
            break

        # One optimisation step.
        xb, yb = data.get_batch("train", cfg.batch_size, generator=gen)  # (B, T)
        _, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        # Clip gradients to keep training stable if a batch produces a large loss.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

    return history
