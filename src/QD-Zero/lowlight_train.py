"""Train the qd-zero enhancement branch using unpaired low-light images."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dataloader import lowlight_loader
from model import QDZero, load_qd_zero
from Myloss import QDZeroLoss


def weights_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def train(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(config.device)
    dataset = lowlight_loader(config.lowlight_images_path, size=config.image_size)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=device.type == "cuda",
    )
    checkpoint = None
    if config.resume:
        network, checkpoint = load_qd_zero(config.resume, device)
    else:
        network = QDZero().to(device)
        network.apply(weights_init)
    objective = QDZeroLoss(
        lambda_spa=config.lambda_spa, lambda_exp=config.lambda_exp,
        lambda_col=config.lambda_col, lambda_tv=config.lambda_tv,
        lambda_qd=config.lambda_qd, omega=config.omega,
    ).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    start_epoch = 0
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
    output = Path(config.snapshots_folder)
    output.mkdir(parents=True, exist_ok=True)
    network.train()
    for epoch in range(start_epoch, config.num_epochs):
        for iteration, original in enumerate(loader, start=1):
            original = original.to(device, non_blocking=True)
            _, enhanced, curves = network(original)
            total, terms = objective(original, enhanced, curves)
            if not torch.isfinite(total):
                raise FloatingPointError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(network.parameters(), config.grad_clip_norm,
                                     error_if_nonfinite=True)
            optimizer.step()
            if iteration % config.display_iter == 0:
                values = " ".join("{}={:.6f}".format(key, value.detach().item())
                                  for key, value in terms.items())
                print("epoch={} step={} total={:.6f} {}".format(
                    epoch + 1, iteration, total.item(), values), flush=True)
        # A complete epoch checkpoint avoids losing short training runs.
        payload = {
            "architecture": "qd-zero", "model_config": network.model_config,
            "state_dict": network.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "training_config": vars(config),
        }
        torch.save(payload, output / "qd_zero_last.pth")
        if (epoch + 1) % config.snapshot_iter == 0:
            torch.save(payload, output / "qd_zero_epoch_{:04d}.pth".format(epoch + 1))


def build_parser():
    parser = argparse.ArgumentParser(description="Train qd-zero")
    parser.add_argument("--lowlight_images_path", default="data/train_data/")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--display_iter", type=int, default=10)
    parser.add_argument("--snapshot_iter", type=int, default=10, help="epoch interval")
    parser.add_argument("--snapshots_folder", default="snapshots/")
    parser.add_argument("--resume", default=None, help="qd-zero training checkpoint")
    parser.add_argument("--seed", type=int, default=1143)
    for name, value in (("spa", 1.0), ("exp", 10.0), ("col", 5.0),
                        ("tv", 200.0), ("qd", 1.0)):
        parser.add_argument("--lambda_" + name, type=float, default=value)
    parser.add_argument("--omega", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if min(args.num_epochs, args.train_batch_size, args.display_iter,
           args.snapshot_iter) < 1 or args.num_workers < 0:
        parser.error("counts and intervals must be positive; workers may be zero")
    train(args)
