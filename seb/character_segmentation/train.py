import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from unet_character import UNet

import sys
sys.path.append("../..")
from synthetic import DataGenerator, SynthSettings


def parse_args():
    parser = argparse.ArgumentParser(description="UNet Training Script")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--downscale", type=float, default=0.3)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()


class ScrollDataset(Dataset):
    def __init__(self, scrolls: np.ndarray, masks: np.ndarray):
        self.scrolls = torch.from_numpy(scrolls).float().unsqueeze(1)
        self.masks = torch.from_numpy(masks).float()

    def __len__(self):
        return len(self.scrolls)

    def __getitem__(self, idx):
        return self.scrolls[idx], self.masks[idx]


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
    avg_loss = float(np.mean(losses))
    print(f"  Training Loss: {avg_loss:.4f}")
    return avg_loss


def validate_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)
            losses.append(loss.item())
    avg_loss = float(np.mean(losses))
    print(f"  Validation Loss: {avg_loss:.4f}")
    return avg_loss


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"unet_{timestamp}"
    ckpt_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data generator
    gen_settings = SynthSettings(downscale_factor=args.downscale)
    generator = DataGenerator(settings=gen_settings)

    # Model, loss, optimizer
    model = UNet(num_classes=27).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch:03d}")
        # Generate fresh batches each epoch
        _, train_masks, train_scrolls, _ = generator.generate_ngram_scrolls(
            args.train_size, skip_char_seg=False
        )
        _, val_masks, val_scrolls, _ = generator.generate_ngram_scrolls(
            args.val_size, skip_char_seg=False
        )

        train_ds = ScrollDataset(train_scrolls, train_masks)
        val_ds = ScrollDataset(val_scrolls, val_masks)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # Log and checkpoint
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)

        with open(os.path.join(ckpt_dir, "loss.txt"), "a") as f:
            f.write(f"{train_loss:.4f},{val_loss:.4f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pt"))

    writer.close()


if __name__ == "__main__":
    main()