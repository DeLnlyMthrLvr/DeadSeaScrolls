import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project path
sys.path.append("..")
from unet import UNet
from synthetic import DataGenerator, SynthSettings

# Typing aliases
LineMasks = np.ndarray
ScrollImages = np.ndarray

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Run config
run_name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
checkpoint_dir = os.path.join("checkpoints_line", run_name)
os.makedirs(checkpoint_dir, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join("runs_line", run_name))

best_val_loss = float('inf')


class CustomDataset(Dataset):
    def __init__(self, scrolls: ScrollImages, masks: LineMasks):
        self.scrolls = scrolls
        self.masks = masks

    def __len__(self) -> int:
        return len(self.scrolls)

    def __getitem__(self, index: int):
        x = torch.tensor(self.scrolls[index], dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        y = torch.tensor(self.masks[index], dtype=torch.float32).unsqueeze(0)    # (1, H, W)
        return x, y


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_function) -> float:
    model.train()
    train_losses = []

    for batch_scrolls, batch_masks in dataloader:
        batch_scrolls, batch_masks = batch_scrolls.to(device), batch_masks.to(device)

        predictions = model(batch_scrolls)
        loss = loss_function(predictions, batch_masks)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_losses.append(loss.item())

    return np.mean(train_losses)


def validate_one_epoch(model: nn.Module, dataloader: DataLoader, loss_function) -> float:
    model.eval()
    val_losses = []

    with torch.no_grad():
        for batch_scrolls, batch_masks in dataloader:
            batch_scrolls, batch_masks = batch_scrolls.to(device), batch_masks.to(device)

            predictions = model(batch_scrolls)
            loss = loss_function(predictions, batch_masks)

            val_losses.append(loss.item())

    return np.mean(val_losses)


def train(model: nn.Module, 
          train_scrolls: ScrollImages, train_masks: LineMasks,
          val_scrolls: ScrollImages, val_masks: LineMasks):

    train_dataset = CustomDataset(train_scrolls, train_masks)
    val_dataset = CustomDataset(val_scrolls, val_masks)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loss = train_one_epoch(model, train_loader, optimizer, loss_function)
    val_loss = validate_one_epoch(model, val_loader, loss_function)

    return train_loss, val_loss


if __name__ == "__main__":
    gen_settings = SynthSettings(downscale_factor=0.3)
    generator = DataGenerator(settings=gen_settings)
    model = UNet(num_classes=1).to(device)

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting Training")
    epoch = 0

    max_epochs = 10  # Optional: to avoid infinite training

    for epoch in range(0,max_epochs):
        # Generate new synthetic dataset
        tokens, masks, scrolls, line_masks = generator.generate_ngram_scrolls(1_000)
        val_tokens, val_masks, val_scrolls, val_line_masks = generator.generate_ngram_scrolls(200)

        train_loss, val_loss = train(
            model=model,
            train_scrolls=scrolls, train_masks=line_masks,
            val_scrolls=val_scrolls, val_masks=val_line_masks
        )

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        # Save model checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f"model_epoch_{epoch:03d}.pth"))

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))

        epoch += 1

    print("Training completed.")
