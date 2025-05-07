from pathlib import Path
import numpy as np
from datetime import datetime
import sys

sys.path.append("..")

import torch.nn as nn
import torch
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
import tqdm
from noise_designer import load_batches
from unet import UNet
from torch.utils.tensorboard import SummaryWriter

import random

def create_experiment_folder(name: str = "run") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(__file__).parent  / "runs" / f"{timestamp}_{name}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
verbose: int = 2

class LineSegmentationDataset(Dataset):
    def __init__(self, scrolls: np.ndarray, line: np.ndarray):
        self.scrolls = scrolls
        self.lines = line

    def __len__(self):
        return self.scrolls.shape[0]

    def __getitem__(self, index):
        scrolls = torch.tensor(self.scrolls[index], dtype=torch.float32).unsqueeze(0)
        scrolls = 1- (scrolls / 255)
        lines = torch.tensor(self.lines[index], dtype=torch.float32).unsqueeze(0)
        return scrolls, lines


def train_epoch(
    model: UNet,
    train_data: LineSegmentationDataset,
    validation_data: LineSegmentationDataset,
    optimizer: Optimizer,
    criterion: nn.Module,
    batch_size: int = 32,
):

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size)
    val_loader = DataLoader(dataset=validation_data, batch_size=batch_size)

    #Training
    model.train()
    track_loss = []
    for batch_scrolls, batch_lines in tqdm.tqdm(train_loader, total=len(train_loader), disable=verbose < 2):

        batch_scrolls, batch_lines = batch_scrolls.to(device), batch_lines.to(device)
        lines_hat = model(batch_scrolls)
        loss = criterion(lines_hat, batch_lines)
        track_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #Validation
    model.eval()
    val_track_loss = []
    with torch.no_grad():
        for batch_scrolls, batch_lines in val_loader:
            batch_scrolls, batch_lines = batch_scrolls.to(device), batch_lines.to(device)
            lines_hat = model(batch_scrolls)
            loss = criterion(lines_hat, batch_lines)
            val_track_loss.append(loss.item())

    return np.mean(track_loss), np.mean(val_track_loss)


def train_level(
        pool: set,
        model: UNet | None = None,
        optimizer: Optimizer | None = None,
        experiment_folder: Path | None = None,
        experiment_name: str | None = "unet",
        epoch: int | None = None
    ):

    level = random.choice(list(pool))
    print(f"------------------Noise Level {level}------------------")
    iterator = load_batches(level=level)
    _, val_scrolls, val_lines = next(iterator)
    val_data = LineSegmentationDataset(val_scrolls, val_lines)

    if model is None:
        model = UNet()
        model = model.to(device)

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    criterion = nn.BCEWithLogitsLoss()

    for _, train_scrolls, train_lines in iterator:
        train_data = LineSegmentationDataset(train_scrolls, train_lines)
        train_loss, val_losss = train_epoch(
            model,
            train_data,
            val_data,
            optimizer,
            criterion
        )
        model.save(experiment_folder)
        with open(experiment_folder / "loss.txt", "a") as f:
            f.write(f"{train_loss:.4f},{val_losss:.4f}\n")
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_losss, epoch)
        if verbose > 0:
            print(f"Losss {train_loss:.4f}, {val_losss:.4f}")

    return model, optimizer

if __name__ == "__main__":
    experiment_folder = create_experiment_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(__file__).parent  / "runs" / f"{timestamp}_run"
    writer = SummaryWriter(log_dir=f"truns/{folder}")

    model, optimizer =  train_level(pool = {0}, experiment_folder=experiment_folder)
    print("Starting noise trainig")
    for epoch in range(200):
        train_level(model=model, pool = {i for i in range(5)}, optimizer=optimizer, experiment_folder=experiment_folder, epoch=epoch)

