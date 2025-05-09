from pathlib import Path
import numpy as np
from datetime import datetime
import sys
import cv2
from unet_line import UNet
import os
sys.path.append("../../")
sys.path.append(str(Path(__file__).parent.parent))
import torch.nn as nn
import torch
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
import tqdm
from noise_designer import load_batches
from torch.utils.tensorboard import SummaryWriter

import random

def create_experiment_folder(name: str = "run") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(__file__).parent  / "runs" / f"{timestamp}_{name}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
verbose: int = 2

def resize(imgs: np.ndarray, factor: float):

    height, width = imgs.shape[1:]
    nh = int(height * factor)
    nw = int(width * factor)

    new_imgs = []
    for img in imgs:
        ri = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        new_imgs.append(ri)

    return np.stack(new_imgs)

class LineSegmentationDataset(Dataset):
    def __init__(self, scrolls: np.ndarray, line: np.ndarray):
        factor = np.random.uniform(0.4, 0.6)
        self.scrolls = resize(scrolls, factor=factor)
        self.lines = resize(line, factor=factor)

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
    batch_size: int = 128,
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
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
        epoch: int | None = None,
        best_loss: float = float("inf")
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4*2)

    criterion = nn.BCEWithLogitsLoss()

    for _, train_scrolls, train_lines in iterator:
        train_data = LineSegmentationDataset(train_scrolls, train_lines)
        train_loss, val_loss = train_epoch(
            model,
            train_data,
            val_data,
            optimizer,
            criterion
        )
        with open(experiment_folder / "loss.txt", "a") as f:
            f.write(f"{train_loss:.4f},{val_loss:.4f}\n")

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        if verbose > 0:
            print(f"Losss {train_loss:.4f}, {val_loss:.4f}")

        if val_loss < best_loss:
            model.save(experiment_folder)
            best_loss = val_loss

    return model, optimizer, best_loss

if __name__ == "__main__":
    experiment_folder = create_experiment_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"truns/{timestamp}_run")

    model, optimizer, best_loss =  train_level(pool = {0}, experiment_folder=experiment_folder)
    print("Starting noise trainig")
    for epoch in range(200):
        _,_,best_loss = train_level(model=model, pool = {i for i in range(5)}, optimizer=optimizer, experiment_folder=experiment_folder, best_loss=best_loss,epoch=epoch)

