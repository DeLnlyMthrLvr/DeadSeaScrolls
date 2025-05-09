from pathlib import Path
import numpy as np
from datetime import datetime
import sys

sys.path.append("..")
sys.path.append(str(Path(__file__).parent.parent))

import torch.nn as nn
import torch
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
import tqdm
from noise_designer import load_batches
from unet import LineSegmenter, LineSegmentationDataset
import random

def create_experiment_folder(name: str = "run") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(__file__).parent  / "runs" / f"{timestamp}_{name}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
verbose: int = 2


def train_epoch(
    model: LineSegmenter,
    train_data: LineSegmentationDataset,
    validation_data: LineSegmentationDataset,
    optimizer: Optimizer,
    criterion: nn.Module,
    batch_size: int = 256,
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
        model: LineSegmenter | None = None,
        optimizer: Optimizer | None = None,
        experiment_folder: Path | None = None,
        experiment_name: str | None = "wide_unet_largek_fixed",
        best_loss: float = float("inf")
    ):

    level = random.choice(list(pool))
    print(f"noise_{level}")
    iterator = load_batches(level=level)
    _, val_scrolls, val_lines = next(iterator)

    val_scrolls = val_scrolls[:1000]
    val_lines = val_lines[:1000]

    val_data = LineSegmentationDataset(val_scrolls, val_lines)

    if experiment_folder is None and (experiment_name is not None):
        experiment_folder = create_experiment_folder(experiment_name)

    if model is None:
        model = LineSegmenter()
        model = model.to(device)

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3 * 1e-4)

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

        if val_loss < best_loss:
            model.save(experiment_folder)
            best_loss = val_loss

        with open(experiment_folder / "loss.txt", "a") as f:
            f.write(f"{train_loss:.4f},{val_loss:.4f}\n")
        if verbose > 0:
            print(f"Loss {train_loss:.4f}, {val_loss:.4f}")

    return model, optimizer, experiment_folder, best_loss

if __name__ == "__main__":
    model, optimizer, experiment_folder, best_loss =  train_level(pool = {0})
    for _ in range(5_000):
        _, _, _, best_loss = train_level(model=model, pool = {i for i in range(5)}, optimizer=optimizer, experiment_folder=experiment_folder, best_loss=best_loss)

