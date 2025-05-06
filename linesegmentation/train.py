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

from synthetic import DataGenerator, SynthSettings
from unet import UNet

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
        scrolls = scrolls / 255
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


def _train(
        generator: DataGenerator,
        gen_batch_size: int = 2_500,
        gen_batches: int = 50,
        redo_batch: int = 1,
        model: UNet | None = None,
        optimizer: Optimizer | None = None,
        experiment_folder: Path | None = None,
        experiment_name: str | None = "unet"
    ):
    _, _, val_scrolls, val_lines = generator.generate_passages_scrolls(N=200)
    val_data = LineSegmentationDataset(val_scrolls, val_lines)

    if experiment_folder is None and (experiment_name is not None):
        experiment_folder = create_experiment_folder(experiment_name)

    if model is None:
        model = UNet()
        model = model.to(device)

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(gen_batches):
        _, _, scrolls, lines = generator.generate_passages_scrolls(gen_batch_size)

        train_data = LineSegmentationDataset(scrolls, lines)

        for _ in range(redo_batch):
            train_loss, val_losss = train_epoch(
                model,
                train_data,
                val_data,
                optimizer,
                criterion
            )

            model.save(experiment_folder)
            with open(experiment_folder / "loss.txt", "a") as f:
                f.write(f"{train_loss:.4f},{val_losss:.4f}")

            if verbose > 0:
                print(f"Losss {train_loss:.4f}, {val_losss:.4f}")

        if verbose > 0:
            print("Noisel")

    return model, optimizer

def train_noiseless(
    gen_batch_size: int = 2_500,
    gen_batches: int = 50,
    redo_batch: int = 1,
    model: UNet | None = None,
    optimizer: Optimizer | None = None,
    experiment_folder: Path | None = None,
    experiment_name: str | None = "unet"
):

    gen_settings = SynthSettings(downscale_factor=0.35)
    generator = DataGenerator(settings=gen_settings)

    return _train(gen_batch_size, gen_batches, redo_batch, model, optimizer, experiment_folder, experiment_name, generator)


def train_noiseless(
    gen_batch_size: int = 2_500,
    gen_batches: int = 50,
    redo_batch: int = 1,
    model: UNet | None = None,
    optimizer: Optimizer | None = None,
    experiment_folder: Path | None = None,
    experiment_name: str | None = "unet"
):

    gen_settings = SynthSettings(downscale_factor=0.35)
    generator = DataGenerator(settings=gen_settings)

    return _train(gen_batch_size, gen_batches, redo_batch, model, optimizer, experiment_folder, experiment_name, generator)



if __name__ == "__main__":
    train_noiseless(redo_batch=2, gen_batch_size=500, gen_batches=10)
