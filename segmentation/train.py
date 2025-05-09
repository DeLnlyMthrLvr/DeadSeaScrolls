from pathlib import Path
import pickle
import cv2
import numpy as np
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
import tqdm
from unet import UNet
import random

from alphabet import char_to_token
n_tokens = len(char_to_token) - 1

def create_experiment_folder(name: str = "seg") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(__file__).parent  / "runs" / f"{timestamp}_{name}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
verbose: int = 2

def load_batches(level: int):
    level_path = Path(__file__).parent.parent / "data" / "scrolls_seg" / f"level_{level}"
    chunks = sorted(
        level_path.glob("chunk_*.npz"),
        key=lambda p: int(p.stem.split("_")[1])
    )

    for chunk_path in chunks:
        base = chunk_path.parent
        chunk = int(chunk_path.stem.split("_")[1])

        with open(base / f"chunk_{chunk}.pickle", "rb") as f:
            tokens: list[list[str]] = pickle.load(f) # not tokens but characters (batch, n_lines, sequence)

        data = np.load(chunk_path)
        scrolls: np.ndarray = data["scrolls"] # (batch, h, w)
        line_masks: np.ndarray = data["line_masks"] # (batch, h, w)
        segmentation_masks: np.ndarray = data["segmentation"] # (batch, 27, h, w)
        yield tokens, scrolls, line_masks, segmentation_masks

def extract_lines_segs_cc(
        img: np.ndarray,
        seg: np.ndarray,
        binary_mask: np.ndarray,
        min_area: int = 500,
        inflate: int = 6
    ) -> list[np.ndarray]:

    mask8 = (binary_mask > 0).astype(np.uint8) * 255

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask8, connectivity=8)

    h, w = binary_mask.shape

    lines = []
    segs = []
    for lab in range(1, n_labels):
        x, y, bw, bh, area = stats[lab]
        if area < min_area:
            continue

        x0 = max(x - inflate, 0)
        y0 = max(y - inflate, 0)
        x1 = min(x + bw + inflate, w)
        y1 = min(y + bh + inflate, h)

        crop = img[y0:y1, x0:x1].copy()
        lines.append(crop)
        cs = seg[:, y0:y1, x0:x1].copy()
        segs.append(cs)

    return lines, segs

class SegmentationDataset(Dataset):
    def __init__(self, scrolls: np.ndarray, segs: np.ndarray, lines: np.ndarray):

        self.line_images = [] # Each image will be (h, w)
        self.line_segmentations = [] # each mask will be (27, h, w)

        for scroll, seg, line in zip(scrolls, segs, lines, strict=True):
            li, ls = extract_lines_segs_cc(scroll, seg, line)
            self.line_images.extend(li)
            self.line_segmentations.extend(ls)

        self.line_images = [1 - (torch.from_numpy(img).float() / 255) for img in self.line_images]
        self.line_segmentations = [torch.from_numpy(seg).float() for seg in self.line_segmentations]

        max_h = max(img.shape[-2] for img in self.line_images)
        max_w = max(img.shape[-1] for img in self.line_images)

        def symmetric_pad(tensor, target_h, target_w):
            h, w = tensor.shape[-2], tensor.shape[-1]
            pad_h = target_h - h
            pad_w = target_w - w
            pad = [
                pad_w // 2, pad_w - pad_w // 2,
                pad_h // 2, pad_h - pad_h // 2
            ]
            return F.pad(tensor, pad, value=0)

        self.line_images = [symmetric_pad(img, max_h, max_w) for img in self.line_images]
        self.line_segmentations = [symmetric_pad(seg, max_h, max_w) for seg in self.line_segmentations]

        self.line_images = torch.stack(self.line_images)
        self.line_segmentations = torch.stack(self.line_segmentations)

    def __len__(self):
        return len(self.line_images)

    def __getitem__(self, index):
        return self.line_images[index], self.line_segmentations[index]

def train_epoch(
    model: UNet,
    train_data: SegmentationDataset,
    validation_data: SegmentationDataset,
    optimizer: Optimizer,
    criterion: nn.Module,
    batch_size: int = 64,
):

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size)
    val_loader = DataLoader(dataset=validation_data, batch_size=batch_size)

    #Training
    model.train()
    track_loss = []
    for batch_scrolls, batch_seg in tqdm.tqdm(train_loader, total=len(train_loader), disable=verbose < 2):

        batch_scrolls, batch_seg  = batch_scrolls.to(device), batch_seg.to(device)
        seg_hat = model(batch_scrolls)
        loss = criterion(seg_hat, batch_seg)
        track_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #Validation
    model.eval()
    val_track_loss = []
    with torch.no_grad():
        for batch_scrolls, batch_seg in val_loader:
            batch_scrolls, batch_seg = batch_scrolls.to(device), batch_seg.to(device)
            seg_hat = model(batch_scrolls)
            loss = criterion(seg_hat, batch_seg)
            val_track_loss.append(loss.item())

    return np.mean(track_loss), np.mean(val_track_loss)


def train_level(
        pool: set,
        model: UNet | None = None,
        optimizer: Optimizer | None = None,
        experiment_folder: Path | None = None,
        experiment_name: str | None = "unetseg",
        best_loss: float = float("inf")
    ):

    level = random.choice(list(pool))
    print(f"noise_{level}")
    iterator = load_batches(level=level)
    _, val_scrolls, val_lines, val_seg = next(iterator)

    val_data = SegmentationDataset(val_scrolls, val_seg, val_lines)

    if experiment_folder is None and (experiment_name is not None):
        experiment_folder = create_experiment_folder(experiment_name)

    if model is None:
        model = UNet(n_tokens)
        model = model.to(device)

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3 * 1e-4)

    criterion = nn.BCEWithLogitsLoss()

    for _, train_scrolls, train_lines, train_seg in iterator:
        train_data = SegmentationDataset(train_scrolls, train_seg, train_lines)
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

