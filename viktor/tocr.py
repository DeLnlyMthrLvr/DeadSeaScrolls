from __future__ import annotations
from datetime import datetime

"""Train a ViT‑GRU Seq‑to‑Seq Hebrew OCR model with curriculum noise levels.

Key features compared to the original script
-------------------------------------------
1. **Chunk‑aware streaming** – We iterate over the dataset **one chunk at a time**. The
   *first* chunk of every noise level is *always* treated as **validation** data; the
   remaining chunks are used for training. Nothing is ever fully pre‑loaded into RAM.
2. **Per‑chunk metrics** – After *every* chunk we print both the current **training**
   loss (if it was a training chunk) **and** the latest **validation** loss for that
   noise level, giving near‑real‑time feedback throughout an epoch.
3. **Multi‑level training** – Once the warm‑up phase on level 0 is done, the script
   cycles through *all* requested noise levels in every epoch, seamlessly switching
   between them chunk‑by‑chunk.
"""

import argparse
import random
import pickle
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from transformers import ViTModel, ViTImageProcessor
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from alphabet import char_to_token, char_token  # noqa: E402 – local import

n_tokens = len(char_token) - 1
PAD_TOKEN, SOS_TOKEN, EOS_TOKEN = n_tokens, n_tokens + 1, n_tokens + 2
VOCAB_SIZE = n_tokens + 3

# ---------------------------------------------------------------------------
#  Dataset helpers
# ---------------------------------------------------------------------------

def encode_tokens(str_tokens: List[str], *, max_len: int = 256) -> List[int]:
    """Convert a list of character strings to token ids with <SOS>/<EOS>."""
    ids = [char_to_token[t] for t in str_tokens]
    ids = [SOS_TOKEN] + ids[: max_len - 2] + [EOS_TOKEN]
    return ids


def load_batches(level: int) -> Iterable[Tuple[int, List[List[str]], np.ndarray, np.ndarray]]:
    """Yield **single pre‑batched** arrays for one *chunk* of a given noise level.

    Yields
    ------
    chunk_idx : int
        The integer index of the chunk (0‑based).  *chunk 0 is validation by design.*
    tokens    : list[list[str]]
        A batch of token sequences per text line (Python list of lists).
    scrolls   : np.ndarray
        Pre‑corrupted scroll images (uint8 H×W×3 or H×W).
    line_masks: np.ndarray
        Binary masks the same height/width as *scrolls* where each connected component
        roughly corresponds to a text line.
    """

    level_path = Path(__file__).parent.parent / "data" / "scrolls" / f"level_{level}"
    chunks = sorted(level_path.glob("chunk_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    for chunk_path in chunks:
        base = chunk_path.parent
        chunk = int(chunk_path.stem.split("_")[1])
        with open(base / f"chunk_{chunk}.pickle", "rb") as f:
            tokens = pickle.load(f)
        data = np.load(chunk_path)
        scrolls, line_masks = data["scrolls"], data["line_masks"]
        yield chunk, tokens, scrolls, line_masks


def extract_lines_cc(img: np.ndarray, binary_mask: np.ndarray, *,
                     min_area: int = 500, inflate: int = 6) -> List[np.ndarray]:
    """Extract individual line crops using connected‑components on `binary_mask`."""

    mask8 = (binary_mask > 0).astype(np.uint8) * 255
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask8, connectivity=8)
    h, w = binary_mask.shape
    lines = []
    for lab in range(1, n_labels):
        x, y, bw, bh, area = stats[lab]
        if area < min_area:
            continue
        x0, y0 = max(x - inflate, 0), max(y - inflate, 0)
        x1, y1 = min(x + bw + inflate, w), min(y + bh + inflate, h)
        lines.append(img[y0:y1, x0:x1].copy())
    return lines


class ScrollLineChunkDataset(IterableDataset):

    def __init__(self, tokens_batch, scrolls_batch, masks_batch, processor: ViTImageProcessor,
                 max_target_len: int = 150):
        super().__init__()
        self.tokens_batch = tokens_batch
        self.scrolls_batch = scrolls_batch
        self.masks_batch = masks_batch
        self.processor = processor
        self.max_target_len = max_target_len

    def __iter__(self):
        for sample_idx, (scroll, lm) in enumerate(zip(self.scrolls_batch, self.masks_batch)):
            line_imgs = extract_lines_cc(scroll, lm)
            line_tokens = self.tokens_batch[sample_idx]
            for img, tok in zip(line_imgs, line_tokens):
                # RGB convert
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if img.ndim == 2 else cv2.COLOR_BGR2RGB)
                pv = self.processor(images=img_rgb, return_tensors="pt")["pixel_values"].squeeze(0)
                tgt = torch.tensor(encode_tokens(tok, max_len=self.max_target_len), dtype=torch.long)
                yield pv, tgt


def collate_fn(batch):
    pvs, tgt = zip(*batch)
    pvs = torch.stack(pvs)  # (B,3,224,224)
    max_len = max(len(t) for t in tgt)
    input_ids = torch.full((len(tgt), max_len), PAD_TOKEN, dtype=torch.long)
    labels = torch.full_like(input_ids, PAD_TOKEN)
    for i, seq in enumerate(tgt):
        seq_tensor = seq.clone().detach() if isinstance(seq, torch.Tensor) else torch.tensor(seq, dtype=torch.long)
        input_ids[i, :len(seq)] = seq_tensor
        labels[i, :len(seq) - 1] = seq_tensor[1:]
    return pvs, input_ids, labels

class ViTGRUSeq2Seq(nn.Module):
    def __init__(self, hidden_size: int = 768, num_layers: int = 1):
        super().__init__()
        self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        print(sum([p.numel() for p in self.encoder.parameters()]) / 1_000_000)
        sys.exit(1)

        self.dec_embed = nn.Embedding(VOCAB_SIZE, hidden_size, padding_idx=PAD_TOKEN)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers,
                          batch_first=True)
        self.fc_out = nn.Linear(hidden_size, VOCAB_SIZE)

    def forward(self, pixel_values, decoder_input_ids):
        enc_out = self.encoder(pixel_values=pixel_values).last_hidden_state.mean(dim=1)  # (B,H)
        h0 = enc_out.unsqueeze(0)  # (1,B,H)
        dec_emb = self.dec_embed(decoder_input_ids)
        dec_out, _ = self.gru(dec_emb, h0)
        return self.fc_out(dec_out)


def run_pass(model, dataloader, criterion, optimizer, device, *, train: bool):
    phase = "Train" if train else "Val"
    model.train() if train else model.eval()
    running, count = 0.0, 0
    for pv, inp, lbl in tqdm(dataloader, desc=phase, leave=False):
        pv, inp, lbl = (t.to(device) for t in (pv, inp, lbl))
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(pv, inp)
            loss = criterion(logits.view(-1, VOCAB_SIZE), lbl.view(-1))
            if train:
                loss.backward(); optimizer.step()
        running += loss.item() * pv.size(0)
        count += pv.size(0)
    return running / max(count, 1)

def parse_cli():
    ap = argparse.ArgumentParser("Train ViT+GRU Hebrew OCR with curriculum noise levels")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--warmup_epochs", type=int, default=2,
                    help="Train only on level 0 for this many initial epochs")
    ap.add_argument("--train_levels", type=str, default="0,1,2,3,4",
                    help="Comma‑separated noise levels used *after* warm‑up phase")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--output", type=str, default="vit_gru_hebrew.pth")
    ap.add_argument("--freeze_encoder", action="store_true",
                    help="Do not update ViT encoder parameters if set")
    return ap.parse_args()

def main():
    args = parse_cli()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = Path(__file__).parent / "runs" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_vitgru"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "losses.csv"

    # Write CSV header if file didn't exist before
    if not csv_path.exists():
        csv_path.write_text("epoch,level,mean_train_loss,val_loss\n")

    # Processor & model
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTGRUSeq2Seq().to(device)

    # Optionally freeze encoder
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    # Optimiser over *trainable* parameters
    trainable_params = (p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    train_levels = [0] if args.warmup_epochs > 0 else [int(l) for l in args.train_levels.split(",")]

    val_loaders = {}
    for lvl in {int(l) for l in args.train_levels.split(",")}:  # unique levels
        chunk0, tok, scr, msk = next(load_batches(lvl))
        assert chunk0 == 0, "Expected first chunk index to be 0!"
        ds = ScrollLineChunkDataset(tok, scr, msk, processor)
        val_loaders[lvl] = DataLoader(ds, batch_size=args.batch_size,
                                      collate_fn=collate_fn)


    for epoch in range(args.epochs):

        if epoch == args.warmup_epochs:
            train_levels = [int(l) for l in args.train_levels.split(",")]

        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        random.shuffle(train_levels)  # level order randomisation

        for lvl in train_levels:
            # ------------------ Training chunks (1 … N) ------------------
            train_losses = []
            for chunk_idx, tok, scr, msk in load_batches(lvl):
                if chunk_idx == 0:
                    continue  # already validated
                ds = ScrollLineChunkDataset(tok, scr, msk, processor)
                loader = DataLoader(ds, batch_size=args.batch_size,
                                    collate_fn=collate_fn)
                tr_loss = run_pass(model, loader, criterion, optimizer, device, train=True)
                train_losses.append(tr_loss)
                print(f"Level {lvl}  Chunk {chunk_idx:<2} (TRAIN) loss={tr_loss:.4f}")

            mean_train_loss = float(np.mean(train_losses)) if train_losses else float('nan')

            # ----------------------- Validation -------------------------
            val_loss = run_pass(model, val_loaders[lvl], criterion, optimizer, device, train=False)

            # --------------- Checkpoint & CSV logging -------------------
            ckpt_path = run_dir / f"model.pth"
            torch.save(model.state_dict(), ckpt_path)

            with csv_path.open("a", encoding="utf-8") as f:
                f.write(f"{epoch + 1},{lvl},{mean_train_loss:.5f},{val_loss:.5f}\n")



if __name__ == "__main__":
    main()