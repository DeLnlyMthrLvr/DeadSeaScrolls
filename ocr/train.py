import os
import sys
import shutil
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import ocr_model
import data_loader
import inference
from tokenizer import Tokenizer
import alphabet


def ocr_collate_fn(batch, pad_token_id):
    images, targets = zip(*batch)
    images = torch.stack(images)
    lengths = [len(t) for t in targets]
    max_len = max(lengths)
    padded_targets = torch.full((len(targets), max_len), pad_token_id, dtype=torch.long)
    for i, t in enumerate(targets):
        padded_targets[i, :len(t)] = t
    return images, padded_targets


def evaluate_loss(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            tgt_input = targets[:, :-1]
            tgt_output = targets[:, 1:]
            logits = model(images, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item() / len(images)
            n_batches += 1
    return total_loss / n_batches


def save_scripts(destination):
    current_dir = os.path.dirname(__file__)
    for script in ["train.py", "ocr_model.py", "data_loader.py"]:
        src = os.path.join(current_dir, script)
        shutil.copy(src, os.path.join(destination, script))


def log_metrics(save_dir, train_loss, val_loss, val_acc, acc_tf, test_acc, test_acc_tf, ngram_acc, ngram_tf_acc, lr):
    log_file = os.path.join(save_dir, "metrics_log.txt")
    with open(log_file, "a") as f:
        f.write(
            f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
            f"Validation Accuracy: {val_acc * 100:.2f}%, Accuracy (Teacher Forced): {acc_tf:.2f}%, "
            f"Learning Rate: {lr:.8f}\n"
            f"Test Accuracy: {test_acc * 100:.2f}%, Test Accuracy (Teacher Forced): {test_acc_tf:.2f}\n"
            f"N-Gram Accuracy: {ngram_acc * 100:.2f}%, N-Gram Accuracy (Teacher Forced): {ngram_tf_acc:.2f}\n\n"
        )


class OCRTrainer:
    def __init__(self):
        self.patch_size = 16
        self.embedding_dim = 192
        self.encoder_layers = 12
        self.decoder_layers = 6
        self.num_heads = 3
        self.vocab_size = 30
        self.mlp_ratio = 4
        self.dropout = 0.1
        self.batch_size = 128
        self.cross_attention_scale = 1.5
        self.image_size = (32, 416)
        self.device = torch.device("cuda")
        self.tokenizer = Tokenizer(alphabet.char_token)

        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.01)
        self.scheduler = self._get_scheduler()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_dir = os.path.join("/scratch/s3799042/weights/OCR/", timestamp)
        os.makedirs(self.save_dir, exist_ok=True)
        save_scripts(self.save_dir)

        self._load_datasets()

    def _build_model(self):
        vit = ocr_model.ViT(
            self.image_size[1], self.image_size[0], self.patch_size,
            self.embedding_dim, self.num_heads, self.encoder_layers,
            self.vocab_size, self.mlp_ratio, self.dropout
        )
        model = ocr_model.OCR(
            vit, self.embedding_dim, self.num_heads, self.decoder_layers,
            self.vocab_size, cross_attention_scale=self.cross_attention_scale
        )
        model = nn.DataParallel(model).to(self.device)
        #model.load_state_dict(torch.load("/scratch/s3799042/weights/OCR/2025-05-08_16-03-01/model_weights_70_accuracy.pth"))
        return model

    def _get_scheduler(self):
        num_steps = 15000
        warmup_steps = int(0.1 * num_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(0.0, float(num_steps - step) / float(max(1, num_steps - warmup_steps)))

        return LambdaLR(self.optimizer, lr_lambda)

    def _load_datasets(self):
        current_dir = os.path.dirname(__file__)
        data_base = os.path.join(current_dir, '..', 'data')
        test_base = os.path.join(data_base, 'sample-test-2025', 'Lines')

        self.train_loader = DataLoader(
            data_loader.BibleDatasetIterable(self.tokenizer, self.image_size),
            batch_size=self.batch_size, shuffle=False,
            collate_fn=lambda b: ocr_collate_fn(b, self.tokenizer.pad_token_id)
        )
        self.val_loader = DataLoader(
            data_loader.ScrollLineDataset(os.path.join(data_base, 'test_images_ocr', 'tokens.parquet'),
                                                     os.path.join(data_base, 'test_images_ocr'), self.tokenizer),
            batch_size=self.batch_size, shuffle=False,
            collate_fn=lambda b: ocr_collate_fn(b, self.tokenizer.pad_token_id)
        )
        self.ngram_loader = DataLoader(
            data_loader.ScrollLineDataset(os.path.join(data_base, 'n_gram_images', 'tokens.parquet'),
                                                     os.path.join(data_base, 'n_gram_images'), self.tokenizer),
            batch_size=self.batch_size, shuffle=False,
            collate_fn=lambda b: ocr_collate_fn(b, self.tokenizer.pad_token_id)
        )
        self.test_loader = DataLoader(
            data_loader.ScrollLineDatasetWithPadding(os.path.join(test_base, 'tokens.parquet'),
                                                     test_base, self.tokenizer, self.image_size),
            batch_size=self.batch_size, shuffle=False,
            collate_fn=lambda b: ocr_collate_fn(b, self.tokenizer.pad_token_id)
        )

    def train(self):
        self.model.train()
        average_loss = 0
        count = 0
        evaluate = False
        for images, targets in self.train_loader:
            if evaluate or (count > 0 and count % 500== 0):
                self._evaluate_and_log(average_loss / 100)
                average_loss = 0

            images, targets = images.to(self.device), targets.to(self.device)
            tgt_input, tgt_output = targets[:, :-1], targets[:, 1:]
            logits = self.model(images, tgt_input)
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            average_loss += loss.item() / len(images)
            count += 1

        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "model_weights.pth"))

    def _evaluate_and_log(self, train_loss):
        self.model.eval()

        test_acc = inference.evaluate_accuracy(self.model, self.test_loader, self.tokenizer, self.device)
        test_token_acc = inference.token_accuracy(self.model, self.test_loader, self.tokenizer, self.device)


        val_loss = evaluate_loss(self.model, self.criterion, self.val_loader, self.device)
        val_acc = inference.evaluate_accuracy(self.model, self.val_loader, self.tokenizer, self.device)
        val_token_acc = inference.token_accuracy(self.model, self.val_loader, self.tokenizer, self.device)


        ngram_acc = inference.evaluate_accuracy(self.model, self.ngram_loader, self.tokenizer, self.device)
        ngram_token_acc = inference.token_accuracy(self.model, self.ngram_loader, self.tokenizer, self.device)

        current_lr = self.optimizer.param_groups[0]['lr']

        log_metrics(self.save_dir, train_loss, val_loss, val_acc, val_token_acc, test_acc, test_token_acc,
                    ngram_acc, ngram_token_acc, current_lr)
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "model_weights.pth"))
        self.model.train()


if __name__ == "__main__":
    trainer = OCRTrainer()
    trainer.train()