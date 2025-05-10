import os
import torch
import torch.nn as nn
import torch.optim as optim
import ocr_refactored.ocr_model as ocr_model
from ocr_refactored.tokenizer import Tokenizer
import data_loader
import sys
import ocr_refactored.inference as inference
from datetime import datetime
import shutil
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.data import DataLoader
# Add parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import alphabet

# batching variable-length sequences
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
    average_loss = 0
    n_batches = 0
    with torch.no_grad():
        for images, target_sequences in dataloader:
            n_batches +=1
            images = images.to(device)
            target_sequences = target_sequences.to(device)

            # Shift inputs for teacher forcing
            tgt_input = target_sequences[:, :-1]
            tgt_output = target_sequences[:, 1:]

            logits = model(images, tgt_input)  # (batch_size, seq_len, vocab_size)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            average_loss += loss.item()/len(images)

    print(f"average loss val: {average_loss/n_batches}")
    return average_loss/n_batches

def save_scripts(save_dir):
    train_script = os.path.abspath(__file__)
    shutil.copy(train_script, os.path.join(save_dir, "train.py"))

    model_script = os.path.join(os.path.dirname(__file__), "ocr_model.py")
    shutil.copy(model_script, os.path.join(save_dir, "ocr_model.py"))

    dataloader_script = os.path.join(os.path.dirname(__file__), "data_loader.py")
    shutil.copy(dataloader_script, os.path.join(save_dir, "data_loader.py"))

def log_metrics(save_dir, train_loss, validation_loss, validation_accuracy,
                 accuracy_teacher_forced, test_accuracy, test_accuracy_teacher_forced,
                  ngram_accuracy, ngram_token_accuracy, lr):
    log_file = os.path.join(save_dir, "metrics_log.txt")

    line = (
    f"Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}, "
    f"Validation Accuracy: {validation_accuracy * 100:.2f}%, "
    f"Accuracy (Teacher Forced): {accuracy_teacher_forced:.2f}%, "
    f"Learning Rate: {lr:.8f}\n"
    f"Test Accuracy: {test_accuracy * 100:.2f}%, "
    f"Test Accuracy (Teacher Forced): {test_accuracy_teacher_forced:.2f}\n"
    f"N-Gram Accuracy: {ngram_accuracy * 100:.2f}%, "
    f"N-Gram Accuracy (Teacher Forced): {ngram_token_accuracy:.2f}\n\n"
    )

    with open(log_file, "a") as f:
        f.write(line)


def train():
  patch_size = 16 
  embedding_dimension = 384 #768 base
  encoder_layers = 12 #12 base
  decoder_layers = 6
  num_heads = 6 #12 base
  vocab_size = 30
  mlp_ratio = 4
  dropout = 0.1
  batch_size = 128
  cross_attention_scale = 1.5
  
  image_size = (32, 416)
  
  ViT = ocr_model.ViT(image_size[1], image_size[0], patch_size, 
                      embedding_dimension, num_heads, encoder_layers, vocab_size, mlp_ratio,
                      dropout)

  model = ocr_model.OCR(ViT, embedding_dimension, num_heads, decoder_layers, vocab_size, cross_attention_scale=cross_attention_scale)
  model = nn.DataParallel(model)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
  device = torch.device('cuda')
  model.to(device)
  tokenizer = Tokenizer(alphabet.char_token)
  current_dir = os.path.dirname(__file__)
  image_dir_normal = os.path.join(current_dir, '..', 'data', 'test_images_ocr')
  image_dir_ngrams = os.path.join(current_dir, '..', 'data', 'n_gram_images')
  image_dir_noise = os.path.join(current_dir, '..', 'data', 'test_images_ocr')
  image_dir_test = os.path.join(current_dir, '..', 'data', 'sample-test-2025', 'Lines')
  parquet_normal_path = os.path.join(image_dir_normal, "tokens.parquet")
  parquet_ngrams_path = os.path.join(image_dir_ngrams, "tokens.parquet")
  parquet_noise_path = os.path.join(image_dir_normal, "tokens.parquet")
  parquet_test_path = os.path.join(image_dir_test, "tokens.parquet")

  weights_path = os.path.join(current_dir, '..', 'model_weights.pth')

  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  save_dir = os.path.join("/scratch/s3799042/weights/OCR/", timestamp)
  os.makedirs(save_dir, exist_ok=True)
  save_scripts(save_dir)

  num_training_steps = 15_000_000
  num_warmup_steps = int(0.02 * num_training_steps)

  def lr_lambda(current_step):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
    )

  scheduler = LambdaLR(optimizer, lr_lambda)
  #model.load_state_dict(torch.load("/scratch/s3799042/weights/OCR/2025-05-07_12-46-57/model_weights.pth"))  
  dataset_train = data_loader.ScrollLineDatasetIterable(tokenizer, image_size)
  dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = False, collate_fn=lambda b: ocr_collate_fn(b, tokenizer.pad_token_id))
  
  dataset_val = data_loader.ScrollLineDataset(parquet_normal_path, image_dir_normal, tokenizer)
  dataloader_val = DataLoader(dataset_val, batch_size = batch_size, shuffle = False, collate_fn=lambda b: ocr_collate_fn(b, tokenizer.pad_token_id))
  
  dataset_ngrams = data_loader.ScrollLineDataset(parquet_ngrams_path, image_dir_ngrams, tokenizer)
  dataloader_ngrams = DataLoader(dataset_ngrams, batch_size = batch_size, shuffle = False, collate_fn=lambda b: ocr_collate_fn(b, tokenizer.pad_token_id))

  dataset_test = data_loader.ScrollLineDatasetWithPadding(parquet_test_path, image_dir_test, tokenizer, image_size)
  dataloader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle = False, collate_fn=lambda b: ocr_collate_fn(b, tokenizer.pad_token_id))
  train_ocr(model, dataloader_train, dataloader_val, dataloader_test, dataloader_ngrams, optimizer, criterion, device, 
            tokenizer, save_dir, scheduler)
  
def train_ocr(model, dataloader_train, dataloader_val, dataloader_test, dataloader_ngrams,
              optimizer, criterion, device, tokenizer, save_dir, scheduler):
    model.train()
    average_loss = 0
    count = 0
    print_after_n_batches = 100
    save_after_n_batches = 500
    for images, target_sequences in dataloader_train:
        if count > 0 and count % save_after_n_batches == 0:
            current_lr = optimizer.param_groups[0]['lr']
            test_token_accuracy = inference.token_accuracy(model, dataloader_test, tokenizer, device)
            test_accuracy = inference.evaluate_accuracy(model, dataloader_test, tokenizer, device)
            
            ngram_token_accuracy = inference.token_accuracy(model, dataloader_ngrams, tokenizer, device)
            ngram_accuracy = inference.evaluate_accuracy(model, dataloader_ngrams, tokenizer, device)

            token_accuracy = inference.token_accuracy(model, dataloader_val, tokenizer, device)
            val_accuracy = inference.evaluate_accuracy(model, dataloader_val, tokenizer, device)

            
            
            
            val_loss = evaluate_loss(model, criterion, dataloader_val, device)
            log_metrics(save_dir, average_loss/print_after_n_batches, val_loss, val_accuracy, 
                        token_accuracy, test_accuracy, test_token_accuracy, ngram_accuracy, ngram_token_accuracy,
                         current_lr)
            torch.save(model.state_dict(), os.path.join(save_dir,"model_weights.pth"))
            model.train()
        if count > 0 and count % print_after_n_batches == 0:
            print(f"average loss train: {average_loss/print_after_n_batches}")
            average_loss = 0


        images = images.to(device)
        target_sequences = target_sequences.to(device)

        optimizer.zero_grad()

        # Shift inputs for teacher forcing
        tgt_input = target_sequences[:, :-1]   
        tgt_output = target_sequences[:, 1:]     

        logits = model(images, tgt_input)  # (batch_size, seq_len, vocab_size)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        average_loss += loss.item()/len(images)
        count += 1

    torch.save(model.state_dict(), "model_weights.pth")


if __name__ == "__main__":
    train()
