import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import ocr_model
from tokenizer import Tokenizer
import data_loader
import sys

from torch.utils.data import DataLoader
# Add parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import alphabet

# batching variable-length sequences
def ocr_collate_fn(batch, pad_token_id):
    images, targets = zip(*batch)

    # Stack images into a tensor [batch_size, 1, H, W]
    images = torch.stack(images)

    # Pad target sequences
    lengths = [len(t) for t in targets]
    max_len = max(lengths)
    padded_targets = torch.full((len(targets), max_len), pad_token_id, dtype=torch.long)

    for i, t in enumerate(targets):
        padded_targets[i, :len(t)] = t

    return images, padded_targets

def train():
  patch_size = 16
  embedding_dimension = 64 #768 base
  depth = 12 #12 base
  num_heads = 4 #12 base
  vocab_size = 30
  mlp_ratio = 0.4
  dropout = 0.4
  num_encoder_blocks = 4
  num_decoder_blocks = 4
  
  image_size = (32, 416)
  
  ViT = ocr_model.ViT(image_size[1], image_size[0], patch_size, 
                      embedding_dimension, num_heads, depth, vocab_size, mlp_ratio,
                      dropout, num_encoder_blocks)

  model = ocr_model.OCR(ViT, embedding_dimension, num_heads, depth, vocab_size)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.0001)
  device = torch.device('cuda')
  model.to(device)
  tokenizer = Tokenizer(alphabet.char_token)
  bos_token_id = tokenizer.bos_token_id
  eos_token_id = tokenizer.eos_token_id
  num_epochs = 3
  

  
  dataset = data_loader.ScrollLineDatasetIterable(tokenizer, image_size)
  dataloader = DataLoader(dataset, batch_size = 32, shuffle = False, collate_fn=lambda b: ocr_collate_fn(b, tokenizer.pad_token_id))
  
  train_ocr(model, dataloader, optimizer, criterion, device, bos_token_id, eos_token_id, num_epochs)
  
def train_ocr(model, dataloader, optimizer, criterion, device, bos_token_id, eos_token_id, num_epochs):
    model.train()

    #for epoch in range(num_epochs):
    total_loss = 0
    count = 0
    print_after_n_batches = 13
    save_after_n_batches = 10_000
    for images, target_sequences in dataloader:
        if count > 0 and count % print_after_n_batches == 0:
            print(total_loss)
            total_loss = 0
        if count > 0 and count % save_after_n_batches == 0:
            print("save")
            torch.save(model.state_dict(), "model_weights.pth")

        images = images.to(device)
        target_sequences = target_sequences.to(device)

        optimizer.zero_grad()

        # Shift inputs for teacher forcing
        tgt_input = target_sequences[:, :-1]     # [BOS, t, e, s, ...]
        tgt_output = target_sequences[:, 1:]     # [t, e, s, ..., EOS]

        logits = model(images, tgt_input)  # (batch_size, seq_len, vocab_size)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        #print(loss.item())
        count += 1

    #print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), "model_weights.pth")


if __name__ == "__main__":
    train()
