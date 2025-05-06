import torch
import ocr_model
from PIL import Image
import torchvision.transforms as transforms
import os
import pathlib
import data_loader
import sys
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
import train
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import alphabet

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
device = torch.device('cuda')
model.to(device)
current_dir = os.path.dirname(__file__)
image_dir = os.path.join(current_dir, '..', 'data', 'test_images_ocr')
parquet_path = os.path.join(image_dir, "tokens.parquet")
weights_path = os.path.join(current_dir, '..', 'model_weights.pth')
tokenizer = Tokenizer(alphabet.char_token)
dataset = data_loader.ScrollLineDataset(parquet_path, image_dir, tokenizer)
dataloader = DataLoader(dataset, batch_size = 32, shuffle = False, collate_fn=lambda b: train.ocr_collate_fn(b, tokenizer.pad_token_id))

model.load_state_dict(torch.load(weights_path))  
model.eval()
correct = 0
total = 0
max_length = 150
with torch.no_grad():
    for images, target_sequences in tqdm(dataloader, desc="Evaluating", ncols=100, leave=True):

        images = images.to(device)
        target_sequences = target_sequences.to(device)
        # Step 1: Generate predictions using the `generate` function
        predictions = []
        for batch_idx in range(images.size(0)):
            # For each image in the batch, generate a sequence
            image = images[batch_idx].unsqueeze(0)  # Add batch dimension
            generated_tokens = model.generate(image, max_length, tokenizer.bos_token_id, tokenizer.eos_token_id)
            predictions.append(generated_tokens)

        # Step 2: Compute accuracy
        # Pad the target sequences and generated tokens to the same length if necessary
        for i in range(images.size(0)):
            target = target_sequences[i].cpu().numpy()
            prediction = predictions[i]

            # Calculate how many tokens match (ignoring the eos_token)
            correct += sum([1 if t == p else 0 for t, p in zip(target, prediction) if p != tokenizer.eos_token_id])
            total += sum([1 for p in prediction if p != tokenizer.eos_token_id])

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}%")