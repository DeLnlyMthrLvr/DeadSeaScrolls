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
import torch.nn as nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import alphabet


def evaluate_accuracy(model, dataloader, tokenizer, device):
    model.eval()
    correct = 0
    total = 0
    max_length = 150
    with torch.no_grad():
        index = 0
        for images, target_sequences in tqdm(dataloader, desc="Evaluating", ncols=100, leave=True):

            images = images.to(device)
            target_sequences = target_sequences.to(device)
            predictions = []
            for batch_idx in range(images.size(0)):
                # For each image in the batch, generate a sequence
                image = images[batch_idx].unsqueeze(0)  # Add batch dimension
                generated_tokens = model.module.generate(image, max_length, tokenizer.bos_token_id, tokenizer.eos_token_id)
                predictions.append(generated_tokens)
            

            worst_images = []

            for i in range(images.size(0)):
                correct_per_image = 0 
                target = target_sequences[i].cpu().numpy()
                prediction = predictions[i]

                correct_per_image += sum([1 if t == p else 0 for t, p in zip(target, prediction) if p != tokenizer.eos_token_id])
                if correct_per_image <= 3:
                    worst_images.append([index, target, prediction])
                # Calculate how many tokens match (ignoring the eos_token)
                correct += sum([1 if t == p else 0 for t, p in zip(target, prediction) if p != tokenizer.eos_token_id])
                total += sum([1 for p in prediction if p != tokenizer.eos_token_id])
                index += 1

        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

def token_accuracy(model, dataloader, tokenizer, device):
    model.eval()
    total_correct = 0
    total_tokens = 0
    pad_token_id = tokenizer.pad_token_id

    with torch.no_grad():
        for images, target_sequences in dataloader:
            images = images.to(device)
            target_sequences = target_sequences.to(device)

            # Shift for teacher forcing
            tgt_input = target_sequences[:, :-1]
            tgt_output = target_sequences[:, 1:]

            logits = model(images, tgt_input)  # (batch_size, seq_len, vocab_size)
            predictions = logits.argmax(dim=-1)  # (batch_size, seq_len)

            # Ignore padding tokens in accuracy calculation
            mask = tgt_output != pad_token_id
            correct = (predictions == tgt_output) & mask

            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

    accuracy = 100.0 * total_correct / total_tokens if total_tokens > 0 else 0.0
    print(f"Token-Level Accuracy (Teacher Forcing): {accuracy:.2f}%")
    return accuracy
    
    
def inference():
    patch_size = 16
    embedding_dimension = 384 #768 base
    depth = 12 #12 base
    num_heads = 6 #12 base
    vocab_size = 30
    mlp_ratio = 4
    dropout = 0.1
    batch_size = 32
    
    image_size = (32, 416)

    ViT = ocr_model.ViT(image_size[1], image_size[0], patch_size, 
                      embedding_dimension, num_heads, depth, vocab_size, mlp_ratio,
                      dropout)

    model = ocr_model.OCR(ViT, embedding_dimension, num_heads, depth, vocab_size)
    device = torch.device('cuda')
    model.to(device)
    current_dir = os.path.dirname(__file__)
    image_dir = os.path.join(current_dir, '..', 'data', 'test_images_ocr')
    parquet_path = os.path.join(image_dir, "tokens.parquet")
    weights_path = os.path.join(current_dir, '..', 'model_weights.pth')
    tokenizer = Tokenizer(alphabet.char_token)

    model.load_state_dict(torch.load("/scratch/s3799042/weights/OCR/2025-05-05_18-23-54/model_weights.pth"))  
    dataset = data_loader.ScrollLineDataset(parquet_path, image_dir, tokenizer)
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = False, collate_fn=lambda b: train.ocr_collate_fn(b, tokenizer.pad_token_id))
    evaluate_accuracy(model, dataloader, tokenizer, device)

if __name__ == "__main__":
    inference()