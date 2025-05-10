import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from PIL import Image
import os
import sys
from pathlib import Path
from tokenizer import Tokenizer
import random
import tqdm
from datetime import datetime
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.nn import DataParallel
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import alphabet
import noise_designer
import synthetic

# batching variable-length sequences

def collate_fn(batch, tokenizer):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])  # shape: (B, 3, H, W)
    
    labels = [item['labels'] for item in batch]
    # Pad labels to the longest in the batch
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    # Replace pad token with -100 to ignore in loss
    labels_padded[labels_padded == tokenizer.pad_token_id] = -100

    return {
        'pixel_values': pixel_values,
        'labels': labels_padded
    }

def visualize(pixel_values, labels, tokenizer: Tokenizer):
    for idx in range(pixel_values.shape[0]):
        img = to_pil_image(pixel_values[idx])
        label_ids = labels[idx]
        label_ids = label_ids[label_ids != -100]  # remove padding/ignore tokens
        label_text = tokenizer.decode(label_ids)
        plt.imshow(img)
        plt.title(f"Label: {label_text}, label_ids: {label_ids}")
        plt.axis("off")

        # Save the figure
        plt.savefig("output_example.png", bbox_inches="tight", dpi=300)
        print("created_image")

def train_epoch(model, train_data, val_data, 
                optimizer, batch_size, epoch):
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, collate_fn=lambda b: collate_fn(b, tokenizer))
    #eval_model(model, val_loader)
    model.train()
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, collate_fn=lambda b: collate_fn(b, tokenizer))
    total_train_loss = 0
    count = 1
    for batch in tqdm.tqdm(train_loader, total=len(train_loader)):
        #visualize(batch['pixel_values'], batch['labels'], tokenizer)
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss #list of scalar tensors
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_train_loss += loss.item()

        avg_train_loss = total_train_loss / count
        count+=1
    print(f"Epoch {epoch} | Training Loss: {avg_train_loss:.4f}")

    avg_val_loss = eval_model(model, val_loader)
    return avg_train_loss, avg_val_loss

def eval_model(model, val_loader):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_val_loss += outputs.loss

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch} | Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

def log_metrics(save_dir, train_loss, validation_loss):
    log_file = os.path.join(save_dir, "metrics_log.txt")

    line = (
    f"Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f} \n\n"
    )

    with open(log_file, "a") as f:
        f.write(line)

tokenizer = Tokenizer(alphabet.char_token)


UsePretrained = True
# 2. Load pretrained TrOCR model & feature extractor
model = None
if UsePretrained:
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-stage1')
else:
    model = VisionEncoderDecoderModel.from_pretrained("/scratch/s3799042/weights/huggingface_ocr/Checkpoint/trocr-hebrew-finetuned/")

feature_extractor = ViTFeatureExtractor.from_pretrained('microsoft/trocr-base-stage1')
# Resize token embeddings to new vocab size
model.config.vocab_size = tokenizer.vocab_size()
model.decoder.resize_token_embeddings(tokenizer.vocab_size())
model.config.pad_token_id = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join("/scratch/s3799042/weights/huggingface_ocr/", timestamp)
os.makedirs(save_dir, exist_ok=True)

# 3. Dataset class
class OCRDataset(Dataset):
    def __init__(self, scrolls: np.ndarray, lines: np.ndarray, tokens, feature_extractor,  max_length=128):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.scrolls = scrolls
        self.tokens = tokens
        self.all_lines = []
        self.all_tokens = []
        for i in range(scrolls.shape[0]):
            extracted_lines = synthetic.extract_lines_cc(scrolls[i], lines[i])
            for j in range(len(extracted_lines)):
                self.all_tokens.append(tokens[i][j])
                self.all_lines.append(extracted_lines[j])

    def __len__(self):
        return len(self.all_lines)

    def _pad_resize_images(self, image: np.array):
        image_resized = (384, 384)
        max_width = 3000
        width = image.shape[1]
        height = image.shape[0]
        ratio_width = width / max_width
        pad_width = (1-ratio_width) * image_resized[1]
        pad_height = pad_width + width - height
        padding = (int(pad_width), 0, 0, int(pad_height))  # left, top, right, bottom
        
        image = Image.fromarray(image)
        image = F.pad(image, padding, fill=255)

        #plt.imshow(image)
        #plt.title(f"height: {image.height}, width: {image.width}")
        #plt.axis("off")

        # Save the figure
        #plt.savefig("output_example.png", bbox_inches="tight", dpi=300)
        #print("created_image")
        return image

    def __getitem__(self, idx):
        tokens = self.all_tokens[idx]
        image = self.all_lines[idx]
        image = self._pad_resize_images(image)
        # If float image (0.0–1.0), scale to 0–255
        #if image.dtype in [np.float32, np.float64]:
        #    image = (image * 255).astype(np.uint8)
        #elif image.dtype != np.uint8:
        #    image = image.astype(np.uint8)

        # Create and save the image
        #img = Image.fromarray(image)
        #img.save("output.png")

        rgb = np.stack([image] * 3, axis=-1)  # Grayscale
        pixel_values = self.feature_extractor(images=rgb, return_tensors='pt').pixel_values[0]

        labels = tokenizer.add_control_tokens(tokens)
        labels = torch.tensor(labels)

        return {
            'pixel_values': pixel_values,
            'labels': labels
        }


# 5. Optimizer & scheduler
batch_size = 32
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
num_epochs = 10
total_steps = 60_000 * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 6. Training & validation loops
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = DataParallel(model, device_ids=[0, 1])
model.to(device)

for epoch in range(1, num_epochs+1):
    model.train()
    total_train_loss = 0
    pool = {i for i in range(5)}
    level = random.choice(list(pool))
    print(f"noise_{level}")
    iterator = noise_designer.load_batches(level=level)
    val_tokens, val_scrolls, val_lines = next(iterator)
    val_scrolls = val_scrolls[:1000]
    val_lines = val_lines[:1000]

    val_data = OCRDataset(val_scrolls, val_lines, val_tokens, feature_extractor)

    for train_tokens, train_scrolls, train_lines in iterator:
        train_data = OCRDataset(train_scrolls, train_lines, train_tokens, feature_extractor)
        avg_train_loss, avg_val_loss = train_epoch(model, train_data, val_data, optimizer, batch_size, epoch)
        model.cpu()
        model.save_pretrained(
            os.path.join(save_dir, f"trocr-hebrew-finetuned_{str(epoch)}"),
            safe_serialization=False,  # or True with sharding
            max_shard_size="1GB"
        )
        model.to(device)
        log_metrics(save_dir, avg_train_loss, avg_val_loss)

