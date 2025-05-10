import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import timm
from torch.nn.functional import pad
from torchvision import transforms
import os
from PIL import Image
from tokenizer import Tokenizer
import sys
import image_creator
import torch.nn.functional as F
import pandas as pd
from torchvision.transforms.functional import to_pil_image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import synthetic
import alphabet

# --- OCR Model with CTC Loss ---
class OCRWithCTC(nn.Module):
    def __init__(self, vit_model, num_classes):
        super(OCRWithCTC, self).__init__()
        self.vit = vit_model  # Pretrained Vision Transformer (ViT)
        self.embed_dim = vit_model.num_features  # Typically 192 or 384 for DeiT models

        # Remove the classification head (not needed for CTC)
        self.vit.reset_classifier(0)

        # CTC head: Project to class logits (num_classes including blank token)
        self.ctc_head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, images):
        # 1. Get patch embeddings from ViT model
        x = self.vit.patch_embed(images)  # Shape: (B, N, D)
        cls_token = self.vit.cls_token.expand(x.size(0), -1, -1)  # (B, 1, D)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.vit.pos_embed[:, :x.size(1), :]
        x = self.vit.pos_drop(x)

        # 2. Pass through transformer blocks
        for blk in self.vit.blocks:
            x = blk(x)

        # 3. Remove class token, only keep patch tokens (B, T, D)
        x = x[:, 1:, :]  # B x T x D

        # 4. Project to character logits (B, T, C)
        logits = self.ctc_head(x)

        # 5. Permute for CTC loss (T, B, C)
        return logits.permute(1, 0, 2)  # (T, B, C)

# --- Collate Function for Padding Images ---
def ocr_collate(batch):
    images, targets, input_lengths, target_lengths = zip(*batch)
    images = torch.stack(images)
    # Flatten targets
    targets_concat = torch.cat(targets)
    input_lengths = torch.tensor(input_lengths)
    target_lengths = torch.tensor(target_lengths)
    return images, targets_concat, input_lengths, target_lengths

class BibleDatasetIterable(IterableDataset):
    def __init__(self, tokenizer, image_size):
        self.tokenizer = tokenizer
        self.image_size = image_size
        # Example generator; replace with real data generation.
        self.generator = synthetic.DataGenerator(settings=synthetic.SynthSettings(downscale_factor=1))
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def line_iterator(self):
        while True:
            tokens, _, scrolls, lines = self.generator.generate_passages_scrolls(100, skip_char_seg=False)
            for i in range(scrolls.shape[0]):
                line_images = synthetic.extract_lines_cc(scrolls[i], lines[i])
                for img, tok in zip(line_images, tokens[i]):
                    image = self.transform(image_creator.pad(Image.fromarray(img), self.image_size))
                    #to_pil_image(image).save("synthetic.png")
                    image = image.repeat(3, 1, 1) # 3 channels instead of one so it matches with pre-trained model
                    #img_tensor = torch.tensor(img).unsqueeze(0)  # Now the shape will be (1, height, width)
                
                    # Ensure the image is in the correct shape for the ViT model (batch, channels, height, width)
                    height, width = image.shape[1:3]
                    num_patches_height = height // 16
                    num_patches_width = width // 16
                    
                    # Total number of patches (tokens)
                    input_length = num_patches_height * num_patches_width
                    yield image, torch.tensor(tok), input_length, len(tok)

    def __iter__(self):
        return self.line_iterator()

class DatasetFromFolder(Dataset):
    def __init__(self, tokens_parquet, image_dir, tokenizer, image_size):
        self.tokenizer = tokenizer
        self.tokens = pd.read_parquet(tokens_parquet, engine='pyarrow')
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, f"{self.tokens.at[idx, 'image']}.png")
        image = Image.open(image_path).convert("L")
        image = image_creator.pad(image, self.image_size)
        image = self.transform(image)
        image = image.repeat(3, 1, 1)
        #to_pil_image(image).save("real.png")
        height, width = image.shape[1:3]
        num_patches_height = height // 16
        num_patches_width = width // 16
        input_length = num_patches_height * num_patches_width
        tokens = self.tokens.at[idx, "tokens"]
        return image, torch.tensor(tokens, dtype=torch.long), input_length, len(tokens)
    

def greedy_decode_indices(log_probs, blank=27):
    """
    Args:
        log_probs: Tensor of shape (T, B, C)
    Returns:
        List[List[int]]: Decoded index sequences for each batch element (no blanks, no repeats)
    """
    pred_indices = log_probs.argmax(dim=2)  # (T, B)
    pred_indices = pred_indices.transpose(0, 1)  # (B, T)

    decoded_indices = []
    for indices in pred_indices:
        decoded = []
        prev = -1
        for idx in indices:
            idx = idx.item()
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        decoded_indices.append(decoded)
    return decoded_indices

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    target_pointer = 0
    with torch.no_grad():
        for images, targets, input_lengths, target_lengths in data_loader:
            images = images.to(device)
            logits = model(images)                # (T, B, C)
            log_probs = F.log_softmax(logits, 2)  # CTC expects log-probs
            decoded_indices = greedy_decode_indices(log_probs, blank=27)
                    
            for i, target_len in enumerate(target_lengths):
            # Get ground truth slice for this sample
                target_seq = targets[target_pointer:target_pointer + target_len].tolist()
                target_pointer += target_len

                # Get predicted sequence
                pred_seq = decoded_indices[i]

                # Compare
                correct += sum([1 if t == p else 0 for t, p in zip(target_seq, pred_seq)])
                total += len(target_seq)

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}")
    return accuracy


# --- Training Loop ---
def train(model, train_loader,test_loader, criterion, optimizer, scheduler, device):
    model.train()
    average_loss = 0
    count = 0
    
    for images, targets, input_lengths, target_lengths in train_loader:
        images, targets = images.to(device), targets.to(device)
        input_lengths, target_lengths = input_lengths.to(device), target_lengths.to(device)

        # Forward pass
        logits = model(images)
        log_probs = F.log_softmax(logits, dim=2)
        # Compute CTC loss
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        
        # Backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Track average loss
        average_loss += loss.item()
        count += 1

        if count % 1000 == 0:
            print(f"Iteration {count}: Loss = {average_loss / count}")
            evaluate(model, test_loader,device)
            model.train()

        if count == 3_000_000:
            torch.save(model.state_dict(), "model_weights.pth")
            break

    return average_loss / count

# --- Main Execution ---
if __name__ == "__main__":
    # Initialize the model (e.g., `deit_tiny_patch16_224`)
    vit_model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
    model = OCRWithCTC(vit_model, num_classes=28) #+ 1 blank
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Initialize optimizer, scheduler, and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CTCLoss(blank=27, zero_infinity=True)
    batch_size = 32
    # Dataset and DataLoader
    tokenizer = Tokenizer(alphabet.char_token)

    image_size = (224, 224)
    current_dir = os.path.dirname(__file__)
    data_base = os.path.join(current_dir, '..', 'data')
    test_base = os.path.join(data_base, 'sample-test-2025', 'Lines')
    train_dataset = BibleDatasetIterable(tokenizer, image_size=image_size)  # Example image size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=ocr_collate)
    
    test_dataset = DatasetFromFolder(os.path.join(test_base, 'tokens.parquet'),
                                     test_base, tokenizer, image_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=ocr_collate)


    synt_dataset = DatasetFromFolder(os.path.join(test_base, 'tokens.parquet'),
                                     test_base, tokenizer, image_size)
    synt_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=ocr_collate)
    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        avg_loss = train(model, train_loader,test_loader, criterion, optimizer, scheduler, device)
        print(f"Average loss: {avg_loss}")

    # Save the model after training
    torch.save(model.state_dict(), "ocr_model.pth")
    print("Model saved!")