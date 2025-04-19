import sys
sys.path.append("..")
from unet import UNet
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from synthetic import DataGenerator, SynthSettings
from torchsummary import summary
import numpy as np
import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

TARGET_H, TARGET_W = 80, 200  
from datetime import datetime

run_name = f"unet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=f"runs/{run_name}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class CustomDataset(Dataset):
    def __init__(self,scrolls, masks):
        self.scrolls = scrolls
        self.masks = masks

    def __len__(self):
        return self.scrolls.shape[0]
    
    def __getitem__(self, index):
        
        return torch.tensor(self.scrolls[index], dtype=torch.float).unsqueeze(0), torch.tensor(self.masks[index], dtype=torch.float32)

def train(model, val_masks, val_scrolls, masks, scrolls):

    scrolls_dataset = CustomDataset(scrolls, masks) #Scrolls and segmentation masks
    val_scrolls_dataset = CustomDataset(val_scrolls, val_masks)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    dataloader = DataLoader(dataset=scrolls_dataset, batch_size=32)
    val_dataloader = DataLoader(dataset=val_scrolls_dataset, batch_size=32)
    
    track_loss = []
    #Training
    for batch_scrolls, batch_masks in tqdm.tqdm(dataloader, desc="Training"):     
        batch_scrolls, batch_masks = batch_scrolls.to(device), batch_masks.to(device)
        segmentation_masks = model(batch_scrolls)             
        loss = loss_function(segmentation_masks, batch_masks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        track_loss.append(loss.item())

    #Validation
    val_track_loss = []
    with torch.no_grad():
        for batch_scrolls, batch_masks in tqdm.tqdm(val_dataloader, desc="Validation"):
            batch_scrolls, batch_masks = batch_scrolls.to(device), batch_masks.to(device)
            segmentation_masks = model(batch_scrolls)
            loss = loss_function(segmentation_masks, batch_masks)
            val_track_loss.append(loss.item())

    return np.mean(track_loss), np.mean(val_track_loss)


if __name__ == "__main__":
    gen_settings = SynthSettings(downscale_factor= 0.3)
    generator = DataGenerator( settings=gen_settings)
    model = UNet(num_classes=27).to(device)
    summary(model,(1,120,300))
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting Training")
    for epoch in tqdm.tqdm(range(10), "Cooking"):
        print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Generating batch number {epoch}")
        tokens, masks, scrolls = generator.generate_ngram_scrolls(256) #Shapes: tokens(8000,150) , masks(8000, 27, H, W), scrolls(8000, 1, H, W)
        val_tokens, val_masks, val_scrolls = generator.generate_ngram_scrolls(64)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Data Generated")
        train_loss, val_loss = train(model=model, 
              val_masks=val_masks, 
              val_scrolls=val_scrolls,
              masks=masks, 
              scrolls=scrolls)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Epoch trainig loss: {np.mean(train_loss)}, Epoch validation loss: {np.mean(val_loss)}")    
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)