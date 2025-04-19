import sys
sys.path.append("..")
from unet import UNet
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from synthetic import DataGenerator, SynthSettings
from torchsummary import summary

TARGET_H, TARGET_W = 80, 200 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
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
    print("Dataset loaded")
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    dataloader = DataLoader(dataset=scrolls_dataset, batch_size=32)
    val_dataloader = DataLoader(dataset=val_scrolls_dataset, batch_size=32)
    
    track_loss = []

    print("Starting Training Loop")
    #Training
    for batch_scrolls, batch_masks in dataloader:     
        batch_scrolls, batch_masks = batch_scrolls.to(device), batch_masks.to(device)
        print(f"batch_masks shape: {batch_masks.shape} batch_scrolls shape: {batch_scrolls.shape}")
        segmentation_masks = model(batch_scrolls)             
        loss = loss_function(segmentation_masks, batch_masks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Training loss{loss.item()}")
        track_loss.append(loss.item())

    #Validation
    val_track_loss = []
    with torch.no_grad():
        for batch_scrolls, batch_masks in val_dataloader:
            batch_scrolls, batch_masks = batch_scrolls.to(device), batch_masks.to(device)
            segmentation_masks = model(batch_scrolls)
            loss = loss_function(segmentation_masks, batch_masks)
            val_track_loss.append(loss.item())
            print(f"Validation loss{loss.item()}")

    print(f"Epoch trainig loss: {torch.mean(track_loss)}, Epoch validation loss: {torch.mean(val_track_loss)}")


if __name__ == "__main__":
    gen_settings = SynthSettings(downscale_factor= 0.3)
    print("Loading Data")
    generator = DataGenerator( settings=gen_settings)
    tokens, masks, scrolls = generator.generate_ngram_scrolls(800) #Shapes: tokens(8000,150) , masks(8000, 27, H, W), scrolls(8000, 1, H, W)
    print("Generated training data")
    val_tokens, val_masks, val_scrolls = generator.generate_ngram_scrolls(200)
    print("Data Generated")
    model = UNet(num_classes=27).to(device)
    summary(model,(1,120,300))
    print(model)
    print("Starting Training")
    for epoch in range(10):
        train(model=model, 
              val_masks=val_masks, 
              val_scrolls=val_scrolls,
              masks=masks, 
              scrolls=scrolls)
    