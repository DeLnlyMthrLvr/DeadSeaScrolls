import sys
sys.path.append("..")
from unet import UNet
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from synthetic import DataGenerator, SynthSettings
import cv2, numpy as np

TARGET_H, TARGET_W = 80, 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, val_masks, val_scrolls, masks, scrolls):

    scrolls = torch.tensor(scrolls)
    scrolls = torch.unsqueeze(scrolls, 1)
    masks = torch.tensor(masks)
    val_scrolls = torch.tensor(val_scrolls)
    val_scrolls = torch.unsqueeze(val_scrolls, 1)
    val_masks = torch.tensor(val_masks)

    scrolls_dataset = TensorDataset(scrolls, masks) #Scrolls and segmentation masks
    val_scrolls_dataset = TensorDataset(val_scrolls, val_masks)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    dataloader = DataLoader(dataset=scrolls_dataset)
    val_dataloader = DataLoader(dataset=val_scrolls_dataset)
    
    track_loss = []
    #Training
    for batch_scrolls, batch_masks in dataloader:     
        batch_scrolls, batch_masks = batch_scrolls.to(device), batch_masks.to(device)
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
    val_tokens, val_masks, val_scrolls = generator.generate_ngram_scrolls(200)
    print("Data Generated")
    model = UNet(num_classes=27).to(device)
    print("Starting Training")
    for epoch in range(10):
        train(model=model, 
              val_masks=val_masks, 
              val_scrolls=val_scrolls,
              masks=masks, 
              scrolls=scrolls)
    