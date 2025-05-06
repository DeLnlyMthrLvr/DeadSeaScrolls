import os
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from ganv2.cyclegan_dataset import create_unpaired_dataloader
from ganv2.cyclegan_models import ResnetGenerator, NLayerDiscriminator
from synthetic import DataGenerator, SynthSettings
import time
import matplotlib.pyplot as plt


MAX_SEQ_LEN = 100 
gen = DataGenerator(max_sequence_length=MAX_SEQ_LEN,
                    settings=SynthSettings(downscale_factor=0.3))

num_synthetic = 10
root_real     = 'data/image-data'
output_dir    = 'checkpoints/cyclegan'
img_size           = (120, 300)
batch_size         = 16
epochs             = 100

# DataLoader
dataloader = create_unpaired_dataloader(
    synthetic_generator=gen,
    num_synthetic=num_synthetic,
    root_real=root_real,
    img_size=img_size,
    batch_size=batch_size
)

for img_A, img_B in dataloader:
    plt.imshow(img_A[0].squeeze(0).cpu().numpy(), cmap='gray')
    plt.savefig('testt.png')
    print('tr')
    break