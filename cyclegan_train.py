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


MAX_SEQ_LEN = 100 
gen = DataGenerator(max_sequence_length=MAX_SEQ_LEN,
                    settings=SynthSettings(downscale_factor=0.3))

num_synthetic = 5000
root_real     = 'data/image-data'
output_dir    = 'checkpoints/cyclegan'
img_size           = (120, 300)
batch_size         = 16
epochs             = 100
lr                 = 2e-4
beta1              = 0.5
lambda_cycle       = 10.0
lambda_id          = 0.1
sample_image_freq  = 200  # log images every N steps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Playing on: {device}")

# DataLoader
dataloader = create_unpaired_dataloader(
    synthetic_generator=gen,
    num_synthetic=num_synthetic,
    root_real=root_real,
    img_size=img_size,
    batch_size=batch_size
)

# Models
G_A2B = ResnetGenerator(1, 1).to(device)
G_B2A = ResnetGenerator(1, 1).to(device)
D_A    = NLayerDiscriminator(1).to(device)
D_B    = NLayerDiscriminator(1).to(device)

# Losses and optimizers
criterion_GAN   = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_id    = nn.L1Loss()
g_optimizer     = optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=lr, betas=(beta1, 0.999))
d_optimizer     = optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr=lr, betas=(beta1, 0.999))


writer = SummaryWriter(f'runs/g/{time.strftime("%Y%m%d-%H%M%S")}/')

os.makedirs(output_dir, exist_ok=True)

best_epoch_loss = float('inf')
best_epoch = -1

global_step = 0
for epoch in range(1, epochs+1):
    epoch_loss_G = 0.0
    num_batches = 0
    for i, (real_A, real_B) in enumerate(dataloader):
        real_A, real_B = real_A.to(device), real_B.to(device)

        # -- Train Generators
        g_optimizer.zero_grad()
        id_A = G_B2A(real_A)
        loss_id_A = criterion_id(id_A, real_A) * lambda_cycle * lambda_id
        id_B = G_A2B(real_B)
        loss_id_B = criterion_id(id_B, real_B) * lambda_cycle * lambda_id
        fake_B = G_A2B(real_A)
        loss_GAN_A2B = criterion_GAN(D_B(fake_B), torch.ones_like(D_B(fake_B)))
        fake_A = G_B2A(real_B)
        loss_GAN_B2A = criterion_GAN(D_A(fake_A), torch.ones_like(D_A(fake_A)))
        recov_A = G_B2A(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A) * lambda_cycle
        recov_B = G_A2B(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B) * lambda_cycle
        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B
        loss_G.backward()
        g_optimizer.step()

        epoch_loss_G += loss_G.item()
        num_batches += 1

        # -- Train Discriminators
        d_optimizer.zero_grad()
        pred_real_A = D_A(real_A)
        loss_D_real_A = criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))
        pred_fake_A = D_A(fake_A.detach())
        loss_D_fake_A = criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
        pred_real_B = D_B(real_B)
        loss_D_real_B = criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))
        pred_fake_B = D_B(fake_B.detach())
        loss_D_fake_B = criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))
        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
        loss_D = loss_D_A + loss_D_B
        loss_D.backward()
        d_optimizer.step()

        # -- Log Scalars
        writer.add_scalar('Loss/GAN_A2B', loss_GAN_A2B.item(), global_step)
        writer.add_scalar('Loss/GAN_B2A', loss_GAN_B2A.item(), global_step)
        writer.add_scalar('Loss/Cycle_A', loss_cycle_A.item(), global_step)
        writer.add_scalar('Loss/Cycle_B', loss_cycle_B.item(), global_step)
        writer.add_scalar('Loss/Identity_A', loss_id_A.item(), global_step)
        writer.add_scalar('Loss/Identity_B', loss_id_B.item(), global_step)
        writer.add_scalar('Loss/Generator', loss_G.item(), global_step)
        writer.add_scalar('Loss/Discriminator', loss_D.item(), global_step)

        # -- Log Sample Images
        if global_step % sample_image_freq == 0:
            with torch.no_grad():
                imgs = torch.cat([real_A[:4], fake_B[:4], real_B[:4], fake_A[:4]], dim=0)
                grid = make_grid(imgs, nrow=4)
                writer.add_image('Train/ImageSamples', grid, global_step)

        global_step += 1

        if i % 100 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

    avg_epoch_loss = epoch_loss_G / num_batches if num_batches > 0 else float('inf')

    with torch.no_grad():
        imgs = torch.cat([real_A[:4], fake_B[:4], real_B[:4], fake_A[:4]], dim=0)
        grid = make_grid(imgs, nrow=4, normalize=True)
        writer.add_image('Epoch/ImageGrid', grid, epoch)
    
    save_image((imgs * 0.5 + 0.5), os.path.join(output_dir, f"epoch_{epoch}.png"), nrow=4)
    torch.save(G_A2B.state_dict(), os.path.join(output_dir, f"G_A2B_{epoch}.pth"))
    torch.save(G_B2A.state_dict(), os.path.join(output_dir, f"G_B2A_{epoch}.pth"))
    torch.save(D_A.state_dict(),  os.path.join(output_dir, f"D_A_{epoch}.pth"))
    torch.save(D_B.state_dict(),  os.path.join(output_dir, f"D_B_{epoch}.pth"))

    if avg_epoch_loss < best_epoch_loss:
        best_epoch_loss = avg_epoch_loss
        best_epoch = epoch
        torch.save(G_A2B.state_dict(), os.path.join(output_dir, "G_A2B_best.pth"))
        torch.save(G_B2A.state_dict(), os.path.join(output_dir, "G_B2A_best.pth"))
        torch.save(D_A.state_dict(),  os.path.join(output_dir, "D_A_best.pth"))
        torch.save(D_B.state_dict(),  os.path.join(output_dir, "D_B_best.pth"))
        print(f"Saved new best checkpoint at epoch {epoch} with avg G loss {avg_epoch_loss:.4f}.")

    writer.flush()

print(f"Training complete. Best epoch: {best_epoch} (avg G loss = {best_epoch_loss:.4f})")
