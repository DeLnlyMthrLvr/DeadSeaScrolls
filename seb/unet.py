import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes=27, base_ch=64):
        super().__init__()
        # Encoder
        self.enc1 = self.conv_block(1, base_ch)
        self.enc2 = self.conv_block(base_ch, base_ch*2)
        self.enc3 = self.conv_block(base_ch*2, base_ch*4)
        self.enc4 = self.conv_block(base_ch*4, base_ch*8)
        # Bottleneck
        self.bottleneck = self.conv_block(base_ch*8, base_ch*16) #512, 1024
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(base_ch*16, base_ch*8)
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(base_ch*2, base_ch)
        # Final 1×1
        self.final = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), 
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), 
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x); p1 = F.max_pool2d(e1, 2)
        e2 = self.enc2(p1); p2 = F.max_pool2d(e2, 2)
        e3 = self.enc3(p2); p3 = F.max_pool2d(e3, 2)
        e4 = self.enc4(p3); p4 = F.max_pool2d(e4, 2)
        # Bottleneck
        b = self.bottleneck(p4)
        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        # Output
        return F.sigmoid(self.final(d1))
