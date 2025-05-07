from pathlib import Path
from typing import Self
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, base_ch: int = 32) -> None:
        super().__init__()
        self.base_ch = base_ch
        self.enc1 = self.conv_block(1, base_ch)
        self.enc2 = self.conv_block(base_ch, base_ch * 2)
        self.enc3 = self.conv_block(base_ch * 2, base_ch * 4)
        self.enc4 = self.conv_block(base_ch * 4, base_ch * 8)

        self.bottleneck = self.conv_block(base_ch * 8, base_ch * 16)

        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, 2)
        self.dec4 = self.conv_block(base_ch * 16, base_ch * 8)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, 2)
        self.dec3 = self.conv_block(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, 2)
        self.dec2 = self.conv_block(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, 2)
        self.dec1 = self.conv_block(base_ch * 2, base_ch)
        self.final = nn.Conv2d(base_ch, 1, 1)

    @staticmethod
    def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 7, padding=3),
            nn.LeakyReLU(inplace=True)
        )

    def _pad(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff_y = target.size(2) - x.size(2)
        diff_x = target.size(3) - x.size(3)
        return F.pad(x, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))
        d4 = self.up4(b)
        d4 = self._pad(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], 1))
        d3 = self.up3(d4)
        d3 = self._pad(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], 1))
        d2 = self.up2(d3)
        d2 = self._pad(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], 1))
        d1 = self.up1(d2)
        d1 = self._pad(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], 1))
        return self.final(d1)

    def save(self, folder: Path, name: str = "unet.pt", optimizer: torch.optim.Optimizer | None = None):

        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
                "base_ch": self.base_ch
            },
            folder / name
        )

    @classmethod
    def load(cls, folder: Path, name: str = "unet.pt", optimizer: torch.optim.Optimizer | None = None) -> Self:

        cp = torch.load(folder / name, weights_only=False)

        msd = cp.pop("model_state_dict")
        osd = cp.pop("optimizer_state_dict")

        instance = cls(**cp)
        instance.load_state_dict(msd)

        if (optimizer is not None) and (osd is not None):
            optimizer.load_state_dict(osd)

        return instance
