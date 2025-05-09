from pathlib import Path
from typing import Self
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes: int, base_ch: int = 30):
        super().__init__()
        self.num_classes = num_classes
        self.base_ch = base_ch

        self.enc1 = self.conv_block(1, base_ch)
        self.enc2 = self.conv_block(base_ch, base_ch * 2)
        self.enc3 = self.conv_block(base_ch * 2, base_ch * 4)
        self.enc4 = self.conv_block(base_ch * 4, base_ch * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        self.dec4 = self.dec_conv_block(base_ch * 8 * 2, base_ch * 4)
        self.dec3 = self.dec_conv_block(base_ch * 4 * 2, base_ch * 2)
        self.dec2 = self.dec_conv_block(base_ch * 2 * 2, base_ch)
        self.dec1 = self.dec_conv_block(base_ch * 2, base_ch)

        self.final = nn.Conv2d(base_ch, num_classes + 1, kernel_size=1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        in_d4 = torch.cat((e4, b), dim=1)
        d4 = self.dec4(in_d4)
        d4 = self._pad(d4, e3)

        in_d3 = torch.cat((e3, d4), dim=1)
        d3 = self.dec3(in_d3)
        d3 = self._pad(d3, e2)

        in_d2 = torch.cat((e2, d3), dim=1)
        d2 = self.dec2(in_d2)
        d2 = self._pad(d2, e1)

        in_d1 = torch.cat((e1, d2), dim=1)
        d1 = self.dec1(in_d1)
        d1 = self._pad(d1, x)

        return self.final(d1)

    def conv_block(self, in_ch, out_ch):

        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU(),
        )

    def dec_conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:

        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(out_ch, out_ch, kernel_size=5, padding=2),
            nn.LeakyReLU(),
        )

    def _pad(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff_y = target.size(2) - x.size(2)
        diff_x = target.size(3) - x.size(3)

        assert diff_y < 5 and diff_x < 5, f"{diff_y}, {diff_x}"

        return F.pad(x, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2))

    def save(self, folder: Path, name: str = "unet.pt", optimizer: torch.optim.Optimizer | None = None):

        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
                "base_ch": self.base_ch,
                "num_classes": self.num_classes
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

        instance = instance.to(instance.device)

        return instance