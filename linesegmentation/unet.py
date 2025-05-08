from pathlib import Path
from typing import Self
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelShuffle1D(nn.Module):
    def __init__(self, upscale_factor, dim_to_upscale="W"):
        super().__init__()
        self.upscale_factor = upscale_factor
        if dim_to_upscale not in {"H", "W"}:
            raise ValueError("dim_to_upscale must be 'H' or 'W'")
        self.dim_to_upscale = dim_to_upscale

    def forward(self, x):
        N, C_in, H, W = x.shape
        r = self.upscale_factor

        if C_in % r != 0:
            raise ValueError(f"Input channels ({C_in}) must be divisible by upscale_factor ({r})")
        C_out = C_in // r

        x = x.view(N, C_out, r, H, W) # (N, C_out, r, H, W)

        if self.dim_to_upscale == "H":
            # Target: (N, C_out, H*r, W)
            x = x.permute(0, 1, 3, 2, 4).contiguous() # (N, C_out, H, r, W)
            x = x.view(N, C_out, H * r, W)
        elif self.dim_to_upscale == "W":
            # Target: (N, C_out, H, W*r)
            x = x.permute(0, 1, 3, 4, 2).contiguous() # (N, C_out, H, W, r)
            x = x.view(N, C_out, H, W * r)
        return x

class UNet(nn.Module):
    def __init__(
            self,
            base_ch: int = 32,
            enc_kernel_size: tuple[int, int] = (12, 40)
        ) -> None:
        super().__init__()
        self.base_ch = base_ch

        self.r = 3
        self.enc_kernel_size = enc_kernel_size
        self.padding  = (int(enc_kernel_size[0] / 2), int(enc_kernel_size[1] / 2))

        self.enc1 = self.enc_conv_block(1, base_ch)
        self.enc2 = self.enc_conv_block(base_ch, base_ch * 2)
        self.enc3 = self.enc_conv_block(base_ch * 2, base_ch * 4)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4 , kernel_size=5, padding=2),
            nn.LeakyReLU(),
        )

        self.dec3 = self.dec_conv_block(base_ch * 4, base_ch * 2)
        self.dec2 = self.dec_conv_block(base_ch * 2, base_ch)
        self.dec1 = self.dec_conv_block(base_ch, base_ch)

        self.final = nn.Conv2d(base_ch, 1 , kernel_size=1, padding=0)

    def enc_conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=self.enc_kernel_size, padding=self.padding, stride=(1, self.r)),
            nn.LeakyReLU(),
        )

    def dec_conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch * self.r, kernel_size=5, padding=2),
            PixelShuffle1D(upscale_factor=self.r),
            nn.LeakyReLU(),
        )

    def _pad(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff_y = target.size(2) - x.size(2)
        diff_x = target.size(3) - x.size(3)

        assert diff_y < 5 and diff_x < 5, f"{diff_y}, {diff_x}"

        return F.pad(x, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        assert b.shape[-1] > 0

        d3 = torch.cat((e3, b), dim=1)
        d3 = self.dec3(d3)

        d3 = self._pad(d3, e2)
        d2 = torch.cat((e2, d3), dim=1)
        d2 = self.dec2(d2)

        d2 = self._pad(d2, e1)
        d1 = torch.cat((e1, d2), dim=1)
        d1 = self.dec1(d1)

        d1 = self._pad(d1, x)
        return self.final(d1)

    def save(self, folder: Path, name: str = "unet.pt", optimizer: torch.optim.Optimizer | None = None):

        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
                "base_ch": self.base_ch,
                "enc_kernel_size": self.enc_kernel_size
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