
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from .alphabet import A

@dataclass
class NoiseSettings:
    ...


def create_alphabet_images(
        text: list[A],
        alphabet: dict[str, list[np.ndarray]],
        noise: NoiseSettings | None = None
    ):

    if noise is None:
        noise = NoiseSettings()

