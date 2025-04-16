
from pathlib import Path
from dataclasses import dataclass


@dataclass
class NoiseSettings:
    ...


def create_font_images(text: str, noise: NoiseSettings | None = None):
    ...

def create_alphabet_images(text: str, noise: NoiseSettings | None = None):
    ...