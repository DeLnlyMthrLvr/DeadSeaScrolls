from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms
from PIL import Image
import torch
import pandas as pd
import os
import numpy as np
import sys
import random
import synthetic
import image_creator
from tokenizer import Tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import noise_designer

class ScrollLineDataset(Dataset):
    def __init__(self, tokens_parquet, image_dir, tokenizer):
        self.tokenizer = tokenizer
        self.tokens = pd.read_parquet(tokens_parquet, engine='pyarrow')
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, f"{self.tokens.at[idx, 'image']}.png")
        image = Image.open(image_path).convert("L")
        image = self.transform(image)
        token_ids = self.tokenizer.add_control_tokens(self.tokens.at[idx, "tokens"])
        return image, torch.tensor(token_ids, dtype=torch.long)


class NoiseDataset(Dataset):
    def __init__(self, tokens_parquet, image_dir, tokenizer):
        self.tokenizer = tokenizer
        self.tokens = pd.read_parquet(tokens_parquet, engine='pyarrow')
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return 60_000

    def __getitem__(self, idx):
        rand_int = random.randint(0, 4)
        tokens, scrolls = noise_designer.load_batches(0)
        image_path = os.path.join(self.image_dir, f"{self.tokens.at[idx, 'image']}.png")
        image = Image.open(image_path).convert("L")
        image = self.transform(image)
        token_ids = self.tokenizer.add_control_tokens(self.tokens.at[idx, "tokens"])
        return image, torch.tensor(token_ids, dtype=torch.long)

class ScrollLineDatasetWithPadding(Dataset):
    def __init__(self, tokens_parquet, image_dir, tokenizer, image_size):
        self.tokenizer = tokenizer
        self.tokens = pd.read_parquet(tokens_parquet, engine='pyarrow')
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, f"{self.tokens.at[idx, 'image']}.png")
        image = Image.open(image_path).convert("L")
        image = image_creator.pad(image, self.image_size)
        image = self.transform(image)
        token_ids = self.tokenizer.add_control_tokens(self.tokens.at[idx, "tokens"])
        return image, torch.tensor(token_ids, dtype=torch.long)


class BaseIterableDataset(IterableDataset):
    def __init__(self, tokenizer, image_size):
        super().__init__()

        self.tokenizer = tokenizer
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

class BibleDatasetIterable(BaseIterableDataset):
    def __init__(self, tokenizer, image_size):
        super().__init__(tokenizer, image_size)

        self.generator = synthetic.DataGenerator(settings=synthetic.SynthSettings(downscale_factor=1))

    def line_iterator(self):
        while True:
            tokens, _, scrolls, lines = self.generator.generate_passages_scrolls(100, skip_char_seg=False)
            for i in range(scrolls.shape[0]):
                line_images = synthetic.extract_lines_cc(scrolls[i], lines[i])
                for img, tok in zip(line_images, tokens[i]):
                    image = self.transform(image_creator.pad(Image.fromarray(img), self.image_size))
                    yield image, torch.tensor(self.tokenizer.add_control_tokens(tok), dtype=torch.long)

    def __iter__(self):
        return self.line_iterator()
    
class NGramsDatasetIterable(BaseIterableDataset):
    def __init__(self, tokenizer, image_size):
        super().__init__(tokenizer, image_size)

        self.generator = synthetic.DataGenerator(settings=synthetic.SynthSettings(downscale_factor=1))

    def line_iterator(self):
        while True:
            tokens, _, scrolls, lines = self.generator.generate_ngram_scrolls(100, skip_char_seg=False)
            for i in range(scrolls.shape[0]):
                line_images = synthetic.extract_lines_cc(scrolls[i], lines[i])
                for img, tok in zip(line_images, tokens[i]):
                    image = self.transform(image_creator.pad(Image.fromarray(img), self.image_size))
                    yield image, torch.tensor(self.tokenizer.add_control_tokens(tok), dtype=torch.long)

    def __iter__(self):
        return self.line_iterator()


class MixedDatasetIterable(BaseIterableDataset):
    def __init__(self, tokenizer, image_size):

        super().__init__(tokenizer, image_size)

        self.generator = synthetic.DataGenerator(settings=synthetic.SynthSettings(downscale_factor=1))

    def line_iterator(self):
        while True:
            tokens_b, _, scrolls_b, lines_b = self.generator.generate_passages_scrolls(100, skip_char_seg=True)
            tokens_n, _, scrolls_n, lines_n = self.generator.generate_ngram_scrolls(100, skip_char_seg=True)
            for i in range(scrolls_b.shape[0]):
                for tokens, scrolls, lines in [(tokens_b, scrolls_b, lines_b), (tokens_n, scrolls_n, lines_n)]:
                    line_images = synthetic.extract_lines_cc(scrolls[i], lines[i])
                    for img, tok in zip(line_images, tokens[i]):
                        image = self.transform(image_creator.pad(Image.fromarray(img), self.image_size))
                        yield image, torch.tensor(self.tokenizer.add_control_tokens(tok), dtype=torch.long)

    def __iter__(self):
        return self.line_iterator()


class NoiseDatasetIterable(BaseIterableDataset):
    def __init__(self, tokenizer, image_size, warp_strength=(0, 10), perlin_strength=(0, 0.25),
                 cutout_size=(0, 40), n_noise_masks=30, n_images_per_level=500, n_progress=5, downscale=0.5):
        
        super().__init__(tokenizer, image_size)

        self.n_images_per_level = n_images_per_level
        self.alphabet = synthetic.load_alphabet()
        self.perlin_strength = np.linspace(perlin_strength[0], perlin_strength[1], n_progress)
        self.warp_strength = np.linspace(warp_strength[0], warp_strength[1], n_progress).round().astype(int)
        self.cutout_size = np.linspace(cutout_size[0], cutout_size[1], n_progress).round().astype(int)
        normal = synthetic.SynthSettings(downscale_factor=downscale)
        self.noise = synthetic.Noise(normal.downscale_size)
        self.noise.create_masks(N=n_noise_masks)
        self.noise.create_masks(N=n_noise_masks)
        self.downscale = downscale

    def line_iterator(self):
        while True:
            for p, w, c in zip(self.perlin_strength, self.warp_strength, self.cutout_size):
                settings = synthetic.SynthSettings(
                    warp_noise=w > 0,
                    warp_noise_strength=w,
                    cutout_noise=c > 0,
                    cutout_noise_size=c,
                    downscale_factor=self.downscale
                )
                generator = synthetic.DataGenerator(settings, self.alphabet)
                tokens, _, scrolls, lines = generator.generate_passages_scrolls(self.n_images_per_level, skip_char_seg=False)
                modified_scrolls = self.noise.damage(scrolls, strength=p) if p > 0 else scrolls
                for i in range(scrolls.shape[0]):
                    line_images = synthetic.extract_lines_cc(modified_scrolls[i], lines[i])
                    for img, tok in zip(line_images, tokens[i]):
                        image = self.transform(image_creator.pad(Image.fromarray(img), self.image_size))
                        yield image, torch.tensor(self.tokenizer.add_control_tokens(tok), dtype=torch.long)

    def __iter__(self):
        return self.line_iterator()


class NoiseMixedDatasetIterable(BaseIterableDataset):
    def __init__(self, tokenizer, image_size, warp_strength=(0, 3), perlin_strength=(0.01, 0.1),
                 cutout_size=(0, 20), n_noise_masks=30, n_images_per_level=500, n_progress=5, downscale=0.5):
        
        super().__init__(tokenizer, image_size)

        self.n_images_per_level = n_images_per_level
        self.alphabet = synthetic.load_alphabet()
        self.perlin_strength = np.linspace(perlin_strength[0], perlin_strength[1], n_progress)
        self.warp_strength = np.linspace(warp_strength[0], warp_strength[1], n_progress).round().astype(int)
        self.cutout_size = np.linspace(cutout_size[0], cutout_size[1], n_progress).round().astype(int)

        normal = synthetic.SynthSettings(downscale_factor=downscale)
        self.noise = synthetic.Noise(normal.downscale_size)
        self.noise.create_masks(N=n_noise_masks)
        self.downscale = downscale

    def line_iterator(self):
        while True:
            for p, w, c in zip(self.perlin_strength, self.warp_strength, self.cutout_size):
                settings = synthetic.SynthSettings(
                    warp_noise=w > 0,
                    warp_noise_strength=w,
                    cutout_noise=c > 0,
                    cutout_noise_size=c,
                    downscale_factor=self.downscale
                )
                generator = synthetic.DataGenerator(settings, self.alphabet)
                tokens_b, seg_b, scrolls_b, lines_b = generator.generate_passages_scrolls(100, skip_char_seg=False)
                tokens_n, seg_n, scrolls_n, lines_n = generator.generate_ngram_scrolls(100, skip_char_seg=False)
                mod_b = self.noise.damage(scrolls_b, strength=p) if p > 0 else scrolls_b
                mod_n = self.noise.damage(scrolls_n, strength=p) if p > 0 else scrolls_n
                for i in range(scrolls_b.shape[0]):
                    for tokens, scrolls, lines in [(tokens_b, mod_b, lines_b), (tokens_n, mod_n, lines_n)]:
                        line_images = synthetic.extract_lines_cc(scrolls[i], lines[i])
                        for img, tok in zip(line_images, tokens[i]):
                            image = self.transform(image_creator.pad(Image.fromarray(img), self.image_size))
                            yield image, torch.tensor(self.tokenizer.add_control_tokens(tok), dtype=torch.long)

    def __iter__(self):
        return self.line_iterator()