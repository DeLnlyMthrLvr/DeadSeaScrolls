from torchvision import transforms
from torchvision.transforms import ToPILImage
import torchvision.transforms.functional as F
import sys
import os
from PIL import Image
import csv
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import synthetic

def pad(image: Image.Image, image_size: tuple):
    factor_height = image.height / image_size[0] 
    factor_width = image.width / image_size[1]
    #left,top,right,bottom
    if factor_height * image_size[1] > image.width:
        padding = ((int)(factor_height * image_size[1] - image.width),0,0, 0)
        return F.pad(image, padding, fill=255)
    if factor_width * image_size[0] > image.height:
        padding = (0,0,0,(int)(factor_width * image_size[0] - image.height))
        return F.pad(image, padding, fill=255)
    
    return image

def create_images(n_scrolls: int, image_size: tuple):
    transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(image_size),
    transforms.ToTensor()
    ]) 
    generator = synthetic.DataGenerator(settings=synthetic.SynthSettings(downscale_factor=1))
    current_dir = os.path.dirname(__file__)
    target_dir = os.path.join(current_dir, '..', 'data', 'test_images_ocr')

    parquet_path = os.path.join(target_dir, "tokens.parquet")
    tokens, seg, scrolls, lines = generator.generate_passages_scrolls(n_scrolls, skip_char_seg=False)
    rows = []
    for i in range(scrolls.shape[0]):
        image_tokens = tokens[i]
        image_lines = synthetic.extract_lines_cc(scrolls[i], lines[i])
        n_indicies = min(len(image_tokens), len(image_lines))

        for idx in range(n_indicies):
            image = Image.fromarray(image_lines[idx])
            image = pad(image, image_size)
            image = transform(image)
            to_pil = ToPILImage()
            image_pil = to_pil(image)
            image_pil.save(os.path.join(target_dir, f"scroll_{i}_line_{idx}.png"))
            rows.append({
            "image": f"scroll_{i}_line_{idx}",
            "tokens": image_tokens[idx]
            })
    df = pd.DataFrame(rows) 
    df.to_parquet(parquet_path,engine="pyarrow", index=False)

def create_images_n_grams(n_scrolls: int, image_size: tuple):
    transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(image_size),
    transforms.ToTensor()
    ]) 
    generator = synthetic.DataGenerator(settings=synthetic.SynthSettings(downscale_factor=1))
    current_dir = os.path.dirname(__file__)
    target_dir = os.path.join(current_dir, '..', 'data', 'n_gram_images')

    parquet_path = os.path.join(target_dir, "tokens.parquet")
    tokens, seg, scrolls, lines = generator.generate_ngram_scrolls(n_scrolls, skip_char_seg=False)
    rows = []
    for i in range(scrolls.shape[0]):
        image_tokens = tokens[i]
        image_lines = synthetic.extract_lines_cc(scrolls[i], lines[i])
        n_indicies = min(len(image_tokens), len(image_lines))

        for idx in range(n_indicies):
            image = Image.fromarray(image_lines[idx])
            image = pad(image, image_size)
            image = transform(image)
            to_pil = ToPILImage()
            image_pil = to_pil(image)
            image_pil.save(os.path.join(target_dir, f"scroll_{i}_line_{idx}.png"))
            rows.append({
            "image": f"scroll_{i}_line_{idx}",
            "tokens": image_tokens[idx]
            })
    df = pd.DataFrame(rows) 
    df.to_parquet(parquet_path,engine="pyarrow", index=False)

def create_noise_images(image_size):
    warp_strength = (0, 3)
    perlin_strength = (0.01, 0.1)
    cutout_size = (0, 20)
    n_noise_masks = 30
    n_images_per_level = 20
    n_progress = 5
    downscale = 0.5
    alphabet = synthetic.load_alphabet()
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    current_dir = os.path.dirname(__file__)
    target_dir = os.path.join(current_dir, '..', 'data', 'noise_images')

    parquet_path = os.path.join(target_dir, "tokens.parquet")

    #generator = synthetic.DataGenerator(settings=synthetic.SynthSettings(downscale_factor=downscale))
    perlin_strength = np.linspace(perlin_strength[0], perlin_strength[1], n_progress)
    warp_strength  = np.linspace(warp_strength[0], warp_strength[1], n_progress).round().astype(int)
    cutout_size = np.linspace(cutout_size[0], cutout_size[1], n_progress).round().astype(int)
    normal = synthetic.SynthSettings(downscale_factor=downscale)
    noise = synthetic.Noise(normal.downscale_size)
    noise.create_masks(N=n_noise_masks)
    image_idx = 0
    for level, (p, w, c) in enumerate(zip(perlin_strength, warp_strength, cutout_size, strict=True)):

        settings = synthetic.SynthSettings(
            warp_noise=w > 0,
            warp_noise_strength=w,
            cutout_noise=c > 0,
            cutout_noise_size=c,
            downscale_factor=downscale
        )
        generator = synthetic.DataGenerator(settings, alphabet)

        tokens_bible, seg_bible, scrolls_bible, lines_bible = generator.generate_passages_scrolls(1, skip_char_seg=True)
        tokens_grams, seg_grams, scrolls_grams, lines_grams = generator.generate_ngram_scrolls(1, skip_char_seg=True)

        if p > 0:
            modified_scrolls_bible = noise.damage(scrolls_bible, strength=p)
            modified_scrolls_grams = noise.damage(scrolls_grams, strength=p)
        for i in range(scrolls_bible.shape[0]):
            image_tokens = None
            modified_scrolls = None
            lines = None
            for mode in range(0,2):
                if mode == 0:
                    image_tokens = tokens_bible[i]
                    modified_scrolls = modified_scrolls_bible
                    lines = lines_bible
                if mode == 1:
                    image_tokens = tokens_grams[i]
                    modified_scrolls = modified_scrolls_grams
                    lines = lines_grams
                #image_tokens = tokens[i]
                image_lines = synthetic.extract_lines_cc(modified_scrolls[i], lines[i])
                n_indicies = min(len(image_tokens), len(image_lines))
                rows = []
                for idx in range(n_indicies):
                    image = Image.fromarray(image_lines[idx])
                    image = pad(image, image_size)
                    image = transform(image)
                    to_pil = ToPILImage()
                    image_pil = to_pil(image)
                    image_pil.save(os.path.join(target_dir, f"scroll_{image_idx}_line_{idx}.png"))
                    rows.append({
                    "image": f"scroll_{i}_line_{idx}",
                    "tokens": image_tokens[idx]
                    })
                image_idx+=1
        df = pd.DataFrame(rows) 
        df.to_parquet(parquet_path,engine="pyarrow", index=False)
                # image, token_tensor

if __name__ == "__main__":
    image_size = (32, 416)
    create_noise_images(image_size)