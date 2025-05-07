from torchvision import transforms
from torchvision.transforms import ToPILImage
import torchvision.transforms.functional as F
import sys
import os
from PIL import Image
import csv
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import synthetic

def pad(image: Image.Image, image_size: tuple):
    factor_height = image.height / image_size[0] 
    factor_width = image.width / image_size[1]

    if factor_height * image_size[1] > image.width:
        padding = (0,0,(int)(factor_height * image_size[1] - image.width), 0)
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


if __name__ == "__main__":
    image_size = (32, 416)
    create_images_n_grams(100, image_size)