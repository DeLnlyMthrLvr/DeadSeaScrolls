from torch.utils.data import IterableDataset
from torchvision import transforms
from torchvision.transforms import ToPILImage
from PIL import Image
import torch
import numpy as np
import sys
import os
import torchvision.transforms.functional as F
import image_creator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import synthetic

class ScrollLineDataset(IterableDataset):
    def __init__(self, tokenizer, image_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.generator = synthetic.DataGenerator(settings=synthetic.SynthSettings(downscale_factor=1))


    def line_iterator(self):
        while True:
            tokens, seg, scrolls, lines = self.generator.generate_passages_scrolls(1, skip_char_seg=False)

            for i in range(scrolls.shape[0]):
                image_tokens = tokens[i]
                image_lines = synthetic.extract_lines_cc(scrolls[i], lines[i])
                n_indicies = min(len(image_tokens), len(image_lines))

                #if (len(image_tokens) < len(image_lines)):
                #    print("test")
                #    image_no_tokens = Image.fromarray(image_lines[-1])
                #    seg_no_token = Image.fromarray(seg[-1])
                #    image_no_tokens.save("no_tokens.png")
                #    seg_with_token = Image.fromarray(seg[0])
                #    seg_with_token.save("seg_with_token.png")
                #    seg_no_token.save("seg_no_token.png")

                for idx in range(n_indicies):
                    image = Image.fromarray(image_lines[idx])

                    #image.save("original.png")
                    image = image_creator.pad(image, self.image_size)
                    #image.save("padded.png")
                    image = self.transform(image)
                    #to_pil = ToPILImage()
                    #image_pil = to_pil(image)
                    #image_pil.save("transformed.png")
                    token_ids = self.tokenizer.add_control_tokens(image_tokens[idx])
                    token_tensor = torch.tensor(token_ids, dtype=torch.long)
                    yield image, token_tensor

    def __iter__(self):
        return self.line_iterator()