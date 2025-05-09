
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from tokenizer import Tokenizer
import sys
import torch
import os
from PIL import Image
import textdistance

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import alphabet

class OCR():
    def __init__(self):
        self.tokenizer = Tokenizer(alphabet.char_token)
        self.model = VisionEncoderDecoderModel.from_pretrained('/home3/s3799042/DeadSeaScrolls/trocr-hebrew-finetuned')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('microsoft/trocr-base-stage1')

    def run_inference(self, line_images):
        generated_texts = []
        for image in line_images:
            pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.model.generate(pixel_values, max_length=128)

            generated_text = self.tokenizer.decode(generated_ids[0].cpu().numpy())
            print(generated_text)
            generated_texts.append(generated_text)
        
        return generated_texts
    

    def read_images_from_folder(self, folder_path):
        images = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):  # Only image files
                img_path = os.path.join(folder_path, filename)
                image = Image.open(img_path).convert('RGB')
                images.append(image)
        return images

    def compute_levensthein(self, s1, s2):
        levenshtein_normalized = textdistance.levenshtein.normalized_distance(s1, s2)
        print(levenshtein_normalized)

ocr = OCR()
images = ocr.read_images_from_folder("/home3/s3799042/DeadSeaScrolls/data/sample-test-2025/Lines")
ocr.run_inference(images)