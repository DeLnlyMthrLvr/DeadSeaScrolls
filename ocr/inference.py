import torch
import ocr_model
from PIL import Image
import torchvision.transforms as transforms
import os
import pathlib


image = Image.open("/home3/s3799042/DeadSeaScrolls/ocr/test-image-deepsea.png").convert("L")

transformer = transforms.ToTensor()

image_tensor = transformer(image)
#add batch dimension
image_tensor = image_tensor.unsqueeze(0)
patch_size = 16
embedding_dimension = 64 #768 base
depth = 12 #12 base
num_heads = 4 #12 base
num_classes = 27
mlp_ratio = 0.4
dropout = 0.1
num_encoder_blocks = 4
#patch_embeding_model = ocr_model.PatchEmbeding(image_tensor.shape[2], image_tensor.shape[1], patch_size, embeding_dimension)
#output = patch_embeding_model(image_tensor)
ViT = ocr_model.ViT(image_tensor.shape[2], image_tensor.shape[1], patch_size, 
                      embedding_dimension, num_heads, depth, num_classes, mlp_ratio,
                      dropout, num_encoder_blocks)

output = ViT(image_tensor)
print(output)
print(output.shape)
