import torch
import ocr_model
from PIL import Image
import torchvision.transforms as transforms

image = Image.open("test-image-deepsea.png").convert("L")

transformer = transforms.ToTensor()

image_tensor = transformer(image)
#add batch dimension
image_tensor = image_tensor.unsqueeze(0)
print(image_tensor.shape)
patch_size = 16
embeding_dimension = 64
patch_embeding_model = ocr_model.PatchEmbeding(image_tensor.shape[2], image_tensor.shape[1], patch_size, embeding_dimension)
output = patch_embeding_model(image_tensor)
print(output)
print(output.shape)
