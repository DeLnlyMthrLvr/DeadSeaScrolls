import torch
import torch.nn as nn

class PatchEmbeding(nn.Module):
  def __init__(self, image_width, image_height, patch_size, embeding_dimension):
    super(PatchEmbeding, self).__init__()

    self.patch_size = patch_size
    self.embeding_dimension = embeding_dimension

    self.num_patches_width = (image_width // patch_size)
    self.num_patches_height = (image_height // patch_size)

    self.linear = nn.Linear(patch_size * patch_size, embeding_dimension)

  def forward(self, images):
    #flattened_patches = self.get_flattened_patches(images)
    # Unfold the image to get patches of shape [batch_size, num_patches, patch_size*patch_size]
    patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
    print(patches.shape)
    # Reshape patches to (batch_size, num_patches, patch_size*patch_size*3)
    patches = patches.contiguous().view(images.size(0), -1, self.patch_size * self.patch_size)

    print(patches)
    print(patches.shape)
    patch_embeddings = self.linear(patches)
    return patch_embeddings




