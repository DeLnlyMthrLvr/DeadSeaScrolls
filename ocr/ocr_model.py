import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
  def __init__(self, image_width, image_height, patch_size, embedding_dimension):
    super(PatchEmbedding, self).__init__()

    self.patch_size = patch_size
    self.embedding_dimension = embedding_dimension

    self.num_patches_width = (image_width // patch_size)
    self.num_patches_height = (image_height // patch_size)
    self.total_num_patches = self.num_patches_width * self.num_patches_height
    
    self.linear = nn.Linear(patch_size * patch_size, embedding_dimension)
     
  def linear_projection(self, images): 
    # Unfold the image to get patches of shape [batch_size, num_patches, patch_size*patch_size]
    patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
    print(patches.shape)
    # Reshape patches to (batch_size, num_patches, patch_size*patch_size*3)
    patches = patches.contiguous().view(images.size(0), -1, self.patch_size * self.patch_size)

    print(patches)
    print(patches.shape)
    patch_embeddings = self.linear(patches)
    return patch_embeddings
    
  def forward(self, images):
    patch_embeddings = self.linear_projection(images)
    return patch_embeddings


class ViT(nn.Module):
  def __init__(self, image_width, image_height, patch_size, embedding_dimension, num_heads, depth, num_classes):
    super(ViT, self).__init__()
    
    self.patch_embedder = PatchEmbedding(image_width, image_height, patch_size, embedding_dimension)
    
    self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dimension))
    self.position_embedding = nn.Parameter(torch.randn(1, 1 + self.patch_embedder.total_num_patches, embedding_dimension))
    
    self.encoder = nn.TransformerEncoder(
      nn.TransformerEncoderLayer(
        d_model =  embedding_dimension,
        nhead = num_heads,
        dim_feedforward = embedding_dimension * 4,
        batch_first = True,
      ),
      num_layers=depth
    )
    
    self.mlp_head = nn.Sequential(
      nn.LayerNorm(embedding_dimension),
      nn.Linear(embedding_dimension, num_classes)
    )
    
  def forward(self, images):
    patch_embeddings = self.patch_embedder(images)
    cls_token = self.cls_token.expand(images.size(0), -1, -1)
    patch_embeddings = torch.cat([cls_token, patch_embeddings], dim=1)
    patch_embeddings = patch_embeddings + self.position_embedding
    
    visual_features = self.encoder(patch_embeddings)
    
    cls_output = visual_features[:, 0]
    return self.mlp_head(cls_output)