import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
  def __init__(self,
               image_width: int,
               image_height: int,
               patch_size: int,
               embedding_dimension: int):
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
    # Reshape patches to (batch_size, num_patches, patch_size*patch_size*3)
    patches = patches.contiguous().view(images.size(0), -1, self.patch_size * self.patch_size)

    patch_embeddings = self.linear(patches)
    return patch_embeddings
    
  def forward(self, images):
    patch_embeddings = self.linear_projection(images)
    return patch_embeddings


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embedding_dimension, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dimension)
        self.norm2 = nn.LayerNorm(embedding_dimension)
        self.norm3 = nn.LayerNorm(embedding_dimension)

        self.self_attention = nn.MultiheadAttention(embedding_dimension, num_heads, dropout=dropout, batch_first=True)

        hidden_dimension = int(embedding_dimension * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, embedding_dimension),
            nn.Dropout(dropout)
        )

        self.cross_attention = nn.MultiheadAttention(embedding_dimension, num_heads, dropout=dropout, batch_first=True)


    def forward(self, tgt, memory, tgt_mask=None):
        # tgt: (batch, tgt_seq_len, embed_dim)
        # memory: (batch, src_seq_len, embed_dim)

        # 1. Masked Self-Attention
        residual = tgt
        tgt = self.norm1(tgt)
        tgt, _ = self.self_attention(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + residual

        # 2. Feedforward (MLP)
        residual = tgt
        tgt = self.norm2(tgt)
        tgt = self.mlp(tgt)
        tgt = tgt + residual

        # 3. Cross Attention with encoder output
        residual = tgt
        tgt = self.norm3(tgt)
        tgt, _ = self.cross_attention(tgt, memory, memory, attn_mask=None)
        tgt = tgt + residual

        return tgt   
        
class ViTEncoderBlock(nn.Module):
  def __init__(self,
               embedding_dimension: int,
               num_heads: int,
               mlp_ratio: float,
               dropout: float):
    super(ViTEncoderBlock, self).__init__()
    self.norm1 = nn.LayerNorm(embedding_dimension)
    self.attention = nn.MultiheadAttention(embedding_dimension, num_heads, dropout=dropout, batch_first=True)

    hidden_dimension = int(embedding_dimension * mlp_ratio)

    self.mlp = nn.Sequential(
      nn.LayerNorm(embedding_dimension),
      nn.Linear(embedding_dimension, hidden_dimension),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dimension, embedding_dimension),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    normalized = self.norm1(x)
    x = x + self.attention(normalized, normalized, normalized)[0]  # MHSA
    x = x + self.mlp(x)
    return x

class ViT(nn.Module):
  def __init__(self,
               image_width: int,
               image_height: int,
               patch_size: int,
               embedding_dimension: int,
               num_heads: int,
               depth: int,
               num_classes: int,
               mlp_ratio: float,
               dropout: float,
               num_encoder_blocks: int):

    super(ViT, self).__init__()
    self.num_encoder_bocks = num_encoder_blocks
    self.patch_embedder = PatchEmbedding(image_width, image_height, patch_size, embedding_dimension)
    
    self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dimension))
    self.position_embedding = nn.Parameter(torch.randn(1, 1 + self.patch_embedder.total_num_patches, embedding_dimension))

    self.blocks = nn.ModuleList([
      ViTEncoderBlock(embedding_dimension, num_heads, mlp_ratio, dropout) for _ in range(depth)
    ])

    self.mlp_head = nn.Sequential(
      nn.LayerNorm(embedding_dimension),
      nn.Linear(embedding_dimension, num_classes)
    )
    
  def forward(self, images):
    patch_embeddings = self.patch_embedder(images)
    cls_token = self.cls_token.expand(images.size(0), -1, -1)
    patch_embeddings = torch.cat([cls_token, patch_embeddings], dim=1)
    x = patch_embeddings + self.position_embedding

    for block in self.blocks:
      x = block(x)
      
    memory = x[:, 1:, :]  # remove CLS token
    return memory
    
class OCR(nn.Module):
  def __init__(self,
               ViT,
               embedding_dimension_decoder: int,
               num_heads_decoder: int,
               depth_decoder: int,
               vocab_size: int,
               mlp_ratio_decoder = 4.0,
               dropout_decoder = 0.1):
    super(OCR, self).__init__()
    self.ViT = ViT
    
    self.token_embedding = nn.Embedding(vocab_size, embedding_dimension_decoder)
    
    self.decoder_blocks = nn.ModuleList([
      TransformerDecoderBlock(embedding_dimension_decoder, num_heads_decoder, mlp_ratio_decoder, dropout_decoder) for _ in range(depth_decoder)
    ])
    
    self.mlp_head = nn.Sequential(
      nn.LayerNorm(embedding_dimension_decoder),
      nn.Linear(embedding_dimension_decoder, vocab_size)
    )
  
  def generate_causal_mask(self, seq_len: int, device: torch.device):
    """
    Returns a (seq_len, seq_len) causal mask for decoder self-attention.
    Masked positions are set to -inf.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()  # Upper triangle
    # PyTorch's MultiheadAttention uses float masks with -inf or 0
    mask = mask.to(device)
    float_mask = mask.masked_fill(mask, float('-inf'))  # [seq_len, seq_len]
    return float_mask

  def forward(self, images, tgt_input):
    # Used for training (parallel decoding with teacher forcing)
    encoder_output = self.ViT(images)

    tgt_embed = self.token_embedding(tgt_input)  # Embed the token IDs

    x = tgt_embed
    seq_len = tgt_input.size(1)
    tgt_mask = self.generate_causal_mask(seq_len, tgt_input.device)
    for block in self.decoder_blocks:
        x = block(x, encoder_output, tgt_mask)
    logits = self.mlp_head(x)
    return logits

  def generate(self, images, max_length, bos_token_id, eos_token_id):
    # Used for inference (step-by-step decoding)
    encoder_output = self.ViT(images)
    output_tokens = [bos_token_id]
    
    for _ in range(max_length):
      tgt_input = torch.tensor(output_tokens).unsqueeze(0).to(images.device)
      tgt_embed = self.token_embedding(tgt_input)

      x = tgt_embed
      for block in self.decoder_blocks:
          x = block(x, encoder_output)
      
      logits = self.mlp_head(x)
      next_token = logits[:, -1, :].argmax(dim=-1).item()

      if next_token == eos_token_id:
          break
      output_tokens.append(next_token)

    return output_tokens

  
  