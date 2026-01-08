from typing import Optional, Tuple
import torch
import torch.nn as nn

# siglip config
class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size: int = 768, # size of the embedding vector de, which we are going to encode
        intermediate_size: int = 3072, # size of the linear layer in FFN
        num_hidden_layers = 12,
        num_attention_heads = 12,
        num_channels = 3,
        image_size = 224,
        patch_size = 16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
        
    )

    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.num_channels = num_channels
    self.patch_size = patch_size
    self.image_size = image_size
    self.attention_dropout = attention_dropout
    self.layer_norm_eps = layer_norm_eps
    self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels = config.num_channels,
            out_channels = self.
        )


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_states = self.encoder(hidden_states)

        last_hidden_states = self.post_layernorm(last_hidden_states)

        return last_hidden_states

class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim], Num_Patches is same as num_image_tokens
        return self.vision_model(pixel_values)

