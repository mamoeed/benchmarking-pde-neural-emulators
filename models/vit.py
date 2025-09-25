import torch
import torch.nn.functional as F
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, NamedTuple, Optional

class MLPBlock(torch.nn.Module):
    def __init__(self, in_dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, mlp_dim)
        self.fc2 = torch.nn.Linear(mlp_dim, in_dim)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class EncoderBlock(torch.nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = torch.nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

class Encoder(torch.nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = torch.nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = torch.nn.Dropout(dropout)
        layers: OrderedDict[str, torch.nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = torch.nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))



class VisionTransformer(torch.nn.Module):
    def __init__(self, grid_size: int = 128, patch_size: int = 16, in_channels: int = 1, out_channels: int = 1, 
                num_layers: int = 1, num_heads: int = 1, d_model: int = 16, mlp_dim: int = 64, attention_dropout: int = 0.1, 
                 dropout: int = 0.1, norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6)):
        super().__init__()
        torch._assert(grid_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.mlp_dim = mlp_dim
        self.attention_dropout = 0.1
        self.dropout = 0.1
        # self.num_classes = num_classes
        # self.representation_size = representation_size
        self.norm_layer = norm_layer

        # patchifying:
        self.conv_proj = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=d_model, kernel_size=patch_size, stride=patch_size
        )

        seq_length = (grid_size // patch_size) ** 2

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            d_model,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        # modification to original ViT for grid output:
        self.upsample = torch.nn.ConvTranspose2d(d_model, out_channels, kernel_size=patch_size, stride=patch_size)    
    
    def forward(self, x: torch.Tensor):
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.grid_size, f"Wrong image height! Expected {self.grid_size} but got {h}!")
        torch._assert(w == self.grid_size, f"Wrong image width! Expected {self.grid_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.d_model, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        n = x.shape[0]

        x = self.encoder(x)

        x = x.permute(0, 2, 1).reshape(n, self.d_model, n_h, n_w)

        return self.upsample(x)
        


def create_VisionTransformer(size="M", **kwargs):
    size_configs = {
        "S": {
                "grid_size" : 128, 
                "patch_size" : 8,
                "num_layers" : 3, 
                "num_heads" : 16, 
                "d_model" : 16, 
                "mlp_dim" : 32
            },
        "M": {
                "grid_size" : 128, 
                "patch_size" : 8,
                "num_layers" : 6, 
                "num_heads" : 24, 
                "d_model" : 48, 
                "mlp_dim" : 64
            },
        "L": {
                "grid_size" : 128, 
                "patch_size" : 8,
                "num_layers" : 9, 
                "num_heads" : 64, 
                "d_model" : 128, 
                "mlp_dim" : 180
            },
    }
    return VisionTransformer(**size_configs[size], **kwargs)







        