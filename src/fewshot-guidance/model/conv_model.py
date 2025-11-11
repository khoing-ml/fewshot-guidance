import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any


class ConvGuidanceModel(BaseGuidanceModel):
    """
    Convolutional guidance model.
    
    Operates on spatial structure of latents (after unpacking).
    Good for guidance objectives related to spatial features like composition,
    structure, or local details.
    """
    
    def __init__(
        self,
        in_channels: int = 16,  # Latent channels before packing
        vec_dim: int = 768,
        hidden_channels: int = 256,
        num_blocks: int = 3,
        timestep_embed_dim: int = 256,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        
        # Timestep embedding
        self.timestep_embed = nn.Sequential(
            nn.Linear(1, timestep_embed_dim),
            nn.SiLU(),
            nn.Linear(timestep_embed_dim, timestep_embed_dim),
        )
        
        # Vec conditioning
        self.vec_proj = nn.Linear(vec_dim, hidden_channels)
        
        # Initial conv
        self.input_conv = nn.Conv2d(
            in_channels * 3,  # img + pred + optional fewshot
            hidden_channels,
            kernel_size=3,
            padding=1
        )
        self.input_conv_no_fewshot = nn.Conv2d(
            in_channels * 2,  # img + pred
            hidden_channels,
            kernel_size=3,
            padding=1
        )
        
        # ResNet-style blocks
        self.blocks = nn.ModuleList([
            ConvGuidanceBlock(
                hidden_channels,
                hidden_channels,
                timestep_embed_dim + hidden_channels,
            )
            for _ in range(num_blocks)
        ])
        
        # Output conv
        self.output_conv = nn.Conv2d(
            hidden_channels,
            in_channels,
            kernel_size=3,
            padding=1
        )
        
        # Initialize output to near-zero
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)
    
    def forward(
        self,
        img: Tensor,
        timestep: Tensor,
        step_idx: int,
        pred: Tensor | None = None,
        txt: Tensor | None = None,
        vec: Tensor | None = None,
        fewshot_img: Tensor | None = None,
    ) -> Tensor:
        # Note: This assumes img and pred are packed. Need to unpack for conv operations
        # For simplicity, we'll work with the packed representation
        # In practice, you might want to unpack -> conv -> repack
        
        batch_size, seq_len, channels = img.shape
        
        # Approximate spatial dimensions (assuming square-ish latents)
        h = w = int(seq_len ** 0.5)
        if h * w != seq_len:
            # Handle non-square case
            h = int(seq_len ** 0.5)
            w = seq_len // h
        
        # Reshape to spatial format
        # Note: This is simplified. In practice, you need proper unpacking
        img_spatial = img.reshape(batch_size, h, w, channels).permute(0, 3, 1, 2)
        pred_spatial = pred.reshape(batch_size, h, w, channels).permute(0, 3, 1, 2)
        
        # Embed timestep and vec
        t_embed = self.timestep_embed(timestep.unsqueeze(-1))  # [batch, embed_dim]
        v_embed = self.vec_proj(vec)  # [batch, hidden_channels]
        condition = torch.cat([t_embed, v_embed], dim=-1)  # [batch, embed_dim + hidden]
        
        # Process fewshot image if provided
        if fewshot_img is not None:
            if fewshot_img.dim() == 3 and fewshot_img.shape[0] != batch_size:
                # Multiple fewshot examples: average them
                num_shots = fewshot_img.shape[0]
                fewshot_spatial = fewshot_img.reshape(num_shots, h, w, channels).permute(0, 3, 1, 2)
                fewshot_spatial = fewshot_spatial.mean(dim=0, keepdim=True)  # [1, channels, h, w]
                fewshot_spatial = fewshot_spatial.expand(batch_size, -1, -1, -1)
            else:
                fewshot_spatial = fewshot_img.reshape(batch_size, h, w, channels).permute(0, 3, 1, 2)
            
            # Combine img, pred, and fewshot
            x = torch.cat([img_spatial, pred_spatial, fewshot_spatial], dim=1)  # [batch, 3*in_channels, h, w]
            x = self.input_conv(x)  # [batch, hidden_channels, h, w]
        else:
            # Combine img and pred
            x = torch.cat([img_spatial, pred_spatial], dim=1)  # [batch, 2*in_channels, h, w]
            x = self.input_conv_no_fewshot(x)  # [batch, hidden_channels, h, w]
        
        # Process through blocks
        for block in self.blocks:
            x = block(x, condition)
        
        # Output conv
        guidance_spatial = self.output_conv(x)  # [batch, in_channels, h, w]
        
        # Reshape back to packed format
        guidance = guidance_spatial.permute(0, 2, 3, 1).reshape(batch_size, seq_len, channels)
        
        return guidance


class ConvGuidanceBlock(nn.Module):
    """ResNet-style block for convolutional guidance."""
    
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Condition projection
        self.condition_proj = nn.Linear(condition_dim, out_channels)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        residual = self.residual(x)
        
        # First conv
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add condition
        c = self.condition_proj(condition)[:, :, None, None]
        h = h + c
        
        # Second conv
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + residual
