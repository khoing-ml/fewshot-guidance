import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any

from base_model import BaseGuidanceModel



class AttentionGuidanceModel(BaseGuidanceModel):
    """
    Attention-based guidance model.
    
    Uses cross-attention to condition on text embeddings and self-attention
    to process spatial relationships in the latent space.
    More powerful but requires more computation.
    """
    
    def __init__(
        self,
        latent_channels: int = 64,
        txt_dim: int = 4096,
        vec_dim: int = 768,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 2,
        timestep_embed_dim: int = 256,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Timestep embedding
        self.timestep_embed = nn.Sequential(
            nn.Linear(1, timestep_embed_dim),
            nn.SiLU(),
            nn.Linear(timestep_embed_dim, hidden_dim),
        )
        
        # Input projections
        self.img_proj = nn.Linear(latent_channels, hidden_dim)
        self.pred_proj = nn.Linear(latent_channels, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)
        self.vec_proj = nn.Linear(vec_dim, hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            AttentionGuidanceLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_channels)
        
        # Initialize output to near-zero
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
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
        batch_size, seq_len, channels = img.shape
        
        # Embed timestep
        t_embed = self.timestep_embed(timestep.unsqueeze(-1))  # [batch, hidden_dim]
        
        # Project inputs
        img_feat = self.img_proj(img)  # [batch, seq_len, hidden_dim]
        pred_feat = self.pred_proj(pred)  # [batch, seq_len, hidden_dim]
        
        # Combine img and pred
        x = img_feat + pred_feat  # [batch, seq_len, hidden_dim]
        
        # Add timestep conditioning
        x = x + t_embed.unsqueeze(1)  # [batch, seq_len, hidden_dim]
        
        # Project text and vec
        txt_feat = self.txt_proj(txt)  # [batch, txt_seq_len, hidden_dim]
        vec_feat = self.vec_proj(vec).unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Process fewshot reference image if provided
        if fewshot_img is not None:
            # fewshot_img: [batch, seq_len, channels] or [num_shots, seq_len, channels]
            if fewshot_img.dim() == 3 and fewshot_img.shape[0] != batch_size:
                # Multiple fewshot examples: process each and average
                fewshot_feat = self.img_proj(fewshot_img)  # [num_shots, seq_len, hidden_dim]
                fewshot_feat = fewshot_feat.mean(dim=0, keepdim=True)  # [1, seq_len, hidden_dim]
                fewshot_feat = fewshot_feat.expand(batch_size, -1, -1)  # [batch, seq_len, hidden_dim]
            else:
                fewshot_feat = self.img_proj(fewshot_img)  # [batch, seq_len, hidden_dim]
            
            # Add fewshot features to context (for cross-attention)
            context = torch.cat([vec_feat, fewshot_feat, txt_feat], dim=1)  # [batch, 1+seq_len+txt_seq_len, hidden_dim]
        else:
            # Combine text conditioning
            context = torch.cat([vec_feat, txt_feat], dim=1)  # [batch, 1+txt_seq_len, hidden_dim]
        
        # Process through attention layers
        for layer in self.layers:
            x = layer(x, context)
        
        # Project to output
        guidance = self.output_proj(x)  # [batch, seq_len, channels]
        
        return guidance
