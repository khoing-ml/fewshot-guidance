import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any

from .base_model import BaseGuidanceModel

class MLPGuidanceModel(BaseGuidanceModel):
    """
    Simple MLP-based guidance model.
    
    Processes the current state through MLP layers to predict guidance signal.
    Good starting point for simple guidance objectives.
    """
    
    def __init__(
        self,
        latent_channels: int = 64,  # channels after packing (16 * 2 * 2)
        txt_dim: int = 4096,  # T5 embedding dimension
        vec_dim: int = 768,  # CLIP embedding dimension
        hidden_dim: int = 1024,
        num_layers: int = 3,
        use_timestep_embedding: bool = True,
        timestep_embed_dim: int = 256,
    ):
        super().__init__()
        
        self.use_timestep_embedding = use_timestep_embedding
        self.latent_channels = latent_channels
        
        # Timestep embedding
        if use_timestep_embedding:
            self.timestep_embed = nn.Sequential(
                nn.Linear(1, timestep_embed_dim),
                nn.SiLU(),
                nn.Linear(timestep_embed_dim, timestep_embed_dim),
            )
            condition_dim = vec_dim + timestep_embed_dim
        else:
            condition_dim = vec_dim + 1  # vec + raw timestep
        
        # Process latent features
        self.img_proj = nn.Linear(latent_channels, hidden_dim)
        self.pred_proj = nn.Linear(latent_channels, hidden_dim)
        
        # Combine with conditioning (with optional fewshot features)
        # Max input: img + pred + fewshot + condition
        self.combine = nn.Linear(hidden_dim * 3 + condition_dim, hidden_dim)
        self.combine_no_fewshot = nn.Linear(hidden_dim * 2 + condition_dim, hidden_dim)
        
        # MLP layers
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
        self.mlp = nn.Sequential(*layers)
        
        # Output projection back to latent space
        self.output_proj = nn.Linear(hidden_dim, latent_channels)
        
        # Initialize output layer to near-zero for stable training
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
        
        # Process timestep
        if self.use_timestep_embedding:
            t_embed = self.timestep_embed(timestep.unsqueeze(-1))  # [batch, embed_dim]
        else:
            t_embed = timestep.unsqueeze(-1)  # [batch, 1]
        
        # Handle vec - if None, create zero vector of expected dimension
        if vec is None:
            vec = torch.zeros(batch_size, self.latent_channels, device=img.device, dtype=img.dtype)
        
        # Combine vec and timestep conditioning
        condition = torch.cat([vec, t_embed], dim=-1)  # [batch, condition_dim]
        condition = condition.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, condition_dim]
        
        # Reshape for linear operations: [batch*seq_len, channels]
        img_flat = img.reshape(-1, channels)
        pred_flat = pred.reshape(-1, channels) if pred is not None else torch.zeros_like(img_flat)
        condition_flat = condition.reshape(-1, condition.shape[-1])
        
        # Process img and pred
        img_features = self.img_proj(img_flat)  # [batch*seq_len, hidden_dim]
        pred_features = self.pred_proj(pred_flat)  # [batch*seq_len, hidden_dim]
        
        # Combine all features
        combined = torch.cat([img_features, pred_features, condition_flat], dim=-1)
        combined = self.combine_no_fewshot(combined)  # [batch*seq_len, hidden_dim]
        
        # Process through MLP
        features = self.mlp(combined)  # [batch*seq_len, hidden_dim]
        
        # Project to output
        guidance = self.output_proj(features)  # [batch*seq_len, channels]
        
        # Reshape back to [batch, seq_len, channels]
        guidance = guidance.reshape(batch_size, seq_len, channels)
        
        return guidance
