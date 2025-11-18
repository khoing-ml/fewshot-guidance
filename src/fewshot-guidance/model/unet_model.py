import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any
from diffusers import UNet2DModel
from base_model import BaseGuidanceModel

class UNetGuidanceModel(BaseGuidanceModel):
    """
    UNet-based guidance model.
    
    Leverages a UNet architecture to process the current state and predict
    guidance signals. Suitable for complex guidance objectives.
    """
    
    def __init__(
        self,
        latent_channels: int = 64,  # channels after packing (16 * 2 * 2)
        txt_dim: int = 4096,  # T5 embedding dimension
        vec_dim: int = 768,  # CLIP embedding dimension
        unet_config: Optional[Dict[str, Any]] = None,
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
        
        # UNet model
        if unet_config is None:
            unet_config = {
                "in_channels": latent_channels * 2,  # img + pred
                "out_channels": latent_channels,
                "layers_per_block": 2,
                "block_out_channels": [128, 256, 512],
                "cross_attention_dim": condition_dim,
            }
        
        self.unet = UNet2DModel(**unet_config)

    def forward(
        self,
        img: Tensor,
        pred: Tensor,
        txt: Tensor,
        vec: Tensor,
        timestep: Tensor,
        fewshot: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the UNet guidance model.
        
        Args:
            img: Current image latent tensor (B, C, H, W)
            pred: Current prediction latent tensor (B, C, H, W)
            txt: Text embedding tensor (B, txt_dim)
            vec: CLIP embedding tensor (B, vec_dim)
            timestep: Current timestep tensor (B,)
            fewshot: Optional fewshot features (not used here)
        
        Returns:
            Predicted guidance signal tensor (B, C, H, W)
        """
        B, C, H, W = img.shape
        
        # Combine img and pred
        x = torch.cat([img, pred], dim=1)  # (B, 2C, H, W)
        
        # Prepare conditioning
        if self.use_timestep_embedding:
            t_emb = self.timestep_embed(timestep.unsqueeze(-1))  # (B, timestep_embed_dim)
            cond = torch.cat([vec, t_emb], dim=-1)  # (B, condition_dim)
        else:
            cond = torch.cat([vec, timestep.unsqueeze(-1)], dim=-1)  # (B, condition_dim)
        
        # UNet forward
        out = self.unet(
            sample=x,
            timestep=None,  # Timestep is handled via conditioning
            encoder_hidden_states=cond,
        ).sample  # (B, C, H, W)
        
        return out