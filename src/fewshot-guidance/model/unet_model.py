import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any
from einops import rearrange
from diffusers.models.unets.unet_2d import UNet2DModel
from .base_model import BaseGuidanceModel

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
        timestep: Tensor,
        step_idx: int,
        pred: Tensor | None = None,
        txt: Tensor | None = None,
        vec: Tensor | None = None,
        fewshot_img: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass of the UNet guidance model.
        
        Args:
            img: Current image latent tensor [batch, seq_len, channels]
            timestep: Current timestep tensor [batch]
            step_idx: Current step index
            pred: Current prediction latent tensor [batch, seq_len, channels]
            txt: Text embedding tensor [batch, txt_seq_len, txt_dim]
            vec: CLIP embedding tensor [batch, vec_dim]
            fewshot_img: Optional fewshot features [batch, seq_len, channels]
        
        Returns:
            Predicted guidance signal tensor [batch, seq_len, channels]
        """
        batch_size, seq_len, channels = img.shape
        
        # Ensure pred is provided
        if pred is None:
            raise ValueError("UNetGuidanceModel requires 'pred' tensor")
        
        # Unpack from sequence format to spatial format
        # Assume square latent: seq_len = h * w
        h = w = int(seq_len ** 0.5)
        assert h * w == seq_len, f"seq_len {seq_len} must be a perfect square for UNet"
        
        # Reshape: [batch, seq_len, channels] -> [batch, channels, h, w]
        img_spatial = rearrange(img, 'b (h w) c -> b c h w', h=h, w=w)
        pred_spatial = rearrange(pred, 'b (h w) c -> b c h w', h=h, w=w)
        
        # Combine img and pred
        x = torch.cat([img_spatial, pred_spatial], dim=1)  # [batch, 2*channels, h, w]
        
        # Prepare conditioning
        if self.use_timestep_embedding:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                t_emb = self.timestep_embed(timestep.unsqueeze(-1))  # [batch, timestep_embed_dim]
            cond = torch.cat([vec, t_emb], dim=-1) if vec is not None else t_emb  # [batch, condition_dim]
        else:
            t_emb = timestep.unsqueeze(-1)  # [batch, 1]
            cond = torch.cat([vec, t_emb], dim=-1) if vec is not None else t_emb  # [batch, condition_dim]
        
        # Expand conditioning for each spatial position (UNet cross-attention expects sequence)
        # [batch, condition_dim] -> [batch, 1, condition_dim]
        cond = cond.unsqueeze(1)
        
        # UNet forward with autocast
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.unet(
                sample=x,
                timestep=timestep,  # Pass timestep directly to UNet
                encoder_hidden_states=cond,
                return_dict=False,
            )[0]  # [batch, channels, h, w]
        
        # Pack back to sequence format: [batch, channels, h, w] -> [batch, seq_len, channels]
        guidance = rearrange(out, 'b c h w -> b (h w) c')
        
        return guidance