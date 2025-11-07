"""
Custom Guidance Models for Flux Sampling

This module provides example guidance model architectures and training utilities
that can be used with the custom guidance framework in sampling.py.

A guidance model learns to predict corrections/adjustments to the base diffusion
model's predictions at each timestep to achieve desired objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any


class BaseGuidanceModel(nn.Module):
    """
    Base class for guidance models.
    
    All guidance models should inherit from this and implement the forward method.
    The forward method receives the current denoising state and returns a guidance signal.
    """
    
    def forward(
        self,
        img: Tensor,
        pred: Tensor,
        txt: Tensor,
        vec: Tensor,
        timestep: Tensor,
        step_idx: int,
    ) -> Tensor | Dict[str, Tensor]:
        """
        Args:
            img: Current latent image [batch, seq_len, channels]
            pred: Base model prediction [batch, seq_len, channels]
            txt: Text embeddings [batch, txt_seq_len, txt_dim]
            vec: CLIP pooled embeddings [batch, vec_dim]
            timestep: Current timestep value [batch]
            step_idx: Integer step index in the denoising process
        
        Returns:
            Guidance signal to add to pred, same shape as pred [batch, seq_len, channels]
            OR dictionary with 'guidance' key and optional other outputs
        """
        raise NotImplementedError


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
        
        # Combine with conditioning
        self.combine = nn.Linear(hidden_dim * 2 + condition_dim, hidden_dim)
        
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
        pred: Tensor,
        txt: Tensor,
        vec: Tensor,
        timestep: Tensor,
        step_idx: int,
    ) -> Tensor:
        batch_size, seq_len, channels = img.shape
        
        # Process timestep
        if self.use_timestep_embedding:
            t_embed = self.timestep_embed(timestep.unsqueeze(-1))  # [batch, embed_dim]
        else:
            t_embed = timestep.unsqueeze(-1)  # [batch, 1]
        
        # Combine vec and timestep conditioning
        condition = torch.cat([vec, t_embed], dim=-1)  # [batch, condition_dim]
        condition = condition.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, condition_dim]
        
        # Process img and pred
        img_features = self.img_proj(img)  # [batch, seq_len, hidden_dim]
        pred_features = self.pred_proj(pred)  # [batch, seq_len, hidden_dim]
        
        # Combine all features
        combined = torch.cat([img_features, pred_features, condition], dim=-1)
        combined = self.combine(combined)  # [batch, seq_len, hidden_dim]
        
        # Process through MLP
        features = self.mlp(combined)  # [batch, seq_len, hidden_dim]
        
        # Project to output
        guidance = self.output_proj(features)  # [batch, seq_len, channels]
        
        return guidance


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
        pred: Tensor,
        txt: Tensor,
        vec: Tensor,
        timestep: Tensor,
        step_idx: int,
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
        
        # Combine text conditioning
        context = torch.cat([vec_feat, txt_feat], dim=1)  # [batch, 1+txt_seq_len, hidden_dim]
        
        # Process through attention layers
        for layer in self.layers:
            x = layer(x, context)
        
        # Project to output
        guidance = self.output_proj(x)  # [batch, seq_len, channels]
        
        return guidance


class AttentionGuidanceLayer(nn.Module):
    """Single layer of attention-based guidance processing."""
    
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
    
    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        # Self-attention
        x = x + self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False
        )[0]
        
        # Cross-attention with text conditioning
        x = x + self.cross_attn(
            self.norm2(x), context, context, need_weights=False
        )[0]
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        
        return x


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
        pred: Tensor,
        txt: Tensor,
        vec: Tensor,
        timestep: Tensor,
        step_idx: int,
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
        
        # Combine img and pred
        x = torch.cat([img_spatial, pred_spatial], dim=1)  # [batch, 2*in_channels, h, w]
        
        # Initial conv
        x = self.input_conv(x)  # [batch, hidden_channels, h, w]
        
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


class GuidanceModelTrainer:
    """
    Utility class for training guidance models.
    
    Example training loop for a guidance model that learns to improve
    samples based on some reward/objective function.
    """
    
    def __init__(
        self,
        guidance_model: BaseGuidanceModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        self.guidance_model = guidance_model
        self.optimizer = optimizer
        self.device = device
        
        self.guidance_model.to(device)
    
    def train_step(
        self,
        img: Tensor,
        pred: Tensor,
        txt: Tensor,
        vec: Tensor,
        timestep: Tensor,
        step_idx: int,
        target_correction: Tensor,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            img, pred, txt, vec, timestep, step_idx: Current denoising state
            target_correction: Target guidance signal (e.g., from a reward model)
        
        Returns:
            Dictionary with loss values
        """
        self.guidance_model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        guidance = self.guidance_model(
            img=img.to(self.device),
            pred=pred.to(self.device),
            txt=txt.to(self.device),
            vec=vec.to(self.device),
            timestep=timestep.to(self.device),
            step_idx=step_idx,
        )
        
        # Compute loss
        loss = F.mse_loss(guidance, target_correction.to(self.device))
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def train_from_reward(
        self,
        img: Tensor,
        pred: Tensor,
        txt: Tensor,
        vec: Tensor,
        timestep: Tensor,
        step_idx: int,
        reward_fn: callable,
        num_guidance_steps: int = 10,
        guidance_lr: float = 0.1,
    ) -> Dict[str, float]:
        """
        Train using reward-based guidance.
        
        The guidance model learns to maximize a reward function by
        modifying the prediction.
        
        Args:
            reward_fn: Function that takes modified prediction and returns scalar reward
            num_guidance_steps: Number of gradient steps for guidance optimization
            guidance_lr: Learning rate for guidance updates
        """
        self.guidance_model.train()
        
        total_reward = 0.0
        
        for _ in range(num_guidance_steps):
            self.optimizer.zero_grad()
            
            # Get guidance
            guidance = self.guidance_model(
                img=img.to(self.device),
                pred=pred.to(self.device),
                txt=txt.to(self.device),
                vec=vec.to(self.device),
                timestep=timestep.to(self.device),
                step_idx=step_idx,
            )
            
            # Apply guidance to prediction
            modified_pred = pred + guidance
            
            # Compute reward (negative for minimization)
            reward = reward_fn(modified_pred)
            loss = -reward  # Maximize reward = minimize negative reward
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_reward += reward.item()
        
        return {
            'avg_reward': total_reward / num_guidance_steps,
            'final_loss': loss.item()
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.guidance_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.guidance_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# Example usage functions
def create_mlp_guidance_model(**kwargs) -> MLPGuidanceModel:
    """Create an MLP guidance model with default or custom parameters."""
    return MLPGuidanceModel(**kwargs)


def create_attention_guidance_model(**kwargs) -> AttentionGuidanceModel:
    """Create an attention-based guidance model with default or custom parameters."""
    return AttentionGuidanceModel(**kwargs)


def create_conv_guidance_model(**kwargs) -> ConvGuidanceModel:
    """Create a convolutional guidance model with default or custom parameters."""
    return ConvGuidanceModel(**kwargs)
