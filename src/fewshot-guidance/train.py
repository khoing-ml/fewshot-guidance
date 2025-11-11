import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any

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
        
        loss = F.mse_loss(guidance, target_correction.to(self.device))
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
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
