import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any
from .checkpoint_utils import CheckpointManager, save_checkpoint, load_checkpoint
from model.base_model import BaseGuidanceModel
from model.mlp_model import MLPGuidanceModel
from model.attention_based_model import AttentionGuidanceModel
from model.conv_model import ConvGuidanceModel


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
        checkpoint_dir: str = "./checkpoints",
        keep_last_n_checkpoints: Optional[int] = 5,
    ):
        self.guidance_model = guidance_model
        self.optimizer = optimizer
        self.device = device
        
        self.guidance_model.to(device)
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=checkpoint_dir,
            keep_last_n=keep_last_n_checkpoints,
        )
    
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
    
    def save_checkpoint(
        self,
        path: Optional[str] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> str:
        """
        Save model checkpoint with enhanced metadata.
        
        Args:
            path: Custom checkpoint filename (auto-generated if None)
            epoch: Current epoch number
            step: Current training step
            metrics: Dictionary of metrics (e.g., {'loss': 0.5})
            **kwargs: Additional metadata to save
        
        Returns:
            Path to saved checkpoint
        """
        return self.checkpoint_manager.save_checkpoint(
            model=self.guidance_model,
            optimizer=self.optimizer,
            epoch=epoch,
            step=step,
            metrics=metrics,
            checkpoint_name=path,
            metadata=kwargs,
        )
    
    def load_checkpoint(
        self,
        path: str,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            strict: Whether to strictly enforce state dict keys match
        
        Returns:
            Dictionary containing checkpoint metadata
        """
        return self.checkpoint_manager.load_checkpoint(
            checkpoint_path=path,
            model=self.guidance_model,
            optimizer=self.optimizer,
            device=self.device,
            strict=strict,
        )
    
    def push_to_hub(
        self,
        repo_id: str,
        model_card: Optional[str] = None,
        private: bool = False,
        token: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Push the current model to Hugging Face Hub.
        
        Args:
            repo_id: Repository ID on HF Hub (e.g., "username/model-name")
            model_card: Custom model card content
            private: Whether to make the repository private
            token: Hugging Face API token
            config: Model configuration to save
        
        Returns:
            URL of the uploaded model
        """
        return self.checkpoint_manager.push_to_hub(
            repo_id=repo_id,
            model=self.guidance_model,
            model_card=model_card,
            private=private,
            token=token,
            config=config,
        )


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
