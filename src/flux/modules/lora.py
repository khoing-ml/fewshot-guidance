"""
LoRA (Low-Rank Adaptation) implementation for Flux models.

This module provides LoRA layers and utilities for fine-tuning Flux models
with low-rank adaptations.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Iterator


class LinearLora(nn.Module):
    """
    LoRA-adapted Linear layer.
    
    Implements low-rank adaptation for linear layers by adding
    trainable low-rank matrices to the original weights.
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, scale: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = scale
        
        # Original linear layer
        self.linear = nn.Linear(in_features, out_features, bias=True)
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        
    def forward(self, x: Tensor) -> Tensor:
        # Original forward pass
        out = self.linear(x)
        
        # LoRA adaptation
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
        out = out + self.scale * lora_out
        
        return out
    
    def set_scale(self, scale: float) -> None:
        """Set the LoRA scaling factor."""
        self.scale = scale


def replace_linear_with_lora(
    model: nn.Module,
    max_rank: int = 16,
    scale: float = 1.0,
) -> None:
    """
    Replace all Linear layers in a model with LinearLora layers.
    
    Args:
        model: The model to modify
        max_rank: Maximum rank for LoRA layers
        scale: Scaling factor for LoRA adaptations
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Replace with LoRA version
            lora_layer = LinearLora(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=min(max_rank, min(module.in_features, module.out_features)),
                scale=scale
            )
            
            # Copy weights and bias
            lora_layer.linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_layer.linear.bias.data = module.bias.data.clone()
            
            # Replace in parent module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, lora_layer)
            else:
                setattr(model, child_name, lora_layer)