"""
Few-Shot Guidance Framework

A framework for training and using guidance models for few-shot image generation.
"""

from .checkpoint_utils import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
)
from .train import GuidanceModelTrainer

__all__ = [
    'CheckpointManager',
    'save_checkpoint',
    'load_checkpoint',
    'GuidanceModelTrainer',
]

__version__ = '0.1.0'
