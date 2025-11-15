"""
Few-Shot Guidance Framework

A framework for training and using guidance models for few-shot image generation.
"""

from .checkpoint_utils import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
)
from .train import GuidanceTrainer

__all__ = [
    'CheckpointManager',
    'save_checkpoint',
    'load_checkpoint',
    'GuidanceTrainer',
]

__version__ = '0.1.0'
