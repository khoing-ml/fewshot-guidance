"""
Utilities for saving checkpoints and uploading models to Hugging Face Hub.

This module provides functions to:
- Save model checkpoints with metadata
- Load checkpoints
- Push models to Hugging Face Hub
- Create model cards automatically
"""

import os
import json
import torch
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime


class CheckpointManager:
    """
    Manager for saving, loading, and uploading model checkpoints.
    
    Example usage:
        >>> manager = CheckpointManager(save_dir="./checkpoints")
        >>> manager.save_checkpoint(
        ...     model=guidance_model,
        ...     optimizer=optimizer,
        ...     epoch=10,
        ...     metrics={'loss': 0.05}
        ... )
        >>> manager.push_to_hub(
        ...     repo_id="username/my-guidance-model",
        ...     checkpoint_name="checkpoint_epoch_10.pt"
        ... )
    """
    
    def __init__(
        self,
        save_dir: str = "./checkpoints",
        keep_last_n: Optional[int] = None,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            keep_last_n: If set, only keep the last N checkpoints (auto-cleanup)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.checkpoint_history = []
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_name: Optional[str] = None,
    ) -> str:
        """
        Save a checkpoint with model state and training metadata.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state (optional)
            scheduler: Learning rate scheduler state (optional)
            epoch: Current epoch number
            step: Current training step
            metrics: Dictionary of metrics (e.g., {'loss': 0.5, 'accuracy': 0.95})
            metadata: Additional metadata to save
            checkpoint_name: Custom checkpoint filename (auto-generated if None)
        
        Returns:
            Path to saved checkpoint
        """
        # Generate checkpoint name if not provided
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            parts = [timestamp]
            if epoch is not None:
                parts.append(f"epoch_{epoch}")
            if step is not None:
                parts.append(f"step_{step}")
            checkpoint_name = "_".join(parts) + ".pt"
        
        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'step': step,
            'timestamp': datetime.now().isoformat(),
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        # Save checkpoint
        checkpoint_path = self.save_dir / checkpoint_name
        torch.save(checkpoint, checkpoint_path)
        
        # Track checkpoint history
        self.checkpoint_history.append(checkpoint_path)
        
        # Auto-cleanup old checkpoints if keep_last_n is set
        if self.keep_last_n is not None and len(self.checkpoint_history) > self.keep_last_n:
            old_checkpoint = self.checkpoint_history.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        print(f"✓ Checkpoint saved to: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint and restore model/optimizer/scheduler states.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load checkpoint to
            strict: Whether to strictly enforce state dict keys match
        
        Returns:
            Dictionary containing checkpoint metadata (epoch, step, metrics, etc.)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Load optimizer state if available
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Extract metadata
        metadata = {
            'epoch': checkpoint.get('epoch'),
            'step': checkpoint.get('step'),
            'metrics': checkpoint.get('metrics'),
            'timestamp': checkpoint.get('timestamp'),
            'metadata': checkpoint.get('metadata'),
        }
        
        print(f"✓ Checkpoint loaded from: {checkpoint_path}")
        if metadata['epoch'] is not None:
            print(f"  Epoch: {metadata['epoch']}")
        if metadata['step'] is not None:
            print(f"  Step: {metadata['step']}")
        if metadata['metrics'] is not None:
            print(f"  Metrics: {metadata['metrics']}")
        
        return metadata
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the most recent checkpoint."""
        checkpoints = sorted(self.save_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        return str(checkpoints[-1]) if checkpoints else None
    
    def push_to_hub(
        self,
        repo_id: str,
        checkpoint_path: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        model_card: Optional[str] = None,
        private: bool = False,
        token: Optional[str] = None,
        commit_message: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Push a checkpoint or model to Hugging Face Hub.
        
        Args:
            repo_id: Repository ID on Hugging Face Hub (e.g., "username/model-name")
            checkpoint_path: Path to checkpoint file to upload (uses latest if None)
            model: If provided, save this model directly instead of using checkpoint_path
            model_card: Custom model card content (auto-generated if None)
            private: Whether to make the repository private
            token: Hugging Face API token (uses HF_TOKEN env var if None)
            commit_message: Custom commit message
            config: Model configuration dictionary to save as config.json
        
        Returns:
            URL of the uploaded model on Hugging Face Hub
        """
        try:
            from huggingface_hub import HfApi, create_repo
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to push to hub. "
                "Install it with: pip install huggingface-hub"
            )
        
        # Get API token
        if token is None:
            token = os.environ.get("HF_TOKEN")
        
        # Initialize API
        api = HfApi(token=token)
        
        # Create repository if it doesn't exist
        try:
            create_repo(
                repo_id=repo_id,
                private=private,
                token=token,
                exist_ok=True,
            )
        except Exception as e:
            print(f"Repository creation info: {e}")
        
        # Create temporary directory for hub files
        temp_dir = self.save_dir / "hub_temp"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Save model
            if model is not None:
                model_path = temp_dir / "model.pt"
                torch.save(model.state_dict(), model_path)
            elif checkpoint_path is not None:
                model_path = Path(checkpoint_path)
                # Copy to temp dir with standard name
                shutil.copy(model_path, temp_dir / "checkpoint.pt")
                model_path = temp_dir / "checkpoint.pt"
            else:
                # Use latest checkpoint
                latest = self.get_latest_checkpoint()
                if latest is None:
                    raise ValueError("No checkpoint found to upload")
                model_path = Path(latest)
                shutil.copy(model_path, temp_dir / "checkpoint.pt")
                model_path = temp_dir / "checkpoint.pt"
            
            # Save config if provided
            if config is not None:
                config_path = temp_dir / "config.json"
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            
            # Generate model card if not provided
            if model_card is None:
                model_card = self._generate_model_card(repo_id, config)
            
            readme_path = temp_dir / "README.md"
            with open(readme_path, 'w') as f:
                f.write(model_card)
            
            # Upload to hub
            if commit_message is None:
                commit_message = f"Upload checkpoint from {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            api.upload_folder(
                folder_path=str(temp_dir),
                repo_id=repo_id,
                commit_message=commit_message,
                token=token,
            )
            
            url = f"https://huggingface.co/{repo_id}"
            print(f"✓ Model uploaded to: {url}")
            return url
            
        finally:
            # Cleanup temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _generate_model_card(self, repo_id: str, config: Optional[Dict[str, Any]] = None) -> str:
        """Generate a basic model card for the Hugging Face Hub."""
        model_name = repo_id.split('/')[-1]
        
        card = f"""---
license: apache-2.0
tags:
- guidance-model
- few-shot-guidance
- image-generation
- pytorch
---

# {model_name}

This is a guidance model for few-shot image generation, trained to provide guidance signals during the diffusion sampling process.

## Model Description

This model learns to generate guidance signals that can improve sample quality based on few-shot examples.

## Usage

```python
import torch
from fewshot_guidance import CheckpointManager

# Load the model
manager = CheckpointManager()
model = YourGuidanceModel()  # Initialize your model architecture
manager.load_checkpoint("checkpoint.pt", model)

# Use for inference
with torch.no_grad():
    guidance = model(img, pred, txt, vec, timestep, step_idx)
```

## Training Details

"""
        
        if config is not None:
            card += "\n### Model Configuration\n\n```json\n"
            card += json.dumps(config, indent=2)
            card += "\n```\n"
        
        card += f"""
## Citation

If you use this model, please cite:

```bibtex
@misc{{{model_name.replace('-', '_')},
  author = {{Your Name}},
  title = {{{model_name}}},
  year = {{{datetime.now().year}}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{repo_id}}}}}
}}
```

## License

This model is released under the Apache-2.0 license.
"""
        
        return card


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    **kwargs
) -> None:
    """
    Simple function to save a checkpoint (convenience wrapper).
    
    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save (optional)
        **kwargs: Additional data to save in checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, path)
    print(f"✓ Checkpoint saved to: {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Simple function to load a checkpoint (convenience wrapper).
    
    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        device: Device to load to
        strict: Whether to strictly enforce state dict keys match
    
    Returns:
        Checkpoint dictionary
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Checkpoint loaded from: {path}")
    return checkpoint
