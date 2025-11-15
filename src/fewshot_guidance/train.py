import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Callable
import numpy as np
from einops import rearrange
import tqdm
from datetime import datetime

from fewshot_guidance.dataset_utils import FewshotImageDataset, FewshotLatentDataset, FewshotPairDataset
from fewshot_guidance.checkpoint_utils import CheckpointManager


def encode_dataset_to_latents(
    dataset_dir: str | Path,
    ae_model: nn.Module,
    batch_size: int = 4,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    num_workers: int = 4,
    image_height: int = 512,
    image_width: int = 512,
    save_path: Optional[str | Path] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Load a dataset from disk and encode all images into latent representations using a ae.
    
    This function:
    1. Loads all images from the dataset directory
    2. Processes them in batches using the ae encoder
    3. Stores the encoded latents in memory or saves to disk
    4. Tracks metadata (image paths, classes) for later retrieval
    
    Args:
        dataset_dir: Path to dataset directory containing images
        ae_model: Pretrained ae model for encoding (should have .encode() method)
        batch_size: Batch size for encoding (default: 4)
        device: Device to run ae on (default: "cuda")
        dtype: Data type for computations (default: torch.float32)
        num_workers: Number of workers for data loading (default: 4)
        image_height: Target image height (default: 512)
        image_width: Target image width (default: 512)
        save_path: Optional path to save encoded latents. If None, returns latents in memory.
    
    Returns:
        Tuple of:
            - latents: Tensor of shape [N, latent_channels, latent_h, latent_w]
            - metadata: Dictionary containing:
                - 'image_paths': List of original image file paths
                - 'class_labels': List of class labels for each image
                - 'class_to_idx': Mapping of class names to indices
                - 'image_size': (height, width) of original images
                - 'latent_shape': Shape of each latent [channels, h, w]
    """
    
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    
    # Load image dataset
    print(f"Loading images from {dataset_dir}...")
    image_dataset = FewshotImageDataset(
        root_dir=dataset_dir,
        height=image_height,
        width=image_width,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
    )
    
    # Move ae to device and set to eval mode
    ae_model = ae_model.to(device)
    ae_model.eval()
    
    # Collect all latents
    all_latents = []
    image_paths = []
    class_labels = []
    
    print(f"Encoding {len(image_dataset)} images to latents...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Get images and metadata from batch
            images = batch['image'].to(device, dtype=dtype)  # [B, 3, H, W]
            paths = batch['path']
            classes = batch['class']
            
            # Latent shape is typically [B, latent_channels, latent_h, latent_w]
            latents = ae_model.encode(images)  # [B, latent_channels, latent_h, latent_w]
            
            # Store results
            all_latents.append(latents.cpu())
            image_paths.extend(paths)
            class_labels.extend(classes.cpu().tolist())
            
            if (batch_idx + 1) % max(1, len(dataloader) // 5) == 0:
                print(f"  Encoded {min((batch_idx + 1) * batch_size, len(image_dataset))}"
                      f"/{len(image_dataset)} images")
    
    # Concatenate all latents
    all_latents = torch.cat(all_latents, dim=0)
    latent_shape = all_latents.shape[1:]  # [latent_channels, latent_h, latent_w]
    
    print(f"✓ Successfully encoded {len(all_latents)} images")
    print(f"  Latent shape per image: {latent_shape}")
    print(f"  Total latent tensor shape: {all_latents.shape}")
    
    # Create metadata
    metadata = {
        'image_paths': image_paths,
        'class_labels': class_labels,
        'class_to_idx': image_dataset.class_to_idx,
        'image_size': (image_height, image_width),
        'latent_shape': list(latent_shape),
        'dataset_dir': str(dataset_dir),
    }
    
    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'latents': all_latents,
            'metadata': metadata,
        }, save_path)
        print(f"✓ Saved encoded latents to {save_path}")
    
    return all_latents, metadata


def encode_multiple_datasets(
    datasets_dir: str | Path,
    ae_model: nn.Module,
    batch_size: int = 4,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    num_workers: int = 4,
    image_height: int = 512,
    image_width: int = 512,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Encode multiple datasets from a parent directory.
    
    Useful for encoding all datasets in the 'datasets' folder at once.
    
    Args:
        datasets_dir: Parent directory containing subdirectories with datasets
        ae_model: Pretrained ae model for encoding
        batch_size: Batch size for encoding
        device: Device to run ae on
        dtype: Data type for computations
        num_workers: Number of workers for data loading
        image_height: Target image height
        image_width: Target image width
        output_dir: Directory to save encoded latents. Creates subdirectories for each dataset.
    
    Returns:
        Dictionary mapping dataset names to (latents, metadata) tuples
    """
    
    datasets_dir = Path(datasets_dir)
    if not datasets_dir.exists():
        raise ValueError(f"Datasets directory does not exist: {datasets_dir}")
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all dataset subdirectories
    dataset_dirs = [d for d in datasets_dir.iterdir() if d.is_dir()]
    
    if not dataset_dirs:
        raise ValueError(f"No dataset subdirectories found in {datasets_dir}")
    
    print(f"Found {len(dataset_dirs)} datasets to encode")
    
    results = {}
    
    for dataset_path in sorted(dataset_dirs):
        dataset_name = dataset_path.name
        print(f"\n{'='*60}")
        print(f"Encoding dataset: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            # Determine save path
            save_path = None
            if output_dir is not None:
                save_path = output_dir / f"{dataset_name}_latents.pt"
            
            # Encode dataset
            latents, metadata = encode_dataset_to_latents(
                dataset_dir=dataset_path,
                ae_model=ae_model,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                num_workers=num_workers,
                image_height=image_height,
                image_width=image_width,
                save_path=save_path,
            )
            
            results[dataset_name] = (latents, metadata)
            
        except Exception as e:
            print(f"✗ Error encoding {dataset_name}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Successfully encoded {len(results)}/{len(dataset_dirs)} datasets")
    print(f"{'='*60}")
    
    return results


def load_encoded_dataset(
    latent_path: str | Path,
    device: str = "cpu",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Load pre-encoded latents from disk.
    
    Args:
        latent_path: Path to saved latent file (created by encode_dataset_to_latents)
        device: Device to load tensors to
    
    Returns:
        Tuple of (latents, metadata)
    """
    latent_path = Path(latent_path)
    if not latent_path.exists():
        raise ValueError(f"Latent file does not exist: {latent_path}")
    
    data = torch.load(latent_path, map_location=device)
    return data['latents'], data['metadata']


class TrainingConfig:
    """Configuration for guidance model training."""
    
    def __init__(
        self,
        # Model
        guidance_model: nn.Module,
        flow_model: nn.Module,
        ae_model: nn.Module,
        
        # Data
        dataset_dir: str | Path,
        batch_size: int = 8,
        num_workers: int = 4,
        
        # Training
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        gradient_clip: float = 1.0,
        
        # Guidance sampling
        num_shots: int = 4,
        guidance_scale: float = 7.5,
        
        # Device and precision
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        
        # Checkpointing
        checkpoint_dir: str | Path = "./checkpoints",
        save_every_n_steps: int = 1000,
        
        # Logging
        log_every_n_steps: int = 10,
    ):
        self.guidance_model = guidance_model
        self.flow_model = flow_model
        self.ae_model = ae_model
        
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        
        self.num_shots = num_shots
        self.guidance_scale = guidance_scale
        
        self.device = device
        self.dtype = dtype
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every_n_steps = save_every_n_steps
        
        self.log_every_n_steps = log_every_n_steps


class GuidanceTrainer:
    """
    Trainer for fewshot guidance model following the training objective:
    
    min L = E[||ψ(X̂_t, t) + c_φ^t(t, X̂_t) - (X̂_1 - X̂_0)||²]
    
    Where:
    - X̂_t = t*X_0 + (1-t)*X_1  (interpolated latent)
    - X_0 is initial noise
    - X_1 is fewshot dataset image latent
    - ψ(·) is the base flow model prediction
    - c_φ^t(·) is the guidance model we're training
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        
        # Move models to device
        self.guidance_model = config.guidance_model.to(self.device).train()
        self.flow_model = config.flow_model.to(self.device).eval()
        self.ae_model = config.ae_model.to(self.device).eval()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.guidance_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(save_dir=config.checkpoint_dir)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            return 1.0
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def prepare_data(self) -> Tuple[DataLoader, Dict[str, Any]]:
        """Load and prepare training data."""
        print(f"Loading dataset from {self.config.dataset_dir}...")
        
        # Load fewshot image dataset
        image_dataset = FewshotImageDataset(
            root_dir=self.config.dataset_dir,
            height=512,
            width=512,
        )
        
        # Create fewshot pairs
        pair_dataset = FewshotPairDataset(
            base_dataset=image_dataset,
            num_shots=self.config.num_shots,
            same_class=True,
        )
        
        # Create dataloader
        dataloader = DataLoader(
            pair_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )
        
        metadata = {
            'num_images': len(image_dataset),
            'num_classes': len(image_dataset.class_to_idx),
            'classes': image_dataset.class_to_idx,
        }
        
        return dataloader, metadata
    
    def encode_batch_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        """Encode batch of images to ae latents."""
        with torch.no_grad():
            # Encode to latent space
            latents = self.ae_model.encode(images)  # [B, latent_channels, latent_h, latent_w]
            # Rearrange to [B, seq_len, channels] for guidance model
            if latents.dim() == 4:
                B, C, H, W = latents.shape
                latents = rearrange(latents, 'b c h w -> b (h w) c')
        return latents
    
    def sample_timestep(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample timestep t ~ Uniform[0, 1] and prepare interpolation coefficient.
        
        Returns:
            t: Timestep values [batch_size]
            alpha: Interpolation coefficient (same as t)
        """
        t = torch.rand(batch_size, device=self.device, dtype=self.dtype)
        return t, t
    
    def prepare_training_target(
        self,
        X0: torch.Tensor,
        X1: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare training pair:
        X̂_t = t*X_0 + (1-t)*X_1
        
        Returns:
            X_hat_t: Interpolated latent [batch, seq_len, channels]
            target: X_1 - X_0 (the guidance target) [batch, seq_len, channels]
        """
        t_expanded = t.view(-1, 1, 1)  # [batch, 1, 1]
        
        # Interpolate between noise (X0) and image (X1)
        X_hat_t = t_expanded * X0 + (1 - t_expanded) * X1
        
        # Training target: difference between image and noise
        target = X1 - X0
        
        return X_hat_t, target
    
    def get_flow_prediction(
        self,
        X_hat_t: torch.Tensor,
        t: torch.Tensor,
        conditioning: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Get prediction from Flux flow model."""
        with torch.no_grad():
            # X_hat_t shape: [B, seq_len, C]
            B, seq_len, C = X_hat_t.shape
            
            # Create img_ids
            img_ids = torch.arange(seq_len, device=self.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
            
            # Create txt_ids (max length 77 for CLIP)
            txt_len = 77
            txt_ids = torch.arange(txt_len, device=self.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
            
            pred = self.flow_model(
                img=X_hat_t,
                img_ids=img_ids,
                timesteps=t,
                txt=conditioning.get('txt') if conditioning else torch.zeros(B, txt_len, 4096, device=self.device, dtype=X_hat_t.dtype),
                txt_ids=txt_ids,
                y=conditioning.get('vec') if conditioning else torch.zeros(B, 768, device=self.device, dtype=X_hat_t.dtype),
            )
        return pred
    
    def training_step(
        self,
        query_image: torch.Tensor,
        fewshot_images: torch.Tensor,
        conditioning: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Single training step implementing:
        
        L = ||ψ(X̂_t, t) + c_φ^t(t, X̂_t) - (X̂_1 - X̂_0)||²
        
        Args:
            query_image: Query image [batch, 3, H, W] (not used for X1, kept for interface compatibility)
            fewshot_images: Fewshot reference images [batch, num_shots, 3, H, W]
            conditioning: Optional conditioning (text embeddings, etc.)
        
        Returns:
            Dictionary with loss values
        """
        batch_size = query_image.shape[0]
        
        # Process fewshot images and sample X1 from them
        if fewshot_images.dim() == 5:  # [batch, num_shots, 3, H, W]
            # Encode fewshot images
            B, S, C, H, W = fewshot_images.shape
            fewshot_flat = fewshot_images.view(B * S, C, H, W)
            fewshot_latents = self.encode_batch_to_latents(fewshot_flat)
            fewshot_latents = fewshot_latents.view(B, S, *fewshot_latents.shape[1:])
            
            # Sample one random fewshot image per batch as X1
            # Shape: [batch, seq_len, channels]
            shot_indices = torch.randint(0, S, (B,), device=self.device)
            X1 = fewshot_latents[torch.arange(B, device=self.device), shot_indices]
        else:
            # If fewshot_images is already [batch, seq_len, channels] or [batch, 3, H, W]
            fewshot_latents = self.encode_batch_to_latents(fewshot_images)
            X1 = fewshot_latents
        
        # Sample random noise as X0
        X0 = torch.randn_like(X1, device=self.device, dtype=self.dtype)
        
        # Sample timestep
        t, alpha = self.sample_timestep(batch_size)
        
        # Prepare interpolated latent and target
        X_hat_t, target = self.prepare_training_target(X0, X1, alpha)
        
        # Get base model prediction
        # For now, we use zero prediction since Flux expects different input format
        # In practice, you would want to use a properly formatted flow matching model
        pred = torch.zeros_like(X_hat_t)
        
        # Forward through guidance model
        # c_φ^t(t, X̂_t) without fewshot context
        guidance = self.guidance_model(
            img=X_hat_t,
            timestep=t,
            step_idx=int(alpha[0].item()),  # Use alpha as step indicator
            pred=pred,
            txt=conditioning.get('txt') if conditioning else None,
            vec=conditioning.get('vec') if conditioning else None,
        )
        
        # Handle guidance output (could be Tensor or Dict)
        if isinstance(guidance, dict):
            guidance = guidance.get('guidance', guidance.get('output'))

        # Loss: ||pred + guidance - target||²
        pred_with_guidance = pred + guidance
        loss = F.mse_loss(pred_with_guidance, target)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.guidance_model.parameters(),
                self.config.gradient_clip
            )
        
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'lr': self.optimizer.param_groups[0]['lr'],
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.guidance_model.train()
        epoch_losses = []
        
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Extract batch components
            query_image = batch['query_image'].to(self.device, dtype=self.dtype)
            fewshot_images = batch['fewshot_images'].to(self.device, dtype=self.dtype)
            
            # Optional conditioning
            conditioning = {}
            if 'txt' in batch:
                conditioning['txt'] = batch['txt'].to(self.device, dtype=self.dtype)
            if 'vec' in batch:
                conditioning['vec'] = batch['vec'].to(self.device, dtype=self.dtype)
            
            # Training step
            metrics = self.training_step(query_image, fewshot_images, conditioning)
            epoch_losses.append(metrics['loss'])
            
            # Logging
            if (self.global_step + 1) % self.config.log_every_n_steps == 0:
                avg_loss = np.mean(epoch_losses[-self.config.log_every_n_steps:])
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f"{metrics['lr']:.2e}"})
            
            # Checkpointing
            if (self.global_step + 1) % self.config.save_every_n_steps == 0:
                self.save_checkpoint()
            
            self.global_step += 1
        
        return {
            'epoch_loss': np.mean(epoch_losses),
        }
    
    def train(self):
        """Train the guidance model for multiple epochs."""
        # Prepare data
        dataloader, metadata = self.prepare_data()
        
        print(f"\n{'='*60}")
        print(f"Training Configuration")
        print(f"{'='*60}")
        print(f"Dataset: {self.config.dataset_dir}")
        print(f"Dataset size: {metadata['num_images']} images")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Num epochs: {self.config.num_epochs}")
        print(f"Learning rate: {self.config.learning_rate:.2e}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            epoch_metrics = self.train_epoch(dataloader)
            
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs} - "
                  f"Loss: {epoch_metrics['epoch_loss']:.4f}")
            
            # Save checkpoint at end of epoch
            self.save_checkpoint(epoch=epoch)
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Checkpoints saved to: {self.config.checkpoint_dir}")
        print(f"{'='*60}")
    
    def save_checkpoint(self, epoch: Optional[int] = None):
        """Save model checkpoint."""
        checkpoint_name = self.checkpoint_manager.save_checkpoint(
            model=self.guidance_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            step=self.global_step,
            metrics={'last_loss': 0.0},
        )
        print(f"✓ Checkpoint saved to {checkpoint_name}")
    
    def load_checkpoint(self, checkpoint_path: str | Path):
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.guidance_model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']
        
        if 'step' in checkpoint:
            self.global_step = checkpoint['step']
        
        print(f"✓ Checkpoint loaded from {checkpoint_path}")


if __name__ == "__main__":
    # Example usage
    print("Fewshot Guidance Training Module")
    print("This module provides utilities for training guidance models.")
    print("\nKey classes:")
    print("  - TrainingConfig: Configuration for training")
    print("  - GuidanceTrainer: Main trainer class implementing the training loop")
    print("\nKey functions:")
    print("  - encode_dataset_to_latents(): Encode single dataset to latents")
    print("  - encode_multiple_datasets(): Encode all datasets in a directory")
    print("  - load_encoded_dataset(): Load pre-encoded latents from disk")
