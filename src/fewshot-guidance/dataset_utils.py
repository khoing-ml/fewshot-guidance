"""
Dataset utilities for processing fewshot guidance datasets.

This module provides utilities for:
- Loading image datasets from directories
- Creating fewshot batches and pairs
- Processing and encoding images with the VAE
- Creating PyTorch datasets and dataloaders for training
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
import random
from einops import rearrange
import json


class FewshotImageDataset(Dataset):
    """
    Dataset for loading fewshot reference images from a directory structure.
    
    Expected directory structure:
        dataset_root/
            class1/
                img1.jpg
                img2.png
                ...
            class2/
                img1.jpg
                ...
    
    Or flat structure:
        dataset_root/
            img1.jpg
            img2.png
            ...
    """
    
    def __init__(
        self,
        root_dir: str | Path,
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.webp'],
        height: int = 1024,
        width: int = 1024,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            root_dir: Root directory containing images
            extensions: List of valid image file extensions
            height: Target height for images
            width: Target width for images
            transform: Optional transform function to apply to images
        """
        self.root_dir = Path(root_dir)
        self.height = height
        self.width = width
        self.transform = transform
        
        # Gather all image paths
        self.image_paths = []
        self.class_labels = []
        self.class_to_idx = {}
        
        if not self.root_dir.exists():
            raise ValueError(f"Dataset root directory does not exist: {root_dir}")
        
        # Check for class-based directory structure
        subdirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            # Class-based structure
            for class_idx, class_dir in enumerate(sorted(subdirs)):
                class_name = class_dir.name
                self.class_to_idx[class_name] = class_idx
                
                for ext in extensions:
                    for img_path in class_dir.glob(f"*{ext}"):
                        self.image_paths.append(img_path)
                        self.class_labels.append(class_idx)
        else:
            # Flat structure - all images in one class
            for ext in extensions:
                for img_path in self.root_dir.glob(f"*{ext}"):
                    self.image_paths.append(img_path)
                    self.class_labels.append(0)
            self.class_to_idx = {'default': 0}
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir} with extensions {extensions}")
        
        print(f"Found {len(self.image_paths)} images in {len(self.class_to_idx)} classes")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            dict with keys:
                - 'image': Preprocessed image tensor [3, H, W] normalized to [-1, 1]
                - 'path': Path to the image file
                - 'class': Class index
        """
        img_path = self.image_paths[idx]
        class_idx = self.class_labels[idx]
        
        # Load and preprocess image
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize to [-1, 1]
        img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [3, H, W]
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return {
            'image': img_tensor,
            'path': str(img_path),
            'class': class_idx,
        }
    
    def get_class_samples(self, class_idx: int, num_samples: int = None) -> List[int]:
        """
        Get indices of samples belonging to a specific class.
        
        Args:
            class_idx: Class index
            num_samples: Number of samples to return (None = all)
        
        Returns:
            List of dataset indices
        """
        indices = [i for i, c in enumerate(self.class_labels) if c == class_idx]
        
        if num_samples is not None:
            indices = random.sample(indices, min(num_samples, len(indices)))
        
        return indices


class FewshotLatentDataset(Dataset):
    """
    Dataset that stores pre-encoded latent representations of images.
    Useful for faster training when you pre-encode your dataset.
    """
    
    def __init__(
        self,
        latents: Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            latents: Pre-encoded latents [N, seq_len, channels]
            metadata: Optional metadata (classes, paths, etc.)
        """
        self.latents = latents
        self.metadata = metadata or {}
    
    def __len__(self) -> int:
        return len(self.latents)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        result = {'latent': self.latents[idx]}
        
        # Add metadata if available
        for key, values in self.metadata.items():
            if isinstance(values, (list, tuple)) and len(values) > idx:
                result[key] = values[idx]
        
        return result
    
    def save(self, path: str):
        """Save latent dataset to disk."""
        torch.save({
            'latents': self.latents,
            'metadata': self.metadata,
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """Load latent dataset from disk."""
        data = torch.load(path, map_location=device)
        return cls(data['latents'], data.get('metadata'))


class FewshotPairDataset(Dataset):
    """
    Dataset that creates (query, fewshot_references) pairs for training.
    Each sample returns a query image and K fewshot reference images.
    """
    
    def __init__(
        self,
        base_dataset: FewshotImageDataset,
        num_shots: int = 5,
        same_class: bool = True,
    ):
        """
        Args:
            base_dataset: Underlying image dataset
            num_shots: Number of fewshot reference images per query
            same_class: If True, fewshot examples come from same class as query
        """
        self.base_dataset = base_dataset
        self.num_shots = num_shots
        self.same_class = same_class
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get query image
        query_data = self.base_dataset[idx]
        query_class = query_data['class']
        
        # Sample fewshot references
        if self.same_class and len(self.base_dataset.class_to_idx) > 1:
            # Sample from same class, excluding the query itself
            class_indices = self.base_dataset.get_class_samples(query_class)
            class_indices = [i for i in class_indices if i != idx]
            
            if len(class_indices) < self.num_shots:
                # Not enough samples in class, sample with replacement
                fewshot_indices = random.choices(class_indices, k=self.num_shots)
            else:
                fewshot_indices = random.sample(class_indices, self.num_shots)
        else:
            # Sample from entire dataset, excluding query
            all_indices = [i for i in range(len(self.base_dataset)) if i != idx]
            fewshot_indices = random.sample(all_indices, min(self.num_shots, len(all_indices)))
        
        # Get fewshot images
        fewshot_images = []
        fewshot_paths = []
        for fs_idx in fewshot_indices:
            fs_data = self.base_dataset[fs_idx]
            fewshot_images.append(fs_data['image'])
            fewshot_paths.append(fs_data['path'])
        
        fewshot_images = torch.stack(fewshot_images)  # [num_shots, 3, H, W]
        
        return {
            'query_image': query_data['image'],  # [3, H, W]
            'query_path': query_data['path'],
            'query_class': query_class,
            'fewshot_images': fewshot_images,  # [num_shots, 3, H, W]
            'fewshot_paths': fewshot_paths,
        }


def encode_dataset_to_latents(
    dataset: FewshotImageDataset,
    ae,
    device: torch.device,
    batch_size: int = 4,
    verbose: bool = True,
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    Encode an entire dataset to latent representations using the VAE.
    
    Args:
        dataset: Image dataset to encode
        ae: Autoencoder model
        device: Device to use for encoding
        batch_size: Batch size for encoding
        verbose: Print progress
    
    Returns:
        latents: [N, seq_len, channels]
        metadata: Dictionary with paths, classes, etc.
    """
    ae.eval()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    all_latents = []
    all_paths = []
    all_classes = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if verbose and batch_idx % 10 == 0:
                print(f"Encoding batch {batch_idx}/{len(dataloader)}")
            
            images = batch['image'].to(device).to(torch.bfloat16)
            
            # Encode
            latent = ae.encode(images)
            
            # Pack the latent: [B, C, H, W] -> [B, seq_len, packed_channels]
            latent = rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            
            all_latents.append(latent.cpu())
            all_paths.extend(batch['path'])
            all_classes.extend(batch['class'].tolist())
    
    # Concatenate all batches
    all_latents = torch.cat(all_latents, dim=0)
    
    metadata = {
        'paths': all_paths,
        'classes': all_classes,
        'class_to_idx': dataset.class_to_idx,
        'height': dataset.height,
        'width': dataset.width,
    }
    
    if verbose:
        print(f"Encoded {len(all_latents)} images to latents of shape {all_latents.shape}")
    
    return all_latents, metadata


def create_fewshot_batch(
    indices: List[int],
    dataset: FewshotImageDataset | FewshotLatentDataset,
    num_shots: int = 5,
    same_class: bool = True,
) -> Dict[str, Tensor]:
    """
    Create a batch of fewshot examples.
    
    Args:
        indices: Indices of query images
        dataset: Source dataset
        num_shots: Number of fewshot references per query
        same_class: Whether to sample from same class
    
    Returns:
        Dictionary with query and fewshot data
    """
    queries = []
    fewshot_sets = []
    
    for idx in indices:
        query_data = dataset[idx]
        
        if isinstance(dataset, FewshotImageDataset):
            queries.append(query_data['image'])
            query_class = query_data['class']
            
            # Sample fewshot examples
            if same_class:
                class_indices = dataset.get_class_samples(query_class)
                class_indices = [i for i in class_indices if i != idx]
                fewshot_indices = random.sample(class_indices, min(num_shots, len(class_indices)))
            else:
                all_indices = [i for i in range(len(dataset)) if i != idx]
                fewshot_indices = random.sample(all_indices, num_shots)
            
            fewshot_images = [dataset[i]['image'] for i in fewshot_indices]
            fewshot_sets.append(torch.stack(fewshot_images))
        
        else:  # FewshotLatentDataset
            queries.append(query_data['latent'])
            # For latent datasets, sample randomly
            all_indices = [i for i in range(len(dataset)) if i != idx]
            fewshot_indices = random.sample(all_indices, min(num_shots, len(all_indices)))
            fewshot_latents = [dataset[i]['latent'] for i in fewshot_indices]
            fewshot_sets.append(torch.stack(fewshot_latents))
    
    return {
        'queries': torch.stack(queries),
        'fewshots': torch.stack(fewshot_sets),
    }


def load_image_and_encode(
    image_path: str,
    ae,
    device: torch.device,
    height: int = 1024,
    width: int = 1024,
) -> Tensor:
    """
    Load a single image and encode it to latent space.
    
    Args:
        image_path: Path to image file
        ae: Autoencoder model
        device: Device to use
        height: Target height
        width: Target width
    
    Returns:
        Encoded latent [1, seq_len, channels]
    """
    # Load and preprocess
    img = Image.open(image_path).convert("RGB")
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img = np.array(img).astype(np.float32) / 127.5 - 1.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Encode
    with torch.no_grad():
        latent = ae.encode(img.to(torch.bfloat16))
        latent = rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    
    return latent


def batch_encode_images(
    image_paths: List[str],
    ae,
    device: torch.device,
    height: int = 1024,
    width: int = 1024,
    batch_size: int = 4,
) -> Tensor:
    """
    Batch encode multiple images to latent space.
    
    Args:
        image_paths: List of image file paths
        ae: Autoencoder model
        device: Device to use
        height: Target height
        width: Target width
        batch_size: Batch size for encoding
    
    Returns:
        Encoded latents [N, seq_len, channels]
    """
    ae.eval()
    all_latents = []
    
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for img_path in batch_paths:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((width, height), Image.Resampling.LANCZOS)
                img = np.array(img).astype(np.float32) / 127.5 - 1.0
                img = torch.from_numpy(img).permute(2, 0, 1)
                batch_images.append(img)
            
            batch_tensor = torch.stack(batch_images).to(device).to(torch.bfloat16)
            
            # Encode
            latent = ae.encode(batch_tensor)
            latent = rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            all_latents.append(latent.cpu())
    
    return torch.cat(all_latents, dim=0)


def collate_fewshot_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for fewshot pair datasets.
    
    Args:
        batch: List of samples from FewshotPairDataset
    
    Returns:
        Batched dictionary
    """
    query_images = torch.stack([item['query_image'] for item in batch])
    fewshot_images = torch.stack([item['fewshot_images'] for item in batch])
    
    return {
        'query_images': query_images,  # [B, 3, H, W]
        'fewshot_images': fewshot_images,  # [B, num_shots, 3, H, W]
        'query_paths': [item['query_path'] for item in batch],
        'query_classes': torch.tensor([item['query_class'] for item in batch]),
        'fewshot_paths': [item['fewshot_paths'] for item in batch],
    }


def create_fewshot_dataloader(
    dataset_path: str,
    num_shots: int = 5,
    batch_size: int = 1,
    same_class: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    """
    Convenience function to create a DataLoader for fewshot training.
    
    Args:
        dataset_path: Path to dataset directory
        num_shots: Number of fewshot examples per query
        batch_size: Batch size
        same_class: Sample fewshot from same class
        num_workers: Number of data loading workers
        shuffle: Shuffle the data
        **dataset_kwargs: Additional arguments for FewshotImageDataset
    
    Returns:
        DataLoader ready for training
    """
    # Create base dataset
    base_dataset = FewshotImageDataset(dataset_path, **dataset_kwargs)
    
    # Wrap in pair dataset
    pair_dataset = FewshotPairDataset(
        base_dataset,
        num_shots=num_shots,
        same_class=same_class,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        pair_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fewshot_batch,
    )
    
    return dataloader


# Example usage and utility functions

def print_dataset_stats(dataset: FewshotImageDataset):
    """Print statistics about the dataset."""
    print(f"\nDataset Statistics:")
    print(f"  Total images: {len(dataset)}")
    print(f"  Number of classes: {len(dataset.class_to_idx)}")
    print(f"  Classes: {list(dataset.class_to_idx.keys())}")
    
    # Count per class
    class_counts = {}
    for class_idx in dataset.class_labels:
        class_name = [k for k, v in dataset.class_to_idx.items() if v == class_idx][0]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"  Images per class:")
    for class_name, count in sorted(class_counts.items()):
        print(f"    {class_name}: {count}")


if __name__ == "__main__":
    # Example usage
    print("Example: Loading a fewshot dataset")
    
    # Create a dataset
    dataset = FewshotImageDataset(
        root_dir="path/to/your/dataset",
        height=1024,
        width=1024,
    )
    
    print_dataset_stats(dataset)
    
    # Create a dataloader with fewshot pairs
    dataloader = create_fewshot_dataloader(
        dataset_path="path/to/your/dataset",
        num_shots=5,
        batch_size=2,
        same_class=True,
    )
    
    # Iterate through batches
    for batch in dataloader:
        print(f"Query images: {batch['query_images'].shape}")
        print(f"Fewshot images: {batch['fewshot_images'].shape}")
        break
