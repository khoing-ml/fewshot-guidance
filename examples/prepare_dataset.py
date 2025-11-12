"""
Example script for preparing and encoding fewshot datasets.

This shows how to:
1. Load a dataset from a directory
2. Encode images to latents for faster training
3. Save and load pre-encoded latents
4. Create dataloaders for training
"""

import torch
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fewshot_guidance.dataset_utils import (
    FewshotImageDataset,
    FewshotLatentDataset,
    FewshotPairDataset,
    encode_dataset_to_latents,
    create_fewshot_dataloader,
    print_dataset_stats,
)
from flux.util import load_ae


def prepare_image_dataset(args):
    """Load and inspect an image dataset."""
    print(f"\n{'='*60}")
    print("LOADING IMAGE DATASET")
    print(f"{'='*60}")
    
    dataset = FewshotImageDataset(
        root_dir=args.dataset_path,
        height=args.height,
        width=args.width,
    )
    
    print_dataset_stats(dataset)
    
    # Show a few samples
    print(f"\nFirst 3 samples:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"  [{i}] {Path(sample['path']).name} - class {sample['class']} - shape {sample['image'].shape}")
    
    return dataset


def encode_and_save_latents(args, dataset):
    """Encode the dataset to latents and save to disk."""
    print(f"\n{'='*60}")
    print("ENCODING DATASET TO LATENTS")
    print(f"{'='*60}")
    
    device = torch.device(args.device)
    
    # Load autoencoder
    print(f"\nLoading autoencoder...")
    ae = load_ae(args.model, device=device)
    
    # Encode dataset
    print(f"\nEncoding {len(dataset)} images...")
    latents, metadata = encode_dataset_to_latents(
        dataset,
        ae,
        device,
        batch_size=args.batch_size,
        verbose=True,
    )
    
    print(f"\nEncoded latents shape: {latents.shape}")
    print(f"Latent size: {latents.element_size() * latents.nelement() / 1024 / 1024:.2f} MB")
    
    # Create latent dataset
    latent_dataset = FewshotLatentDataset(latents, metadata)
    
    # Save to disk
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving latent dataset to {output_path}")
        latent_dataset.save(str(output_path))
        print(f"✓ Saved successfully")
    
    return latent_dataset


def load_latent_dataset(args):
    """Load a pre-encoded latent dataset from disk."""
    print(f"\n{'='*60}")
    print("LOADING PRE-ENCODED LATENTS")
    print(f"{'='*60}")
    
    device = torch.device(args.device)
    latent_dataset = FewshotLatentDataset.load(args.latent_path, device=device)
    
    print(f"Loaded {len(latent_dataset)} latents")
    print(f"Latent shape: {latent_dataset.latents.shape}")
    
    if 'classes' in latent_dataset.metadata:
        num_classes = len(set(latent_dataset.metadata['classes']))
        print(f"Number of classes: {num_classes}")
    
    return latent_dataset


def create_training_dataloader(args, dataset):
    """Create a dataloader for fewshot training."""
    print(f"\n{'='*60}")
    print("CREATING TRAINING DATALOADER")
    print(f"{'='*60}")
    
    # Create pair dataset
    pair_dataset = FewshotPairDataset(
        base_dataset=dataset,
        num_shots=args.num_shots,
        same_class=args.same_class,
    )
    
    print(f"\nDataset configuration:")
    print(f"  Total samples: {len(pair_dataset)}")
    print(f"  Fewshot examples per query: {args.num_shots}")
    print(f"  Same class sampling: {args.same_class}")
    
    # Create dataloader using the convenience function
    dataloader = create_fewshot_dataloader(
        dataset_path=args.dataset_path,
        num_shots=args.num_shots,
        batch_size=args.train_batch_size,
        same_class=args.same_class,
        shuffle=True,
        height=args.height,
        width=args.width,
    )
    
    print(f"\nDataloader configuration:")
    print(f"  Batch size: {args.train_batch_size}")
    print(f"  Number of batches: {len(dataloader)}")
    
    # Show example batch
    print(f"\nExample batch:")
    for batch in dataloader:
        print(f"  Query images: {batch['query_images'].shape}")
        print(f"  Fewshot images: {batch['fewshot_images'].shape}")
        print(f"  Query classes: {batch['query_classes']}")
        break
    
    return dataloader


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and encode fewshot datasets"
    )
    
    # Dataset paths
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for encoded latents (.pt file)")
    parser.add_argument("--latent-path", type=str, default=None,
                       help="Path to pre-encoded latent dataset (to load)")
    
    # Model settings
    parser.add_argument("--model", type=str, default="flux-schnell",
                       choices=["flux-schnell", "flux-dev"],
                       help="Model variant")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    # Image settings
    parser.add_argument("--height", type=int, default=1024,
                       help="Image height")
    parser.add_argument("--width", type=int, default=1024,
                       help="Image width")
    
    # Encoding settings
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for encoding")
    
    # Training dataloader settings
    parser.add_argument("--num-shots", type=int, default=5,
                       help="Number of fewshot examples")
    parser.add_argument("--same-class", action="store_true",
                       help="Sample fewshot examples from same class")
    parser.add_argument("--train-batch-size", type=int, default=1,
                       help="Batch size for training dataloader")
    
    # Mode
    parser.add_argument("--mode", type=str, default="encode",
                       choices=["encode", "load", "dataloader"],
                       help="Mode: encode (create latents), load (load latents), dataloader (create dataloader)")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("FEWSHOT DATASET PREPARATION")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Device: {args.device}")
    print(f"Image size: {args.height}x{args.width}")
    
    if args.mode == "encode":
        # Load image dataset
        dataset = prepare_image_dataset(args)
        
        # Encode and save
        latent_dataset = encode_and_save_latents(args, dataset)
        
        print(f"\n{'='*60}")
        print("ENCODING COMPLETE")
        print(f"{'='*60}")
        if args.output:
            print(f"Latents saved to: {args.output}")
            print(f"Load them later with: --mode load --latent-path {args.output}")
    
    elif args.mode == "load":
        # Load pre-encoded latents
        if not args.latent_path:
            print("ERROR: --latent-path required for load mode")
            return
        
        latent_dataset = load_latent_dataset(args)
        
        print(f"\n{'='*60}")
        print("LOADING COMPLETE")
        print(f"{'='*60}")
    
    elif args.mode == "dataloader":
        # Create training dataloader
        dataset = prepare_image_dataset(args)
        dataloader = create_training_dataloader(args, dataset)
        
        print(f"\n{'='*60}")
        print("DATALOADER READY")
        print(f"{'='*60}")
        print("Use this dataloader in your training script!")
    
    print(f"\n✓ Done!\n")


if __name__ == "__main__":
    main()
