#!/usr/bin/env python3
"""
Example training script for fewshot guidance model.

This script demonstrates how to:
1. Initialize models (guidance, flow, ae)
2. Set up training configuration
3. Run the training loop

Usage:
    python train_guidance.py --dataset datasets/512x512_sleeping_yellow_cat --epochs 10
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
import torch

# Import your models
from fewshot_guidance.model.attention_based_model import AttentionGuidanceModel
from fewshot_guidance.model.conv_model import ConvGuidanceModel
from fewshot_guidance.model.mlp_model import MLPGuidanceModel
from fewshot_guidance.train import TrainingConfig, GuidanceTrainer, encode_dataset_to_latents

# Import Flux models and utilities
from flux.util import load_flow_model, load_ae


def parse_args():
    parser = argparse.ArgumentParser(description="Train fewshot guidance model")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Number of warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for optimizer")
    parser.add_argument("--gradient-clip", type=float, default=1.0,
                        help="Gradient clipping norm")
    
    # Model arguments
    parser.add_argument("--hidden-dim", "--hidden_dim", dest="hidden_dim", type=int, default=1024,
                        help="Hidden dimension for guidance model")
    parser.add_argument("--num-layers", "--num_layers", dest="num_layers", type=int, default=2,
                        help="Number of transformer layers")
    parser.add_argument("--num-heads", "--num_heads", dest="num_heads", type=int, default=8,
                        help="Number of attention heads")
    
    # Guidance arguments
    parser.add_argument("--num-shots", "--num_shots", dest="num_shots", type=int, default=4,
                        help="Number of fewshot reference images")
    parser.add_argument("--guidance-scale", "--guidance_scale", dest="guidance_scale", type=float, default=7.5,
                        help="Guidance scale during inference")
    parser.add_argument("--model-type", "--model_type", dest="model_type", type=str, default="attention",
                        choices=["attention", "conv", "mlp"],
                        help="Type of guidance model to use")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--save-every", "--save_every", dest="save_every", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--load-checkpoint", "--load_checkpoint", dest="load_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to train on")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16"],
                        help="Data type for training")
    
    # Logging
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log metrics every N steps")
    
    # Pre-encode dataset
    parser.add_argument("--pre-encode", action="store_true",
                        help="Pre-encode dataset to latents before training")
    parser.add_argument("--encode-save-path", type=str, default=None,
                        help="Path to save pre-encoded latents")
    
    # Flux model selection
    parser.add_argument("--flux-model", type=str, default="flux-dev",
                        choices=["flux-dev", "flux-schnell"],
                        help="Which Flux model to use")
    
    # Additional arguments (alias for consistency)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default=None,
                        help="Output directory (alias for checkpoint-dir)")
    parser.add_argument("--num-fewshot", "--num_fewshot", dest="num_fewshot", type=int, default=None,
                        help="Number of fewshot images (alias for num-shots)")
    
    return parser.parse_args()


def initialize_models(args):
    """Initialize guidance, flow, and ae models."""
    
    # Initialize guidance model
    print(f"\nInitializing guidance model ({args.model_type})...")
    if args.model_type == "attention":
        guidance_model = AttentionGuidanceModel(
            latent_channels=16,
            txt_dim=4096,
            vec_dim=768,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            timestep_embed_dim=256,
        )
    elif args.model_type == "conv":
        guidance_model = ConvGuidanceModel(
            in_channels=16,
            vec_dim=768,
            hidden_channels=args.hidden_dim,
            num_blocks=args.num_layers,
            timestep_embed_dim=256,
        )
    elif args.model_type == "mlp":
        guidance_model = MLPGuidanceModel(
            latent_channels=16,
            txt_dim=4096,
            vec_dim=768,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            use_timestep_embedding=True,
            timestep_embed_dim=256,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    print(f"✓ Initialized {args.model_type.capitalize()}GuidanceModel")
    print(f"  Parameters: {sum(p.numel() for p in guidance_model.parameters()):,}")
    
    # Load Flux flow and AE models
    print(f"\nLoading Flux models ({args.flux_model})...")
    device = args.device
    
    try:
        flow_model = load_flow_model(args.flux_model, device=device)
        print(f"✓ Loaded Flux flow model: {args.flux_model}")
        print(f"  Parameters: {sum(p.numel() for p in flow_model.parameters()):,}")
    except Exception as e:
        print(f"Error loading flow model: {e}")
        return None, None, None
    
    try:
        ae_model = load_ae(args.flux_model, device=device)
        print(f"✓ Loaded Flux AutoEncoder (AE)")
        print(f"  Parameters: {sum(p.numel() for p in ae_model.parameters()):,}")
    except Exception as e:
        print(f"Error loading AE model: {e}")
        return None, None, None
    
    return guidance_model, flow_model, ae_model


def main():
    args = parse_args()
    
    # Handle aliases for arguments
    if args.output_dir:
        args.checkpoint_dir = args.output_dir
    if args.num_fewshot:
        args.num_shots = args.num_fewshot
    
    print(f"\n{'='*60}")
    print(f"Fewshot Guidance Model Training")
    print(f"{'='*60}\n")
    
    # Parse dtype
    dtype = torch.bfloat16 if args.dtype == "float16" else torch.float32
    
    # Initialize models
    print("Initializing models...")
    guidance_model, flow_model, ae_model = initialize_models(args)
    
    if flow_model is None or ae_model is None:
        print("\n" + "!"*60)
        print("ERROR: Failed to initialize Flux models")
        print("!"*60)
        print("\nMake sure you have:")
        print("  1. Valid Flux model selected (flux-dev or flux-schnell)")
        print("  2. Internet connection to download model checkpoints")
        print("  3. HuggingFace authentication if needed")
        print("\nExample:")
        print("  from src.flux.model import Flux")
        print("  from src.flux.sampling import get_flux_ae")
        print("  flow_model = Flux(...)")
        print("  ae_model = get_flux_ae(device=args.device)")
        return
    
    # Create training configuration
    config = TrainingConfig(
        guidance_model=guidance_model,
        flow_model=flow_model,
        ae_model=ae_model,
        dataset_dir=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        num_shots=args.num_shots,
        guidance_scale=args.guidance_scale,
        device=args.device,
        dtype=dtype,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_steps=args.save_every,
        log_every_n_steps=args.log_every,
    )
    
    # Optional: Pre-encode dataset
    if args.pre_encode:
        print("\nPre-encoding dataset to latents...")
        encode_path = args.encode_save_path or f"{Path(args.dataset).name}_latents.pt"
        latents, metadata = encode_dataset_to_latents(
            dataset_dir=args.dataset,
            ae_model=ae_model,
            batch_size=args.batch_size,
            device=args.device,
            dtype=dtype,
            save_path=encode_path,
        )
        print(f"✓ Dataset pre-encoded and saved to {encode_path}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = GuidanceTrainer(config)
    
    # Load checkpoint if provided
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)
    
    # Start training
    print("\nStarting training...")
    trainer.train()


if __name__ == "__main__":
    main()
