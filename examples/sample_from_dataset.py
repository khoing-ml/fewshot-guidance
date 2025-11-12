"""
Example: Training fewshot guidance with a dataset

This shows how to train the guidance model using a dataset of fewshot images
instead of manually specifying image paths.
"""

import torch
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fewshot_guidance.dataset_utils import (
    FewshotImageDataset,
    batch_encode_images,
    print_dataset_stats,
)
from flux.sampling import denoise_with_guidance, get_noise, get_schedule, prepare, unpack
from flux.util import load_ae, load_clip, load_flow_model, load_t5
from flux.guidance_models import MLPGuidanceModel, AttentionGuidanceModel
from einops import rearrange
from PIL import Image
import numpy as np


def sample_fewshot_from_dataset(
    dataset: FewshotImageDataset,
    num_shots: int = 5,
    class_idx: int | None = None,
) -> list[str]:
    """
    Sample fewshot image paths from the dataset.
    
    Args:
        dataset: The image dataset
        num_shots: Number of images to sample
        class_idx: Optional class to sample from (None = random)
    
    Returns:
        List of image paths
    """
    if class_idx is not None:
        # Sample from specific class
        indices = dataset.get_class_samples(class_idx, num_samples=num_shots)
    else:
        import random
        indices = random.sample(range(len(dataset)), min(num_shots, len(dataset)))
    
    return [dataset.image_paths[idx] for idx in indices]


def main():
    parser = argparse.ArgumentParser(
        description="Sample with fewshot guidance using a dataset"
    )
    
    # Dataset arguments
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to fewshot dataset directory")
    parser.add_argument("--num-shots", type=int, default=5,
                       help="Number of fewshot reference images")
    parser.add_argument("--class-name", type=str, default=None,
                       help="Sample fewshot from this class (optional)")
    
    # Generation arguments
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt")
    parser.add_argument("--model", type=str, default="flux-schnell")
    parser.add_argument("--guidance-type", type=str, default="mlp",
                       choices=["mlp", "attention"])
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-train-steps", type=int, default=10)
    parser.add_argument("--guidance-lr", type=float, default=0.001)
    parser.add_argument("--lambda-reconstruction", type=float, default=1.0)
    parser.add_argument("--lambda-consistency", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output_dataset_fewshot.png")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Load dataset
    print("\n" + "="*60)
    print("LOADING FEWSHOT DATASET")
    print("="*60)
    dataset = FewshotImageDataset(
        root_dir=args.dataset_path,
        height=args.height,
        width=args.width,
    )
    print_dataset_stats(dataset)
    
    # Sample fewshot images
    print(f"\n" + "="*60)
    print("SAMPLING FEWSHOT REFERENCES")
    print("="*60)
    
    class_idx = None
    if args.class_name:
        if args.class_name in dataset.class_to_idx:
            class_idx = dataset.class_to_idx[args.class_name]
            print(f"Sampling {args.num_shots} images from class: {args.class_name}")
        else:
            print(f"WARNING: Class '{args.class_name}' not found. Sampling randomly.")
            print(f"Available classes: {list(dataset.class_to_idx.keys())}")
    else:
        print(f"Sampling {args.num_shots} images randomly from dataset")
    
    fewshot_paths = sample_fewshot_from_dataset(
        dataset,
        num_shots=args.num_shots,
        class_idx=class_idx,
    )
    
    print(f"\nSelected fewshot images:")
    for i, path in enumerate(fewshot_paths):
        print(f"  [{i+1}] {Path(path).name}")
    
    # Load models
    print(f"\n" + "="*60)
    print("LOADING MODELS")
    print("="*60)
    print("Loading base models...")
    t5 = load_t5(device, max_length=256)
    clip = load_clip(device)
    model = load_flow_model(args.model, device=args.device)
    ae = load_ae(args.model, device=args.device)
    
    # Encode fewshot images
    print(f"\n" + "="*60)
    print("ENCODING FEWSHOT IMAGES")
    print("="*60)
    fewshot_latents = batch_encode_images(
        fewshot_paths,
        ae,
        device,
        height=args.height,
        width=args.width,
        batch_size=4,
    )
    print(f"✓ Encoded {len(fewshot_paths)} images to latents: {fewshot_latents.shape}")
    
    # Create guidance controller
    print(f"\n" + "="*60)
    print("CREATING GUIDANCE CONTROLLER")
    print("="*60)
    print(f"Type: {args.guidance_type}")
    
    if args.guidance_type == "mlp":
        guidance_model = MLPGuidanceModel(
            latent_channels=64,
            txt_dim=4096,
            vec_dim=768,
            hidden_dim=512,
            num_layers=2,
        ).to(device)
    else:  # attention
        guidance_model = AttentionGuidanceModel(
            latent_channels=64,
            txt_dim=4096,
            vec_dim=768,
            hidden_dim=512,
            num_heads=8,
            num_layers=2,
        ).to(device)
    
    guidance_optimizer = torch.optim.Adam(
        guidance_model.parameters(),
        lr=args.guidance_lr,
    )
    
    # Prepare inputs
    print(f"\n" + "="*60)
    print("PREPARING GENERATION")
    print("="*60)
    print(f"Prompt: '{args.prompt}'")
    print(f"Size: {args.width}x{args.height}")
    print(f"Steps: {args.steps}")
    
    x = get_noise(1, args.height, args.width, device, torch.bfloat16, args.seed)
    initial_noise = x.clone()
    inp = prepare(t5, clip, x, prompt=args.prompt)
    timesteps = get_schedule(args.steps, x.shape[1], shift=(args.model != "flux-schnell"))
    
    # Create loss function
    import torch.nn.functional as F
    
    def create_fewshot_guidance_loss(
        fewshot_latents: torch.Tensor,
        initial_noise: torch.Tensor,
        lambda_reconstruction: float = 1.0,
        lambda_consistency: float = 0.1,
    ):
        def loss_fn(img, pred, guidance_correction, perturbed_guidance_correction, 
                   txt, vec, timestep, step_idx, fewshot_latents):
            guided_pred = pred + guidance_correction
            
            reconstruction_losses = []
            for i in range(fewshot_latents.shape[0]):
                X_1 = fewshot_latents[i:i+1]
                X_0 = initial_noise
                target_direction = X_1 - X_0
                loss = F.mse_loss(guided_pred, target_direction.expand_as(guided_pred))
                reconstruction_losses.append(loss)
            
            reconstruction_loss = torch.min(torch.stack(reconstruction_losses))
            correction_reg = torch.mean((perturbed_guidance_correction - guidance_correction) ** 2)
            
            total_loss = (
                lambda_reconstruction * reconstruction_loss +
                lambda_consistency * correction_reg
            )
            
            return total_loss
        
        return loss_fn
    
    guidance_loss_fn = create_fewshot_guidance_loss(
        fewshot_latents=fewshot_latents,
        initial_noise=initial_noise,
        lambda_reconstruction=args.lambda_reconstruction,
        lambda_consistency=args.lambda_consistency,
    )
    
    # Sample with fewshot guidance
    print(f"\n" + "="*60)
    print("SAMPLING WITH FEWSHOT GUIDANCE")
    print("="*60)
    print(f"  Training steps per timestep: {args.guidance_train_steps}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Learning rate: {args.guidance_lr}")
    
    x = denoise_with_guidance(
        model=model,
        **inp,
        timesteps=timesteps,
        guidance=args.guidance,
        guidance_model=guidance_model,
        guidance_optimizer=guidance_optimizer,
        guidance_loss_fn=guidance_loss_fn,
        guidance_train_steps=args.guidance_train_steps,
        guidance_scale=args.guidance_scale,
        fewshot_latents=fewshot_latents,
    )
    
    # Decode and save
    print(f"\n" + "="*60)
    print("DECODING AND SAVING")
    print("="*60)
    
    x = unpack(x.float(), args.height, args.width)
    with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
        x = ae.decode(x)
    
    x = x.clamp(-1, 1)
    x = (x + 1) / 2
    x = x.permute(0, 2, 3, 1).cpu().numpy()[0]
    x = (x * 255).astype(np.uint8)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    Image.fromarray(x).save(args.output)
    print(f"✓ Saved image to: {args.output}")
    
    # Save controller
    controller_path = str(output_path).replace('.png', '_controller.pt')
    torch.save({
        'model_state_dict': guidance_model.state_dict(),
        'optimizer_state_dict': guidance_optimizer.state_dict(),
        'args': vars(args),
        'fewshot_paths': fewshot_paths,
    }, controller_path)
    print(f"✓ Saved controller to: {controller_path}")
    
    print(f"\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Output: {args.output}")
    print(f"Controller: {controller_path}")
    print()


if __name__ == "__main__":
    main()
