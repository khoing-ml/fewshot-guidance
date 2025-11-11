"""
Example: Fewshot Guidance with Online Training

This demonstrates how to use fewshot reference images to guide the sampling process.
The guidance controller c^φ is conditioned on fewshot examples from your dataset
and trained online at each timestep to minimize your custom loss.

Formulation:
L_RGM(φ) = E[||u_t^θ(t,X_t) + c_t^φ(t,X_t,X_fewshot) - (X_1 - X_0)||^2] + λ|c_t^φ(X_1) - c_t^φ(X̂_1)|

where:
- u_t^θ: Base diffusion model prediction
- c_t^φ: Learned guidance controller
- X_1: Target fewshot image latent (from dataset)
- X_0: Initial noise
- (X_1 - X_0): Trajectory from noise to target
- X̂_1 = X_1 + ε: Perturbed fewshot example
- λ: Robustness weight

The controller learns to guide the prediction along the trajectory (X_1 - X_0)
from initial noise to the target fewshot image.


"""

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np

from flux.sampling import denoise_with_guidance, get_noise, get_schedule, prepare, unpack
from flux.util import load_ae, load_clip, load_flow_model, load_t5
from flux.guidance_models import MLPGuidanceModel, AttentionGuidanceModel


def load_and_encode_fewshot_images(
    image_paths: list[str],
    ae,
    device: torch.device,
    height: int = 1024,
    width: int = 1024,
) -> torch.Tensor:
    """
    Load and encode fewshot reference images.
    
    Returns:
        Encoded latents [num_shots, seq_len, channels]
    """
    latents = []
    
    for img_path in image_paths:
        print(f"  Loading {img_path}")
        img = Image.open(img_path).convert("RGB")
        img = img.resize((width, height))
        img = np.array(img).astype(np.float32) / 127.5 - 1.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            latent = ae.encode(img.to(torch.bfloat16))
            # Pack the latent
            from einops import rearrange
            latent = rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            latents.append(latent)
    
    # Stack all fewshot latents [num_shots, seq_len, channels]
    fewshot_latents = torch.cat(latents, dim=0)
    return fewshot_latents


def create_fewshot_guidance_loss(
    fewshot_latents: torch.Tensor,
    initial_noise: torch.Tensor,
    lambda_reconstruction: float = 1.0,
    lambda_consistency: float = 0.1,
):
    """
    Create a loss function that implements your formulation:
    
    L_RGM(φ) = E[||u_t^θ(t,X_t) + c_t^φ(t,X_t) - (X_1 - X_0)||^2] + λ|c_t^φ(X_1) - c_t^φ(X̂_1)|
    
    where:
    - X_1: Target fewshot image latent
    - X_0: Initial noise
    - (X_1 - X_0): The trajectory from noise to target
    
    Args:
        fewshot_latents: Target latents X_1 [num_shots, seq_len, channels]
        initial_noise: Initial noise X_0 [batch, seq_len, channels]
        lambda_reconstruction: Weight for reconstruction term
        lambda_consistency: Weight for consistency/regularization
    
    Returns:
        Loss function for guidance training
    """
    
    def loss_fn(img, pred, guidance_correction,perturbed_guidance_correction, txt, vec, timestep, step_idx, fewshot_latents):
        """
        Custom loss at each timestep.
        
        The controller learns to guide the prediction towards the trajectory (X_1 - X_0).
        """
        # Guided prediction: u_t^θ + c_t^φ
        guided_pred = pred + guidance_correction
        
        # Term 1: ||u_t^θ + c_t^φ - (X_1 - X_0)||^2
        if fewshot_latents is not None and len(fewshot_latents) > 0:
            # X_1 - X_0 (target - initial_noise)
            reconstruction_losses = []
            
            for i in range(fewshot_latents.shape[0]):
                X_1 = fewshot_latents[i:i+1]  # Target fewshot latent [1, seq_len, channels]
                X_0 = initial_noise  # Initial noise [batch, seq_len, channels]
                
                # Target trajectory: X_1 - X_0
                target_direction = X_1 - X_0
                
                # Loss: ||guided_pred - target_direction||^2
                loss = F.mse_loss(guided_pred, target_direction.expand_as(guided_pred))
                reconstruction_losses.append(loss)
            
            # Use the best matching fewshot example (minimum loss)
            reconstruction_loss = torch.min(torch.stack(reconstruction_losses))
        else:
            # Fallback if no fewshot available
            reconstruction_loss = torch.mean(guided_pred ** 2)
        
        # Term 2: Robustness term λ|c_t^φ(X_t) - c_t^φ(X̂_t)|

        correction_reg = torch.mean((perturbed_guidance_correction - guidance_correction) ** 2)
        
        # Combined loss
        loss = (
            lambda_reconstruction * reconstruction_loss +
            lambda_consistency * correction_reg
        )
        
        return loss
    
    return loss_fn


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Sample with fewshot-guided controller"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--fewshot-images", type=str, nargs="+", required=True,
                       help="Paths to fewshot reference images")
    parser.add_argument("--model", type=str, default="flux-schnell")
    parser.add_argument("--guidance-type", type=str, default="mlp",
                       choices=["mlp", "attention"],
                       help="Type of guidance model")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0,
                       help="Lambda: scaling for guidance corrections")
    parser.add_argument("--guidance-train-steps", type=int, default=10,
                       help="Optimization steps per timestep")
    parser.add_argument("--guidance-lr", type=float, default=0.001,
                       help="Learning rate for controller")
    parser.add_argument("--lambda-reconstruction", type=float, default=1.0)
    parser.add_argument("--lambda-consistency", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output_fewshot.png")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load base models
    print("Loading models...")
    t5 = load_t5(device, max_length=256)
    clip = load_clip(device)
    model = load_flow_model(args.model, device=args.device)
    ae = load_ae(args.model, device=args.device)
    
    # Load and encode fewshot reference images
    print(f"\nLoading {len(args.fewshot_images)} fewshot reference images...")
    fewshot_latents = load_and_encode_fewshot_images(
        args.fewshot_images,
        ae,
        device,
        args.height,
        args.width,
    )
    print(f"✓ Fewshot latents shape: {fewshot_latents.shape}")
    
    # Create guidance controller
    print(f"\nCreating {args.guidance_type} guidance controller...")
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
    
    # Create optimizer
    guidance_optimizer = torch.optim.Adam(
        guidance_model.parameters(),
        lr=args.guidance_lr,
    )
    
    # Prepare inputs
    print(f"\nGenerating: '{args.prompt}'")
    x = get_noise(1, args.height, args.width, device, torch.bfloat16, args.seed)
    initial_noise = x.clone()  # Save X_0 for loss computation
    inp = prepare(t5, clip, x, prompt=args.prompt)
    timesteps = get_schedule(args.steps, x.shape[1], shift=(args.model != "flux-schnell"))
    
    # Create loss function that uses fewshot examples
    # Loss computes: ||u_t^θ + c_t^φ - (X_1 - X_0)||^2
    # where X_1 = fewshot_latents (target), X_0 = initial_noise
    guidance_loss_fn = create_fewshot_guidance_loss(
        fewshot_latents=fewshot_latents,  # X_1 (target from fewshot)
        initial_noise=initial_noise,       # X_0 (initial noise)
        lambda_reconstruction=args.lambda_reconstruction,
        lambda_consistency=args.lambda_consistency,
    )
    
    # Sample with fewshot-guided controller
    print(f"\nSampling with fewshot guidance...")
    print(f"  - {args.steps} denoising steps")
    print(f"  - {args.guidance_train_steps} training steps per timestep")
    print(f"  - Guidance scale λ: {args.guidance_scale}")
    print(f"  - {len(args.fewshot_images)} fewshot references")
    
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
        fewshot_latents=fewshot_latents,  # Pass fewshot references
    )
    
    # Decode
    print("\nDecoding...")
    x = unpack(x.float(), args.height, args.width)
    with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
        x = ae.decode(x)
    
    # Save
    x = x.clamp(-1, 1)
    x = (x + 1) / 2
    x = x.permute(0, 2, 3, 1).cpu().numpy()[0]
    x = (x * 255).astype(np.uint8)
    
    Image.fromarray(x).save(args.output)
    print(f"✓ Saved to {args.output}")
    
    # Save controller
    controller_path = args.output.replace('.png', '_controller.pt')
    torch.save({
        'model_state_dict': guidance_model.state_dict(),
        'optimizer_state_dict': guidance_optimizer.state_dict(),
        'fewshot_images': args.fewshot_images,
    }, controller_path)
    print(f"✓ Saved controller to {controller_path}")


if __name__ == "__main__":
    main()
