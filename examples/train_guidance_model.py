"""
Example: Training and Using Custom Guidance Models

This script demonstrates how to:
1. Create a custom guidance model
2. Train it with different objectives
3. Use it during sampling

The guidance model learns to adjust the diffusion model's predictions
at each timestep to achieve specific objectives (e.g., better aesthetics,
specific styles, improved composition, etc.)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Import from your flux implementation
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from flux.guidance_models import (
    MLPGuidanceModel,
    AttentionGuidanceModel,
    GuidanceModelTrainer,
    create_mlp_guidance_model,
)
from flux.sampling import denoise, get_noise, prepare, get_schedule
from flux.util import load_flux_model, load_ae, load_t5, load_clip


# ============================================================================
# Example 1: Training with Supervised Learning
# ============================================================================

def train_supervised_guidance():
    """
    Train guidance model with supervised learning.
    
    Assumes you have pairs of (base_prediction, target_correction) collected
    from analyzing good vs bad generations.
    """
    
    # Create guidance model
    guidance_model = create_mlp_guidance_model(
        latent_channels=64,
        hidden_dim=1024,
        num_layers=3,
    )
    
    # Setup optimizer
    optimizer = optim.AdamW(guidance_model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Create trainer
    trainer = GuidanceModelTrainer(
        guidance_model=guidance_model,
        optimizer=optimizer,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Iterate through your dataset
        # for batch in dataloader:  # Assuming you have a dataloader
        #     img, pred, txt, vec, timestep, step_idx, target = batch
        
        # Dummy data for demonstration
        batch_size = 4
        seq_len = 256
        img = torch.randn(batch_size, seq_len, 64)
        pred = torch.randn(batch_size, seq_len, 64)
        txt = torch.randn(batch_size, 77, 4096)
        vec = torch.randn(batch_size, 768)
        timestep = torch.rand(batch_size)
        step_idx = 0
        target = torch.randn(batch_size, seq_len, 64) * 0.1  # Small corrections
        
        # Training step
        metrics = trainer.train_step(
            img=img,
            pred=pred,
            txt=txt,
            vec=vec,
            timestep=timestep,
            step_idx=step_idx,
            target_correction=target,
        )
        
        epoch_loss += metrics['loss']
        num_batches += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss / num_batches:.6f}")
    
    # Save trained model
    trainer.save_checkpoint("checkpoints/guidance_model.pt")
    print("Training complete!")
    
    return guidance_model


# ============================================================================
# Example 2: Training with Reward-Based Learning
# ============================================================================

def train_reward_based_guidance():
    """
    Train guidance model using a reward function.
    
    The model learns to modify predictions to maximize a reward
    (e.g., aesthetic score, CLIP similarity, etc.)
    """
    
    # Create guidance model
    guidance_model = create_mlp_guidance_model(
        latent_channels=64,
        hidden_dim=1024,
        num_layers=3,
    )
    
    # Setup optimizer
    optimizer = optim.AdamW(guidance_model.parameters(), lr=1e-4)
    
    # Create trainer
    trainer = GuidanceModelTrainer(
        guidance_model=guidance_model,
        optimizer=optimizer,
    )
    
    # Define a reward function
    # This is a placeholder - replace with your actual reward model
    class AestheticRewardModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        
        def forward(self, latent):
            # Compute reward based on latent features
            features = latent.mean(dim=1)  # Pool over sequence
            reward = self.mlp(features).mean()  # Single scalar reward
            return reward
    
    reward_model = AestheticRewardModel().cuda()
    
    def reward_fn(modified_pred):
        return reward_model(modified_pred)
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        epoch_reward = 0.0
        num_batches = 0
        
        # Generate or load samples
        batch_size = 4
        seq_len = 256
        img = torch.randn(batch_size, seq_len, 64).cuda()
        pred = torch.randn(batch_size, seq_len, 64).cuda()
        txt = torch.randn(batch_size, 77, 4096).cuda()
        vec = torch.randn(batch_size, 768).cuda()
        timestep = torch.rand(batch_size).cuda()
        step_idx = 0
        
        # Train with reward
        metrics = trainer.train_from_reward(
            img=img,
            pred=pred,
            txt=txt,
            vec=vec,
            timestep=timestep,
            step_idx=step_idx,
            reward_fn=reward_fn,
            num_guidance_steps=5,
            guidance_lr=0.1,
        )
        
        epoch_reward += metrics['avg_reward']
        num_batches += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Avg Reward: {epoch_reward / num_batches:.6f}")
    
    # Save trained model
    trainer.save_checkpoint("checkpoints/reward_guidance_model.pt")
    print("Reward-based training complete!")
    
    return guidance_model


# ============================================================================
# Example 3: Using Guidance Model During Sampling
# ============================================================================

def generate_with_guidance(
    prompt: str,
    guidance_model_path: str,
    guidance_scale: float = 1.0,
):
    """
    Generate images using a trained guidance model.
    
    Args:
        prompt: Text prompt for generation
        guidance_model_path: Path to trained guidance model checkpoint
        guidance_scale: Strength of guidance (0 = no guidance, higher = stronger)
    """
    
    # Load models
    print("Loading models...")
    # model = load_flux_model("flux-dev")  # Load your flux model
    # ae = load_ae("flux-dev")
    # t5 = load_t5()
    # clip = load_clip()
    
    # For demonstration, we'll use dummy models
    class DummyFlux(nn.Module):
        def forward(self, img, img_ids, txt, txt_ids, y, timesteps, guidance):
            return torch.randn_like(img) * 0.1
    
    model = DummyFlux().cuda()
    
    # Load guidance model
    print(f"Loading guidance model from {guidance_model_path}...")
    guidance_model = create_mlp_guidance_model()
    checkpoint = torch.load(guidance_model_path)
    guidance_model.load_state_dict(checkpoint['model_state_dict'])
    guidance_model.eval()
    guidance_model.cuda()
    
    # Generation parameters
    height, width = 512, 512
    num_steps = 28
    guidance = 3.5
    seed = 42
    
    # Get noise
    noise = get_noise(
        num_samples=1,
        height=height,
        width=width,
        device=torch.device('cuda'),
        dtype=torch.bfloat16,
        seed=seed,
    )
    
    # Prepare conditioning (dummy for demonstration)
    txt = torch.randn(1, 77, 4096, dtype=torch.bfloat16).cuda()
    txt_ids = torch.zeros(1, 77, 3, dtype=torch.bfloat16).cuda()
    vec = torch.randn(1, 768, dtype=torch.bfloat16).cuda()
    img_ids = torch.zeros(1, noise.shape[1], 3, dtype=torch.bfloat16).cuda()
    
    # Real usage would be:
    # inp = prepare(t5=t5, clip=clip, img=noise, prompt=prompt)
    # txt, txt_ids, vec, img, img_ids = inp["txt"], inp["txt_ids"], inp["vec"], inp["img"], inp["img_ids"]
    
    # Get timestep schedule
    timesteps = get_schedule(
        num_steps=num_steps,
        image_seq_len=noise.shape[1],
        shift=True,
    )
    
    # Denoise with guidance model
    print("Generating with custom guidance...")
    latents = denoise(
        model=model,
        img=noise,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        vec=vec,
        timesteps=timesteps,
        guidance=guidance,
        # Custom guidance parameters
        guidance_model=guidance_model,
        guidance_scale=guidance_scale,
        guidance_requires_grad=False,  # Set to False for inference
    )
    
    # Decode latents (dummy for demonstration)
    # image = ae.decode(latents)
    
    print("Generation complete!")
    return latents


# ============================================================================
# Example 4: Step-Specific Guidance Models
# ============================================================================

class StepSpecificGuidanceWrapper(nn.Module):
    """
    Wrapper that uses different guidance models for different steps.
    
    Useful when different steps require different types of guidance
    (e.g., early steps for composition, later steps for details).
    """
    
    def __init__(self, guidance_models_by_step):
        """
        Args:
            guidance_models_by_step: Dict mapping step ranges to models
                Example: {(0, 10): model_early, (10, 20): model_mid, (20, 28): model_late}
        """
        super().__init__()
        self.guidance_models_by_step = guidance_models_by_step
    
    def forward(self, img, pred, txt, vec, timestep, step_idx):
        # Find which model to use based on step_idx
        for (start, end), model in self.guidance_models_by_step.items():
            if start <= step_idx < end:
                return model(img, pred, txt, vec, timestep, step_idx)
        
        # Default: no guidance
        return torch.zeros_like(pred)


def train_step_specific_models():
    """Train separate guidance models for different denoising stages."""
    
    # Create models for different stages
    early_model = create_mlp_guidance_model(hidden_dim=512, num_layers=2)  # Lighter
    mid_model = create_mlp_guidance_model(hidden_dim=1024, num_layers=3)  # Standard
    late_model = create_mlp_guidance_model(hidden_dim=1024, num_layers=4)  # Deeper for details
    
    # Train each model on appropriate step ranges
    # (Training code similar to examples above, but filter data by step_idx)
    
    # Create wrapper
    step_specific_guidance = StepSpecificGuidanceWrapper({
        (0, 10): early_model,
        (10, 20): mid_model,
        (20, 28): late_model,
    })
    
    return step_specific_guidance


# ============================================================================
# Example 5: Multi-Objective Guidance
# ============================================================================

class MultiObjectiveGuidance(nn.Module):
    """
    Combines multiple guidance models for different objectives.
    
    Each objective has its own model and weight.
    """
    
    def __init__(self, objective_models, objective_weights=None):
        """
        Args:
            objective_models: Dict of {objective_name: model}
            objective_weights: Dict of {objective_name: weight}
        """
        super().__init__()
        self.objective_models = nn.ModuleDict(objective_models)
        
        if objective_weights is None:
            objective_weights = {k: 1.0 for k in objective_models.keys()}
        self.objective_weights = objective_weights
    
    def forward(self, img, pred, txt, vec, timestep, step_idx):
        # Combine guidance from all objectives
        total_guidance = torch.zeros_like(pred)
        
        for name, model in self.objective_models.items():
            guidance = model(img, pred, txt, vec, timestep, step_idx)
            weight = self.objective_weights.get(name, 1.0)
            total_guidance = total_guidance + weight * guidance
        
        return total_guidance


def create_multi_objective_guidance():
    """Create guidance with multiple objectives."""
    
    # Create models for different objectives
    aesthetic_model = create_mlp_guidance_model(hidden_dim=1024)
    composition_model = create_mlp_guidance_model(hidden_dim=512)
    detail_model = create_mlp_guidance_model(hidden_dim=1024)
    
    # Combine with weights
    multi_guidance = MultiObjectiveGuidance(
        objective_models={
            'aesthetic': aesthetic_model,
            'composition': composition_model,
            'detail': detail_model,
        },
        objective_weights={
            'aesthetic': 1.0,
            'composition': 0.5,
            'detail': 0.3,
        }
    )
    
    return multi_guidance


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Custom Guidance Model Training Examples")
    print("=" * 80)
    
    # Example 1: Supervised training
    print("\n1. Training with supervised learning...")
    # guidance_model = train_supervised_guidance()
    
    # Example 2: Reward-based training
    print("\n2. Training with reward-based learning...")
    # guidance_model = train_reward_based_guidance()
    
    # Example 3: Using trained model for generation
    print("\n3. Generating with custom guidance...")
    # result = generate_with_guidance(
    #     prompt="A beautiful sunset over mountains",
    #     guidance_model_path="checkpoints/guidance_model.pt",
    #     guidance_scale=1.5,
    # )
    
    # Example 4: Step-specific guidance
    print("\n4. Creating step-specific guidance models...")
    # step_guidance = train_step_specific_models()
    
    # Example 5: Multi-objective guidance
    print("\n5. Creating multi-objective guidance...")
    # multi_guidance = create_multi_objective_guidance()
    
    print("\n" + "=" * 80)
    print("Examples complete! Uncomment the functions to run them.")
    print("=" * 80)
