# Custom Guidance Models for Flux Sampling

This directory contains a flexible framework for adding custom guidance to the Flux diffusion sampling process. The guidance models can be trained separately to adjust the denoising process according to specific objectives.

## Overview

The guidance system allows you to train neural networks that modify the diffusion model's predictions at each timestep to achieve desired properties in the generated images. This is useful for:

- **Quality improvement**: Guide towards higher aesthetic scores
- **Style control**: Enforce specific artistic styles
- **Composition guidance**: Improve layout and structure
- **Concept emphasis**: Strengthen specific concepts from the prompt
- **Multi-modal objectives**: Combine multiple guidance signals

## Architecture

### Core Components

1. **`sampling.py`**: Modified denoising loop with guidance support
   - `denoise()`: Main sampling function with guidance parameters
   - `apply_guidance_model()`: Applies guidance at each step

2. **`guidance_models.py`**: Guidance model architectures
   - `BaseGuidanceModel`: Abstract base class
   - `MLPGuidanceModel`: Simple MLP-based guidance
   - `AttentionGuidanceModel`: Attention-based guidance with text conditioning
   - `ConvGuidanceModel`: Convolutional guidance for spatial features
   - `GuidanceModelTrainer`: Training utilities

3. **`examples/train_guidance_model.py`**: Usage examples and training scripts

## Quick Start

### 1. Basic Usage

```python
from flux.sampling import denoise, get_schedule
from flux.guidance_models import create_mlp_guidance_model

# Create and load a trained guidance model
guidance_model = create_mlp_guidance_model(
    latent_channels=64,
    hidden_dim=1024,
    num_layers=3,
)
guidance_model.load_state_dict(torch.load("path/to/checkpoint.pt"))
guidance_model.eval()

# Use during sampling
timesteps = get_schedule(num_steps=28, image_seq_len=img.shape[1])

latents = denoise(
    model=flux_model,
    img=noise,
    img_ids=img_ids,
    txt=txt_embeddings,
    txt_ids=txt_ids,
    vec=vec_embeddings,
    timesteps=timesteps,
    guidance=3.5,
    # Custom guidance parameters
    guidance_model=guidance_model,      # Your trained model
    guidance_scale=1.0,                 # Strength of guidance
    guidance_requires_grad=False,       # False for inference, True for training
)
```

### 2. Training a Guidance Model

#### Option A: Supervised Learning

If you have paired data (base predictions + target corrections):

```python
from flux.guidance_models import MLPGuidanceModel, GuidanceModelTrainer
import torch.optim as optim

# Create model
guidance_model = MLPGuidanceModel(
    latent_channels=64,
    hidden_dim=1024,
    num_layers=3,
)

# Setup trainer
optimizer = optim.AdamW(guidance_model.parameters(), lr=1e-4)
trainer = GuidanceModelTrainer(guidance_model, optimizer)

# Training loop
for batch in dataloader:
    img, pred, txt, vec, timestep, step_idx, target = batch
    
    metrics = trainer.train_step(
        img=img,
        pred=pred,
        txt=txt,
        vec=vec,
        timestep=timestep,
        step_idx=step_idx,
        target_correction=target,
    )
    
    print(f"Loss: {metrics['loss']:.4f}")

# Save model
trainer.save_checkpoint("guidance_model.pt")
```

#### Option B: Reward-Based Learning

If you have a reward function (e.g., aesthetic scorer, CLIP similarity):

```python
# Define reward function
def aesthetic_reward(latent_prediction):
    # Your reward model that scores the prediction
    score = aesthetic_scorer(latent_prediction)
    return score

# Train with reward
for batch in dataloader:
    img, pred, txt, vec, timestep, step_idx = batch
    
    metrics = trainer.train_from_reward(
        img=img,
        pred=pred,
        txt=txt,
        vec=vec,
        timestep=timestep,
        step_idx=step_idx,
        reward_fn=aesthetic_reward,
        num_guidance_steps=10,
        guidance_lr=0.1,
    )
    
    print(f"Reward: {metrics['avg_reward']:.4f}")
```

## Guidance Model Architectures

### MLPGuidanceModel

Simple MLP-based model suitable for most guidance tasks.

```python
from flux.guidance_models import MLPGuidanceModel

model = MLPGuidanceModel(
    latent_channels=64,        # Channels in packed latents
    txt_dim=4096,              # T5 embedding dimension
    vec_dim=768,               # CLIP embedding dimension
    hidden_dim=1024,           # Hidden layer size
    num_layers=3,              # Number of MLP layers
    use_timestep_embedding=True,
    timestep_embed_dim=256,
)
```

**Pros**: Fast, lightweight, good starting point  
**Cons**: Limited spatial reasoning

### AttentionGuidanceModel

Attention-based model with cross-attention to text conditioning.

```python
from flux.guidance_models import AttentionGuidanceModel

model = AttentionGuidanceModel(
    latent_channels=64,
    txt_dim=4096,
    vec_dim=768,
    hidden_dim=1024,
    num_heads=8,
    num_layers=2,
)
```

**Pros**: Better text alignment, captures relationships  
**Cons**: Slower, more parameters

### ConvGuidanceModel

Convolutional model for spatial guidance.

```python
from flux.guidance_models import ConvGuidanceModel

model = ConvGuidanceModel(
    in_channels=16,           # Latent channels before packing
    vec_dim=768,
    hidden_channels=256,
    num_blocks=3,
)
```

**Pros**: Excellent for composition, spatial features  
**Cons**: Requires unpacking/repacking latents

## Advanced Usage

### Step-Specific Guidance

Use different guidance models at different denoising stages:

```python
from examples.train_guidance_model import StepSpecificGuidanceWrapper

# Create specialized models
early_guidance = MLPGuidanceModel(hidden_dim=512)   # Early: composition
late_guidance = MLPGuidanceModel(hidden_dim=1024)   # Late: details

# Wrap them
step_guidance = StepSpecificGuidanceWrapper({
    (0, 15): early_guidance,
    (15, 28): late_guidance,
})

# Use in sampling
latents = denoise(..., guidance_model=step_guidance, ...)
```

### Multi-Objective Guidance

Combine multiple guidance objectives:

```python
from examples.train_guidance_model import MultiObjectiveGuidance

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

latents = denoise(..., guidance_model=multi_guidance, ...)
```

### Custom Guidance Models

Extend `BaseGuidanceModel` to create your own:

```python
from flux.guidance_models import BaseGuidanceModel
import torch.nn as nn

class MyCustomGuidance(BaseGuidanceModel):
    def __init__(self):
        super().__init__()
        # Your architecture here
        self.network = nn.Sequential(...)
    
    def forward(self, img, pred, txt, vec, timestep, step_idx):
        # Your logic here
        guidance = self.network(...)
        return guidance
```

## Data Collection for Training

To train guidance models, you need data. Here are strategies:

### Strategy 1: Collect Prediction-Target Pairs

1. Generate samples with base model
2. Use a reward model to score them
3. For good samples, compute what correction would improve them further
4. Train guidance model to predict these corrections

### Strategy 2: Synthetic Data from Known Objectives

1. Generate diverse samples
2. Apply post-hoc transformations that achieve your objective
3. Compute the "ideal" guidance that would have produced those transformations
4. Train model to predict this guidance

### Strategy 3: Online Learning

1. Generate samples during training
2. Score them with reward model in real-time
3. Use reward gradients to train guidance model
4. Iterate

## Tips and Best Practices

1. **Start Simple**: Begin with `MLPGuidanceModel` and small `guidance_scale`
2. **Initialize Output Near Zero**: The output layer should start near zero for stable training
3. **Gradual Scaling**: Start with `guidance_scale=0.1` and increase gradually
4. **Step-Dependent**: Different steps may need different guidance strengths
5. **Monitor Carefully**: Watch for guidance overwhelming the base model
6. **Validation**: Always validate guidance on held-out prompts
7. **Checkpointing**: Save models frequently during training

## Troubleshooting

### Guidance Too Strong
- Reduce `guidance_scale`
- Add regularization to guidance model
- Use smaller model capacity

### Guidance Not Effective
- Increase `guidance_scale`
- Use larger model
- Ensure training data is relevant
- Check that guidance model sees correct timestep info

### Training Instability
- Reduce learning rate
- Add gradient clipping
- Use smaller guidance model
- Increase batch size

## API Reference

### Core Functions

#### `denoise()`
Main sampling function with guidance support.

**Parameters**:
- `guidance_model` (nn.Module | None): Trained guidance model
- `guidance_scale` (float): Strength of guidance (default: 1.0)
- `guidance_requires_grad` (bool): Enable gradients through guidance (default: True)

#### `apply_guidance_model()`
Internal function that applies guidance at each step.

### Guidance Models

All guidance models inherit from `BaseGuidanceModel` and implement:

```python
def forward(
    self,
    img: Tensor,          # Current latent [B, seq_len, channels]
    pred: Tensor,         # Base prediction [B, seq_len, channels]
    txt: Tensor,          # Text embeddings [B, txt_len, 4096]
    vec: Tensor,          # CLIP embeddings [B, 768]
    timestep: Tensor,     # Timestep [B]
    step_idx: int,        # Step index
) -> Tensor:
    # Return guidance signal [B, seq_len, channels]
```

## Examples

See `examples/train_guidance_model.py` for complete working examples:
- Supervised training
- Reward-based training
- Inference with guidance
- Step-specific guidance
- Multi-objective guidance

## Citation

If you use this guidance framework in your research, please cite:

```bibtex
@misc{flux-custom-guidance,
  title={Custom Guidance Framework for Flux Diffusion Models},
  author={Your Name},
  year={2025}
}
```

## License

[Your License Here]
