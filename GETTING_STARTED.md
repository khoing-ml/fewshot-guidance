# Getting Started: Custom Guidance for Flux Sampling

## What Has Been Added

I've implemented a complete framework for adding custom guidance models to the Flux diffusion sampling process. Here's what's now available:

### 📁 Modified Files

1. **`src/flux/sampling.py`**
   - Added `guidance_model`, `guidance_scale`, and `guidance_requires_grad` parameters to `denoise()`
   - Added `apply_guidance_model()` function that applies guidance at each step
   - Guidance models can now modify predictions at each timestep

### 📁 New Files

1. **`src/flux/guidance_models.py`** - Guidance model architectures
   - `BaseGuidanceModel` - Abstract base class for all guidance models
   - `MLPGuidanceModel` - Simple MLP-based guidance (good starting point)
   - `AttentionGuidanceModel` - Attention-based with text conditioning (more powerful)
   - `ConvGuidanceModel` - Convolutional for spatial guidance
   - `GuidanceModelTrainer` - Training utilities with supervised and reward-based learning

2. **`src/flux/sampling_instrumented.py`** - Data collection utilities
   - `denoise_with_state_collection()` - Modified denoise that saves intermediate states
   - `StateCollectionCallback` - Collect and organize states from multiple generations
   - `create_training_pairs()` - Convert states into training data

3. **`examples/train_guidance_model.py`** - Complete examples
   - Supervised training with target corrections
   - Reward-based training with quality scores
   - Inference with trained guidance
   - Step-specific guidance models
   - Multi-objective guidance

4. **`examples/collect_guidance_data.py`** - Data collection helpers
   - `GuidanceDataCollector` - Automated data collection from generations
   - Multiple strategies for creating training targets
   - Dataset saving/loading utilities

5. **`GUIDANCE_README.md`** - Comprehensive documentation

## Quick Start Guide

### Step 1: Understand the Concept

A guidance model learns to **adjust** the diffusion model's predictions at each timestep to achieve specific objectives:

```
Base Model Prediction + Guidance Model Correction = Improved Prediction
```

The guidance model sees:
- Current latent state (`img`)
- Base model's prediction (`pred`)
- Text embeddings (`txt`, `vec`)
- Current timestep
- Step index

And outputs a correction signal to add to the prediction.

### Step 2: Choose Your Approach

#### Option A: Train with Supervised Learning
If you have examples of good corrections:

```python
from flux.guidance_models import MLPGuidanceModel, GuidanceModelTrainer

# Create model
model = MLPGuidanceModel(latent_channels=64, hidden_dim=1024)

# Train with target corrections
trainer = GuidanceModelTrainer(model, optimizer)
loss = trainer.train_step(img, pred, txt, vec, timestep, step_idx, target_correction)
```

#### Option B: Train with Reward Function
If you have a quality/reward model:

```python
def reward_fn(predicted_latent):
    return aesthetic_scorer(predicted_latent)

# Train to maximize reward
metrics = trainer.train_from_reward(
    img, pred, txt, vec, timestep, step_idx,
    reward_fn=reward_fn,
    num_guidance_steps=10
)
```

### Step 3: Collect Training Data

Use the instrumented sampling to collect states during generation:

```python
from flux.sampling_instrumented import denoise_with_state_collection, StateCollectionCallback

# Setup callback
callback = StateCollectionCallback(save_dir="training_data")

# Generate and collect states
img, states = denoise_with_state_collection(
    model=flux_model,
    img=noise,
    # ... other parameters ...
    save_states=True,
)

# Save this generation
callback.on_generation_complete(
    prompt="your prompt",
    states=states,
    final_image=decoded_image,
)

# Save all collected data
callback.save_to_disk("my_data.pkl")
```

### Step 4: Train Your Guidance Model

```python
from flux.guidance_models import create_mlp_guidance_model, GuidanceModelTrainer
import torch.optim as optim

# Create model
guidance_model = create_mlp_guidance_model(
    latent_channels=64,
    hidden_dim=1024,
    num_layers=3,
)

# Setup training
optimizer = optim.AdamW(guidance_model.parameters(), lr=1e-4)
trainer = GuidanceModelTrainer(guidance_model, optimizer)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        metrics = trainer.train_step(
            img=batch['img'],
            pred=batch['pred'],
            txt=batch['txt'],
            vec=batch['vec'],
            timestep=batch['timestep'],
            step_idx=batch['step_idx'],
            target_correction=batch['target'],
        )
        print(f"Loss: {metrics['loss']:.4f}")

# Save trained model
trainer.save_checkpoint("my_guidance.pt")
```

### Step 5: Use Trained Guidance in Sampling

```python
from flux.sampling import denoise, get_schedule

# Load your trained guidance model
guidance_model = create_mlp_guidance_model()
checkpoint = torch.load("my_guidance.pt")
guidance_model.load_state_dict(checkpoint['model_state_dict'])
guidance_model.eval()

# Use during generation
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
    # Add your guidance
    guidance_model=guidance_model,
    guidance_scale=1.0,  # Adjust strength
    guidance_requires_grad=False,  # False for inference
)
```

## Common Use Cases

### 1. Aesthetic Enhancement
Train guidance to improve aesthetic quality:
```python
# Use aesthetic predictor as reward
def aesthetic_reward(latent):
    return aesthetic_scorer(decode(latent))

trainer.train_from_reward(..., reward_fn=aesthetic_reward)
```

### 2. Style Control
Train guidance to enforce specific styles:
```python
# Use CLIP similarity to style references
def style_reward(latent):
    img_features = clip.encode_image(decode(latent))
    style_features = clip.encode_image(reference_style)
    return cosine_similarity(img_features, style_features)
```

### 3. Composition Improvement
Use ConvGuidanceModel for spatial layout:
```python
from flux.guidance_models import ConvGuidanceModel

guidance_model = ConvGuidanceModel(
    in_channels=16,
    hidden_channels=256,
    num_blocks=3,
)
```

### 4. Prompt Adherence
Use AttentionGuidanceModel for better text alignment:
```python
from flux.guidance_models import AttentionGuidanceModel

guidance_model = AttentionGuidanceModel(
    hidden_dim=1024,
    num_heads=8,
    num_layers=2,
)
```

## Tips for Success

1. **Start Small**: Begin with `MLPGuidanceModel` and low `guidance_scale` (0.1-0.5)

2. **Data Quality Matters**: Collect data from diverse prompts and scenarios

3. **Monitor Training**: Watch that guidance doesn't overwhelm base model

4. **Step-Specific**: Different steps may need different guidance strengths

5. **Validate Often**: Test on held-out prompts regularly

6. **Combine Objectives**: Use `MultiObjectiveGuidance` for multiple goals

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Sampling Loop                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  For each timestep:                               │  │
│  │                                                    │  │
│  │  1. Get base model prediction                     │  │
│  │     pred = model(img, txt, vec, t)                │  │
│  │                                                    │  │
│  │  2. Apply guidance model (if provided)            │  │
│  │     guidance = guidance_model(img, pred, txt, t)  │  │
│  │     pred = pred + guidance_scale * guidance       │  │
│  │                                                    │  │
│  │  3. Update latent                                 │  │
│  │     img = img + (t_prev - t_curr) * pred          │  │
│  │                                                    │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Next Steps

1. **Experiment**: Try the examples in `examples/train_guidance_model.py`

2. **Collect Data**: Use `sampling_instrumented.py` to gather training data

3. **Train Simple Model**: Start with MLPGuidanceModel on small dataset

4. **Evaluate**: Test guidance on diverse prompts

5. **Iterate**: Adjust architecture, data, and training strategy

6. **Scale Up**: Once working, collect more data and train larger models

## Files Reference

- `src/flux/sampling.py` - Modified sampling with guidance support
- `src/flux/guidance_models.py` - Guidance model architectures
- `src/flux/sampling_instrumented.py` - Data collection utilities
- `examples/train_guidance_model.py` - Training examples
- `examples/collect_guidance_data.py` - Data collection helpers
- `GUIDANCE_README.md` - Full documentation

## Questions?

The framework is designed to be flexible. You can:
- Create custom guidance models by inheriting `BaseGuidanceModel`
- Use any training strategy (supervised, reinforcement learning, etc.)
- Apply guidance globally or per-step
- Combine multiple guidance models
- Train separate models for different timestep ranges

Good luck with your custom guidance models! 🚀
