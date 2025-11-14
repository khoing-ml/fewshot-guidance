# Dataset Sampling in Fewshot Guidance

## Overview

The `denoise_with_guidance()` function now supports **automatic dataset loading and sampling** during the generation process. You can simply pass a dataset path instead of pre-encoding images.

## Two Approaches

### 🎯 Approach 1: Direct Dataset Path (Recommended)

**Simplest way** - just pass the dataset path and let the sampler handle everything:

```python
from flux.sampling import denoise_with_guidance, get_noise, get_schedule, prepare
from flux.util import load_ae, load_clip, load_flow_model, load_t5

# Load models
device = torch.device("cuda")
t5 = load_t5(device, max_length=256)
clip = load_clip(device)
model = load_flow_model("flux-schnell", device=device)
ae = load_ae("flux-schnell", device=device)

# Prepare inputs
prompt = "A vending machine"
x = get_noise(1, 512, 512, device, torch.bfloat16, seed=42)
inp = prepare(t5, clip, x, prompt=prompt)
timesteps = get_schedule(4, x.shape[1], shift=False)

# Sample with automatic dataset loading
x = denoise_with_guidance(
    model=model,
    **inp,
    timesteps=timesteps,
    guidance=1.0,
    
    # 🎯 NEW: Just pass the dataset path!
    fewshot_dataset_path="datasets/512x512_vending_machine",
    fewshot_num_shots=5,              # How many images to sample
    fewshot_class_name="vending_machine",  # Optional: specific class
    fewshot_seed=42,                   # Optional: reproducibility
    fewshot_ae=ae,                     # Required for encoding
    fewshot_height=512,
    fewshot_width=512,
    
    # Your guidance components
    guidance_model=my_guidance_model,
    guidance_optimizer=optimizer,
    guidance_loss_fn=my_loss_fn,
)
```

### 📦 Approach 2: Pre-load and Encode (Advanced)

**More control** - manually load and encode images before sampling:

```python
from flux.sampling import (
    load_dataset_from_directory,
    sample_images_from_dataset,
    load_and_encode_images,
    denoise_with_guidance
)

# 1. Load dataset
image_paths, class_labels, class_to_idx = load_dataset_from_directory(
    "datasets/512x512_vending_machine"
)

# 2. Sample specific images
fewshot_paths = sample_images_from_dataset(
    image_paths, class_labels, class_to_idx,
    num_shots=5,
    class_name="vending_machine",
    seed=42
)

# 3. Encode to latents
fewshot_latents = load_and_encode_images(
    fewshot_paths, ae, device, height=512, width=512
)

# 4. Use pre-encoded latents
x = denoise_with_guidance(
    model=model,
    **inp,
    timesteps=timesteps,
    fewshot_latents=fewshot_latents,  # Pre-encoded
    # ... other params
)
```

## Dataset Structure

Supports two directory structures:

### Class-based (Recommended)
```
datasets/my_dataset/
├── class1/
│   ├── img1.jpg
│   ├── img2.png
│   └── ...
├── class2/
│   ├── img1.jpg
│   └── ...
```

### Flat
```
datasets/my_dataset/
├── img1.jpg
├── img2.png
└── ...
```

## Parameters

### Dataset Loading Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fewshot_dataset_path` | `str \| None` | `None` | Path to dataset directory |
| `fewshot_num_shots` | `int` | `5` | Number of images to sample |
| `fewshot_class_name` | `str \| None` | `None` | Specific class to sample from |
| `fewshot_seed` | `int \| None` | `None` | Random seed for sampling |
| `fewshot_ae` | `AutoEncoder` | Required | VAE for encoding images |
| `fewshot_height` | `int` | `1024` | Target image height |
| `fewshot_width` | `int` | `1024` | Target image width |

### Pre-encoded Latents (Alternative)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fewshot_latents` | `Tensor \| None` | `None` | Pre-encoded latents `[N, seq, ch]` |

## What Happens During Sampling

When you pass `fewshot_dataset_path`:

1. **Load dataset** - Scans directory for images
2. **Sample images** - Randomly or from specific class
3. **Encode images** - Converts to latent space
4. **Print info** - Shows selected images and shapes
5. **Continue sampling** - Uses encoded latents for guidance

Example output:
```
=== Loading Fewshot Dataset ===
Dataset path: datasets/512x512_vending_machine
Found 50 images in 1 classes

Sampling 5 fewshot images...
  Randomly from all images
Selected fewshot images:
  [1] vm_001.jpg
  [2] vm_042.jpg
  [3] vm_013.jpg
  [4] vm_029.jpg
  [5] vm_037.jpg

Encoding 5 fewshot images...
✓ Fewshot latents shape: torch.Size([5, 1024, 64])
```

## Benefits

✅ **Simpler code** - No manual dataset loading  
✅ **Automatic encoding** - Handles image preprocessing  
✅ **Flexible sampling** - Class-specific or random  
✅ **Reproducible** - Seed support for deterministic sampling  
✅ **Clear logging** - See exactly what was sampled  

## Complete Example

```python
import torch
from flux.sampling import denoise_with_guidance, get_noise, get_schedule, prepare, unpack
from flux.util import load_ae, load_clip, load_flow_model, load_t5
from PIL import Image
import numpy as np

device = torch.device("cuda")

# Load models
t5 = load_t5(device, max_length=256)
clip = load_clip(device)
model = load_flow_model("flux-schnell", device=device)
ae = load_ae("flux-schnell", device=device)

# Setup guidance (your custom model and loss)
guidance_model = MyGuidanceModel().to(device)
optimizer = torch.optim.Adam(guidance_model.parameters(), lr=0.001)

# Prepare
prompt = "A red vending machine"
x = get_noise(1, 512, 512, device, torch.bfloat16, seed=42)
inp = prepare(t5, clip, x, prompt=prompt)
timesteps = get_schedule(4, x.shape[1], shift=False)

# Sample with dataset guidance
x = denoise_with_guidance(
    model=model,
    **inp,
    timesteps=timesteps,
    guidance=1.0,
    # Dataset-based fewshot
    fewshot_dataset_path="datasets/512x512_vending_machine",
    fewshot_num_shots=5,
    fewshot_class_name="vending_machine",
    fewshot_seed=42,
    fewshot_ae=ae,
    fewshot_height=512,
    fewshot_width=512,
    # Guidance
    guidance_model=guidance_model,
    guidance_optimizer=optimizer,
    guidance_loss_fn=my_loss_function,
    guidance_train_steps=10,
    guidance_scale=1.0,
)

# Decode and save
x = unpack(x.float(), 512, 512)
x = ae.decode(x)
x = (x.clamp(-1, 1) + 1) / 2
x = (x.permute(0, 2, 3, 1).cpu().numpy()[0] * 255).astype(np.uint8)
Image.fromarray(x).save("output.png")
```

## Notes

- **Either/Or**: Pass `fewshot_dataset_path` OR `fewshot_latents`, not both
- **AutoEncoder Required**: Must provide `fewshot_ae` when using dataset path
- **Memory**: Images are encoded on-the-fly, no need to pre-load everything
- **Supported formats**: `.jpg`, `.jpeg`, `.png`, `.webp`
