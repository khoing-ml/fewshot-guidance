# Fewshot Dataset Utilities

This document explains how to use the dataset utilities for processing fewshot guidance datasets.

## Overview

The dataset utilities provide:
- **FewshotImageDataset**: Load images from directories
- **FewshotLatentDataset**: Work with pre-encoded latent representations
- **FewshotPairDataset**: Create (query, fewshot_references) pairs for training
- Helper functions for encoding, batching, and data loading

## Directory Structure

Your dataset should follow one of these structures:

### Option 1: Class-based Structure (Recommended)
```
dataset/
├── class1/
│   ├── img1.jpg
│   ├── img2.png
│   └── ...
├── class2/
│   ├── img1.jpg
│   └── ...
└── class3/
    └── ...
```

This structure allows you to sample fewshot examples from the same class as your query image.

### Option 2: Flat Structure
```
dataset/
├── img1.jpg
├── img2.png
├── img3.jpg
└── ...
```

All images are treated as one class.

## Quick Start

### 1. Basic Dataset Loading

```python
from fewshot_guidance.dataset_utils import FewshotImageDataset, print_dataset_stats

# Load dataset
dataset = FewshotImageDataset(
    root_dir="path/to/dataset",
    height=1024,
    width=1024,
)

# Print statistics
print_dataset_stats(dataset)

# Get a sample
sample = dataset[0]
# sample['image']: [3, 1024, 1024] tensor normalized to [-1, 1]
# sample['path']: path to the image file
# sample['class']: class index
```

### 2. Pre-encode Dataset to Latents (Recommended for Training)

Pre-encoding your dataset speeds up training significantly:

```python
from fewshot_guidance.dataset_utils import encode_dataset_to_latents, FewshotLatentDataset
from flux.util import load_ae
import torch

device = torch.device("cuda")
ae = load_ae("flux-schnell", device=device)

# Encode entire dataset
latents, metadata = encode_dataset_to_latents(
    dataset,
    ae,
    device,
    batch_size=4,
    verbose=True,
)

# Save for later use
latent_dataset = FewshotLatentDataset(latents, metadata)
latent_dataset.save("dataset_latents.pt")

# Load later
latent_dataset = FewshotLatentDataset.load("dataset_latents.pt", device="cuda")
```

### 3. Create Fewshot Pairs for Training

```python
from fewshot_guidance.dataset_utils import FewshotPairDataset

# Wrap your dataset to create (query, fewshot) pairs
pair_dataset = FewshotPairDataset(
    base_dataset=dataset,
    num_shots=5,  # Number of fewshot reference images
    same_class=True,  # Sample fewshot from same class as query
)

# Get a sample
sample = pair_dataset[0]
# sample['query_image']: [3, H, W] query image
# sample['fewshot_images']: [5, 3, H, W] five reference images
# sample['query_class']: class index
```

### 4. Create DataLoader for Training

```python
from fewshot_guidance.dataset_utils import create_fewshot_dataloader

# One-line dataloader creation
dataloader = create_fewshot_dataloader(
    dataset_path="path/to/dataset",
    num_shots=5,
    batch_size=2,
    same_class=True,
    shuffle=True,
    height=1024,
    width=1024,
)

# Use in training
for batch in dataloader:
    query_images = batch['query_images']  # [B, 3, H, W]
    fewshot_images = batch['fewshot_images']  # [B, num_shots, 3, H, W]
    # ... your training code
```

## Command-Line Tool

Use the provided script to prepare your dataset:

### Encode dataset to latents:
```bash
python examples/prepare_dataset.py \
    --dataset-path /path/to/dataset \
    --output dataset_latents.pt \
    --mode encode \
    --batch-size 4 \
    --height 1024 \
    --width 1024
```

### Load and inspect latents:
```bash
python examples/prepare_dataset.py \
    --dataset-path /path/to/dataset \
    --latent-path dataset_latents.pt \
    --mode load
```

### Create and test dataloader:
```bash
python examples/prepare_dataset.py \
    --dataset-path /path/to/dataset \
    --mode dataloader \
    --num-shots 5 \
    --same-class \
    --train-batch-size 2
```

## Integration with Fewshot Guidance

Here's how to use the dataset utilities with the fewshot guidance framework:

```python
from fewshot_guidance.dataset_utils import create_fewshot_dataloader
from flux.util import load_ae, load_clip, load_flow_model, load_t5
from flux.sampling import denoise_with_guidance, get_noise, get_schedule, prepare
import torch

# Setup
device = torch.device("cuda")
ae = load_ae("flux-schnell", device=device)
model = load_flow_model("flux-schnell", device=device)
t5 = load_t5(device, max_length=256)
clip = load_clip(device)

# Create dataloader
dataloader = create_fewshot_dataloader(
    dataset_path="path/to/dataset",
    num_shots=5,
    batch_size=1,
    same_class=True,
)

# Training loop
for batch in dataloader:
    query_images = batch['query_images'].to(device)
    fewshot_images = batch['fewshot_images'].to(device)
    
    # Encode to latents
    with torch.no_grad():
        query_latents = ae.encode(query_images.to(torch.bfloat16))
        # Pack latents...
        from einops import rearrange
        query_latents = rearrange(query_latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        
        # Encode fewshot images
        B, num_shots, C, H, W = fewshot_images.shape
        fewshot_flat = fewshot_images.reshape(B * num_shots, C, H, W)
        fewshot_latents = ae.encode(fewshot_flat.to(torch.bfloat16))
        fewshot_latents = rearrange(fewshot_latents, "(b n) c (h ph) (w pw) -> b n (h w) (c ph pw)", 
                                    b=B, n=num_shots, ph=2, pw=2)
    
    # Now use fewshot_latents in your guidance training
    # ... your training code using denoise_with_guidance
```

## Advanced Usage

### Custom Transformations

```python
def custom_transform(img_tensor):
    # Apply random flip, crop, etc.
    import torchvision.transforms as T
    transform = T.RandomHorizontalFlip(p=0.5)
    return transform(img_tensor)

dataset = FewshotImageDataset(
    root_dir="path/to/dataset",
    transform=custom_transform,
)
```

### Manual Batch Creation

```python
from fewshot_guidance.dataset_utils import create_fewshot_batch

# Manually select indices
indices = [0, 5, 10, 15]
batch = create_fewshot_batch(
    indices=indices,
    dataset=dataset,
    num_shots=5,
    same_class=True,
)
```

### Encode Individual Images

```python
from fewshot_guidance.dataset_utils import load_image_and_encode, batch_encode_images

# Single image
latent = load_image_and_encode("image.jpg", ae, device)

# Multiple images
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
latents = batch_encode_images(image_paths, ae, device, batch_size=4)
```

## API Reference

### FewshotImageDataset

```python
FewshotImageDataset(
    root_dir: str | Path,           # Dataset directory
    extensions: List[str] = ['.jpg', '.jpeg', '.png', '.webp'],
    height: int = 1024,              # Target height
    width: int = 1024,               # Target width
    transform: Optional[Callable] = None,  # Optional transform function
)
```

**Returns**: Dictionary with keys `image`, `path`, `class`

**Methods**:
- `get_class_samples(class_idx, num_samples=None)`: Get indices of samples from a class

### FewshotLatentDataset

```python
FewshotLatentDataset(
    latents: Tensor,                 # Pre-encoded latents [N, seq_len, channels]
    metadata: Optional[Dict] = None, # Metadata (paths, classes, etc.)
)
```

**Methods**:
- `save(path)`: Save to disk
- `load(path, device='cpu')`: Load from disk (class method)

### FewshotPairDataset

```python
FewshotPairDataset(
    base_dataset: FewshotImageDataset,  # Base dataset
    num_shots: int = 5,                  # Number of fewshot examples
    same_class: bool = True,             # Sample from same class
)
```

**Returns**: Dictionary with keys `query_image`, `query_path`, `query_class`, `fewshot_images`, `fewshot_paths`

### Helper Functions

- `encode_dataset_to_latents(dataset, ae, device, batch_size=4)`: Encode entire dataset
- `create_fewshot_dataloader(dataset_path, num_shots=5, batch_size=1, ...)`: Create ready-to-use dataloader
- `load_image_and_encode(image_path, ae, device, height, width)`: Encode single image
- `batch_encode_images(image_paths, ae, device, batch_size=4)`: Encode multiple images
- `print_dataset_stats(dataset)`: Print dataset statistics

## Performance Tips

1. **Pre-encode your dataset**: Encoding images to latents once and saving them is much faster than encoding on-the-fly during training.

2. **Use appropriate batch sizes**: For encoding, use batch_size=4-8 depending on your GPU memory.

3. **Consider same-class sampling**: If your dataset has multiple classes, setting `same_class=True` can help the model learn class-specific features.

4. **Use num_workers**: For large datasets, set `num_workers > 0` in DataLoader for parallel data loading.

## Examples

See `examples/prepare_dataset.py` for a complete example showing all features.

See `src/fewshot-guidance/sample_with_fewshot_guidance.py` for integration with the guidance framework.

## Troubleshooting

**No images found**: Make sure your dataset path is correct and contains images with supported extensions (.jpg, .jpeg, .png, .webp).

**Out of memory**: Reduce batch_size when encoding or training.

**Slow data loading**: Pre-encode your dataset to latents and save to disk.

**Different sized images**: All images are automatically resized to the specified height/width.
