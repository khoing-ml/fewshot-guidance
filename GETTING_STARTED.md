# Getting Started with Fewshot Guidance

## Overview

This project implements fewshot guidance for diffusion models, allowing you to guide image generation using reference images from your dataset.

## Installation

```bash
git clone https://github.com/khoing-ml/fewshot-guidance.git
cd fewshot-guidance
conda create -n fewshot-guidance python=3.10
conda activate fewshot-guidance
pip install -e .
```

## Quick Start

### Option 1: Using a Dataset (Recommended)

1. **Prepare your dataset**:
   ```bash
   # Organize images in class directories
   my_dataset/
     class1/
       img1.jpg
       img2.jpg
     class2/
       img1.jpg
   ```

2. **Encode dataset to latents** (optional but recommended for speed):
   ```bash
   python examples/prepare_dataset.py \
       --dataset-path ./my_dataset \
       --output ./dataset_latents.pt \
       --mode encode \
       --batch-size 4
   ```

3. **Generate images with fewshot guidance**:
   ```bash
   python examples/sample_from_dataset.py \
       --dataset-path ./my_dataset \
       --prompt "a beautiful landscape" \
       --num-shots 5 \
       --class-name "landscapes" \
       --output outputs/result.png
   ```

### Option 2: Using Individual Images

```bash
python src/fewshot-guidance/sample_with_fewshot_guidance.py \
    --prompt "a beautiful landscape" \
    --fewshot-images img1.jpg img2.jpg img3.jpg \
    --guidance-type mlp \
    --output outputs/result.png
```

## Key Features

### Dataset Utilities

The project includes comprehensive dataset utilities for processing fewshot images:

- **FewshotImageDataset**: Load images from directories
- **FewshotLatentDataset**: Work with pre-encoded latents
- **FewshotPairDataset**: Create (query, reference) pairs
- **Pre-encoding**: Speed up training by encoding once

See [docs/DATASET_UTILS.md](docs/DATASET_UTILS.md) for detailed documentation.

### Guidance Models

Three types of guidance models are available:

1. **MLP**: Simple multi-layer perceptron
2. **Attention**: Attention-based model for better fewshot integration
3. **Conv**: Convolutional model (experimental)

## Project Structure

```
fewshot-guidance/
├── src/
│   ├── fewshot-guidance/
│   │   ├── dataset_utils.py      # Dataset processing utilities
│   │   ├── sample_with_fewshot_guidance.py  # Main sampling script
│   │   ├── train.py               # Training utilities
│   │   └── model/                 # Guidance model architectures
│   │       ├── mlp_model.py
│   │       ├── attention_based_model.py
│   │       └── conv_model.py
│   └── flux/                      # Base FLUX model
├── examples/
│   ├── prepare_dataset.py         # Dataset preparation script
│   ├── sample_from_dataset.py     # Sample using dataset
│   └── usage_examples.sh          # Example commands
├── docs/
│   └── DATASET_UTILS.md           # Dataset utilities documentation
└── tests/
    └── test_guidance_framework.py
```

## Documentation

- **[Dataset Utilities Guide](docs/DATASET_UTILS.md)**: Comprehensive guide to dataset processing
- **[Usage Examples](examples/usage_examples.sh)**: Shell script with example commands

## Example Workflows

### Workflow 1: Dataset-based Training

```bash
# 1. Prepare dataset
python examples/prepare_dataset.py \
    --dataset-path ./my_dataset \
    --mode encode \
    --output ./dataset_latents.pt

# 2. Generate with fewshot guidance
python examples/sample_from_dataset.py \
    --dataset-path ./my_dataset \
    --prompt "your prompt" \
    --num-shots 5 \
    --output result.png
```

### Workflow 2: Manual Image Selection

```bash
python src/fewshot-guidance/sample_with_fewshot_guidance.py \
    --prompt "your prompt" \
    --fewshot-images path/to/img1.jpg path/to/img2.jpg \
    --guidance-type attention \
    --steps 4 \
    --output result.png
```

## Advanced Usage

### Custom Loss Functions

You can define custom loss functions for guidance training:

```python
def custom_loss_fn(img, pred, guidance_correction, txt, vec, timestep, step_idx, fewshot_latents):
    # Your custom loss computation
    guided_pred = pred + guidance_correction
    loss = compute_your_loss(guided_pred, fewshot_latents)
    return loss
```

### Custom Guidance Models

Implement your own guidance model by inheriting from `BaseGuidanceModel`:

```python
from fewshot_guidance.model.base_model import BaseGuidanceModel

class MyGuidanceModel(BaseGuidanceModel):
    def forward(self, img, timestep, step_idx, pred, txt, vec, fewshot_img):
        # Your model implementation
        return guidance_correction
```

## Parameters

### Key Parameters

- `--num-shots`: Number of fewshot reference images (default: 5)
- `--guidance-type`: Type of guidance model (mlp, attention, conv)
- `--guidance-scale`: Scaling factor for guidance (default: 1.0)
- `--guidance-train-steps`: Training steps per timestep (default: 10)
- `--guidance-lr`: Learning rate for guidance model (default: 0.001)
- `--lambda-reconstruction`: Weight for reconstruction loss (default: 1.0)
- `--lambda-consistency`: Weight for consistency loss (default: 0.1)
- `--same-class`: Sample fewshot from same class as query

## Troubleshooting

**Out of memory**: Reduce `--batch-size` or `--guidance-train-steps`

**No images found**: Check dataset path and directory structure

**Slow training**: Pre-encode your dataset to latents

**Poor results**: Try adjusting `--guidance-scale`, `--num-shots`, or use `--same-class`

## Citation

If you use this code in your research, please cite:

```bibtex
@software{fewshot-guidance,
  title={Fewshot Guidance for Diffusion Models},
  author={Your Name},
  year={2024},
  url={https://github.com/khoing-ml/fewshot-guidance}
}
```

## License

[Your License Here]

## Contributing

Contributions are welcome! Please see the issues page for areas that need help.
