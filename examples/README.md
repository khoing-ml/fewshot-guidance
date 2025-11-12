# Examples Directory

This directory contains example scripts and usage guides for the fewshot-guidance framework.

## Files

### 1. `prepare_dataset.py`
Prepare and encode your fewshot dataset.

**Modes**:
- `encode`: Load images and encode to latents
- `load`: Load pre-encoded latents
- `dataloader`: Test dataloader creation

**Example**:
```bash
# Encode your dataset
python examples/prepare_dataset.py \
    --dataset-path /path/to/dataset \
    --output dataset_latents.pt \
    --mode encode \
    --batch-size 4
```

### 2. `sample_from_dataset.py`
Generate images using fewshot guidance with automatic dataset sampling.

**Example**:
```bash
# Generate with fewshot guidance
python examples/sample_from_dataset.py \
    --dataset-path /path/to/dataset \
    --prompt "a beautiful landscape" \
    --num-shots 5 \
    --class-name "landscapes" \
    --output result.png
```

**Key Features**:
- Automatically samples fewshot images from dataset
- Can target specific classes
- Full integration with guidance framework
- Saves both image and trained controller

### 3. `usage_examples.sh`
Shell script with copy-paste ready examples.

**Usage**:
```bash
# View examples
./examples/usage_examples.sh

# Or read the file to copy specific commands
cat examples/usage_examples.sh
```

## Quick Start

### Step 1: Organize Your Data

```bash
mkdir -p my_dataset/class1
cp /path/to/images/*.jpg my_dataset/class1/
```

### Step 2: Encode Dataset (Recommended)

```bash
python examples/prepare_dataset.py \
    --dataset-path ./my_dataset \
    --output ./dataset_latents.pt \
    --mode encode
```

This is optional but highly recommended - it makes training 10-100x faster!

### Step 3: Generate Images

```bash
python examples/sample_from_dataset.py \
    --dataset-path ./my_dataset \
    --prompt "your creative prompt" \
    --num-shots 5 \
    --output result.png
```

## Common Use Cases

### Use Case 1: Style Transfer

```bash
# Dataset organized by artist/style
my_dataset/
├── impressionist/
├── abstract/
└── realistic/

# Generate in specific style
python examples/sample_from_dataset.py \
    --dataset-path ./my_dataset \
    --class-name "impressionist" \
    --prompt "a modern cityscape" \
    --num-shots 5
```

### Use Case 2: Product Photography

```bash
# Dataset of product shots
my_dataset/
├── product_A/
├── product_B/
└── product_C/

# Generate new product views
python examples/sample_from_dataset.py \
    --dataset-path ./my_dataset \
    --class-name "product_A" \
    --prompt "product on marble counter" \
    --num-shots 3
```

### Use Case 3: Character Consistency

```bash
# Character reference images
my_dataset/
├── main_character/
└── side_character/

# Generate character in new scenes
python examples/sample_from_dataset.py \
    --dataset-path ./my_dataset \
    --class-name "main_character" \
    --prompt "character hiking in mountains" \
    --num-shots 5
```

## Parameters Reference

### Dataset Parameters

- `--dataset-path`: Path to dataset directory (required)
- `--num-shots`: Number of fewshot reference images (default: 5)
- `--class-name`: Sample from specific class (optional)
- `--same-class`: Sample fewshot from same class flag

### Model Parameters

- `--model`: Model variant (flux-schnell, flux-dev)
- `--guidance-type`: Guidance model type (mlp, attention)
- `--device`: Device to use (cuda, cpu)

### Generation Parameters

- `--prompt`: Text prompt (required)
- `--width`: Image width (default: 1024)
- `--height`: Image height (default: 1024)
- `--steps`: Denoising steps (default: 4)

### Guidance Parameters

- `--guidance-scale`: Scaling for guidance corrections (default: 1.0)
- `--guidance-train-steps`: Training steps per timestep (default: 10)
- `--guidance-lr`: Learning rate for controller (default: 0.001)
- `--lambda-reconstruction`: Weight for reconstruction loss (default: 1.0)
- `--lambda-consistency`: Weight for consistency loss (default: 0.1)

### Output Parameters

- `--output`: Output image path (default: output_*.png)

## Performance Tips

1. **Pre-encode your dataset**: Run `prepare_dataset.py` once with `--mode encode`
2. **Adjust batch size**: Use `--batch-size 4-8` for encoding based on your GPU
3. **Use appropriate image sizes**: Smaller sizes (512x512) are faster
4. **Optimize guidance steps**: Start with `--guidance-train-steps 5-10`

## Troubleshooting

### Issue: Out of Memory

**Solutions**:
- Reduce `--batch-size`
- Use smaller image resolution
- Reduce `--guidance-train-steps`

### Issue: Slow Encoding

**Solutions**:
- Increase `--batch-size` if GPU memory allows
- Pre-encode once with `--mode encode` and reuse

### Issue: Poor Results

**Solutions**:
- Try `--same-class` flag
- Increase/decrease `--num-shots`
- Adjust `--guidance-scale`
- Try different `--guidance-type`

## Advanced Usage

### Custom Dataset Organization

```python
# In your own script
from fewshot_guidance.dataset_utils import FewshotImageDataset

dataset = FewshotImageDataset(
    root_dir="./my_dataset",
    extensions=['.jpg', '.png', '.webp'],
    height=1024,
    width=1024,
)
```

### Batch Processing

```bash
# Process multiple prompts
for prompt in "sunset" "mountain" "ocean"; do
    python examples/sample_from_dataset.py \
        --dataset-path ./my_dataset \
        --prompt "$prompt landscape" \
        --output "outputs/${prompt}.png"
done
```

### Using Pre-encoded Latents

```python
from fewshot_guidance.dataset_utils import FewshotLatentDataset

# Load once
latent_dataset = FewshotLatentDataset.load("dataset_latents.pt")

# Use many times - very fast!
for i in range(100):
    sample = latent_dataset[i]
    # ... your code
```

## Integration Examples

### With Training Loop

```python
from fewshot_guidance.dataset_utils import create_fewshot_dataloader

dataloader = create_fewshot_dataloader(
    dataset_path="./my_dataset",
    num_shots=5,
    batch_size=2,
)

for epoch in range(num_epochs):
    for batch in dataloader:
        # Your training code
        pass
```

### With Custom Loss

```python
def my_custom_loss(img, pred, guidance, ...):
    # Your custom loss computation
    return loss

# Pass to sampling
denoise_with_guidance(
    ...,
    guidance_loss_fn=my_custom_loss,
)
```

## More Resources

- **Full Documentation**: `../docs/DATASET_UTILS.md`
- **Quick Reference**: `../QUICK_REFERENCE.md`
- **Getting Started**: `../GETTING_STARTED.md`
- **Test Suite**: `../test_dataset_utils.py`

## Questions?

1. Read the full documentation in `docs/DATASET_UTILS.md`
2. Check example commands in `usage_examples.sh`
3. Run the test suite: `python test_dataset_utils.py`
4. Review the main README: `../README.md`
