# Fewshot Dataset Utils - Quick Reference

## 📁 Dataset Structure

```
my_dataset/
├── class1/              # ← Organize by class (optional)
│   ├── img001.jpg      # ← Your images
│   ├── img002.png
│   └── img003.jpg
├── class2/
│   ├── img001.jpg
│   └── img002.jpg
└── class3/
    ├── img001.jpg
    └── img002.jpg
```

## 🚀 Quick Commands

### 1️⃣ Encode Your Dataset (One-Time Setup)
```bash
python examples/prepare_dataset.py \
    --dataset-path ./my_dataset \
    --output ./dataset_latents.pt \
    --mode encode
```
**Why?** Pre-encoding makes training 10-100x faster!

### 2️⃣ Generate Images with Fewshot Guidance
```bash
python examples/sample_from_dataset.py \
    --dataset-path ./my_dataset \
    --prompt "a beautiful landscape" \
    --num-shots 5 \
    --class-name "landscapes" \
    --output result.png
```

### 3️⃣ Test Your Setup
```bash
python test_dataset_utils.py
```

## 🐍 Python API Quick Start

```python
from fewshot_guidance.dataset_utils import (
    FewshotImageDataset,
    create_fewshot_dataloader,
)

# Load dataset
dataset = FewshotImageDataset("./my_dataset")
print(f"Loaded {len(dataset)} images")

# Create dataloader
dataloader = create_fewshot_dataloader(
    dataset_path="./my_dataset",
    num_shots=5,          # 5 reference images
    batch_size=2,         # 2 queries per batch
    same_class=True,      # Sample from same class
)

# Use in training
for batch in dataloader:
    query_imgs = batch['query_images']      # [2, 3, H, W]
    fewshot_imgs = batch['fewshot_images']  # [2, 5, 3, H, W]
    # ... your code
```

## 📊 Data Flow

```
Raw Images              Pre-encoded Latents        Fewshot Pairs
┌─────────┐            ┌──────────────┐          ┌──────────────┐
│img1.jpg │            │              │          │ Query: img1  │
│img2.jpg │  ──encode─>│  latents.pt  │──pair──> │ Refs: [2,3,4]│
│img3.jpg │            │   (cached)   │          │              │
│img4.jpg │            │              │          │ Query: img2  │
└─────────┘            └──────────────┘          │ Refs: [1,3,5]│
                                                 └──────────────┘
```

## 🔧 Main Classes

| Class | Purpose | Usage |
|-------|---------|-------|
| `FewshotImageDataset` | Load images from disk | `dataset = FewshotImageDataset("./path")` |
| `FewshotLatentDataset` | Pre-encoded latents | `dataset.save("latents.pt")` |
| `FewshotPairDataset` | Query+fewshot pairs | `pairs = FewshotPairDataset(dataset, num_shots=5)` |

## 📝 Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset-path` | - | Path to your dataset directory |
| `--num-shots` | 5 | Number of fewshot reference images |
| `--same-class` | False | Sample fewshot from same class |
| `--height` | 1024 | Image height |
| `--width` | 1024 | Image width |
| `--batch-size` | 4 | Batch size for encoding |
| `--guidance-type` | mlp | Type of guidance model (mlp/attention) |
| `--guidance-scale` | 1.0 | Scaling factor for guidance |

## 🎯 Use Cases

### Use Case 1: Style Transfer
```bash
# Dataset of paintings
my_dataset/
├── van_gogh/
├── monet/
└── picasso/

# Generate in Van Gogh style
python examples/sample_from_dataset.py \
    --dataset-path ./my_dataset \
    --class-name "van_gogh" \
    --prompt "a modern city" \
    --num-shots 5
```

### Use Case 2: Domain Adaptation
```bash
# Dataset of product photos
my_dataset/
├── product_A/
├── product_B/
└── product_C/

# Generate new views
python examples/sample_from_dataset.py \
    --dataset-path ./my_dataset \
    --class-name "product_A" \
    --prompt "product on white background" \
    --num-shots 3
```

### Use Case 3: Character Consistency
```bash
# Dataset of character images
my_dataset/
├── character1/
└── character2/

# Generate character in new scene
python examples/sample_from_dataset.py \
    --dataset-path ./my_dataset \
    --class-name "character1" \
    --prompt "character in a forest" \
    --num-shots 5
```

## ⚡ Performance Tips

1. **Pre-encode** your dataset first:
   ```bash
   python examples/prepare_dataset.py --mode encode ...
   ```

2. **Use batch encoding**:
   ```python
   latents = batch_encode_images(image_paths, ae, device, batch_size=8)
   ```

3. **Save encoded latents**:
   ```python
   latent_dataset.save("latents.pt")  # Load instantly next time
   ```

4. **Optimize batch size**:
   - RTX 3090/4090: batch_size=4-8
   - RTX 3080: batch_size=2-4
   - RTX 3060: batch_size=1-2

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No images found" | Check path and file extensions (.jpg, .png, etc.) |
| Out of memory | Reduce `--batch-size` or image resolution |
| Slow training | Pre-encode dataset with `--mode encode` |
| Poor results | Try `--same-class` flag or adjust `--num-shots` |

## 📚 More Info

- **Full Docs**: `docs/DATASET_UTILS.md`
- **Examples**: `examples/usage_examples.sh`
- **Test Suite**: `python test_dataset_utils.py`
- **Getting Started**: `GETTING_STARTED.md`

## 🎓 Tutorial: End-to-End

```bash
# 1. Setup
mkdir my_dataset/landscapes
cp ~/Pictures/landscapes/*.jpg my_dataset/landscapes/

# 2. Test it works
python test_dataset_utils.py

# 3. Encode (optional but recommended)
python examples/prepare_dataset.py \
    --dataset-path ./my_dataset \
    --output dataset_latents.pt \
    --mode encode

# 4. Generate!
python examples/sample_from_dataset.py \
    --dataset-path ./my_dataset \
    --prompt "a serene mountain landscape" \
    --num-shots 5 \
    --output my_landscape.png

# 5. Check result
open my_landscape.png
```

## 📦 What You Get

```
Input:
- Dataset with images
- Text prompt

Process:
- Auto-sample K fewshot images
- Encode to latents
- Train guidance model online
- Denoise with guidance

Output:
- Generated image guided by fewshot examples
- Saved guidance controller
```

---

**Need Help?** Check `docs/DATASET_UTILS.md` or run `./examples/usage_examples.sh`
