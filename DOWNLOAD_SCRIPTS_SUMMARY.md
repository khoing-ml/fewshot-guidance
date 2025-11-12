# Dataset Download Scripts - Summary

## What Has Been Created

I've created comprehensive scripts to download and prepare your datasets from the Google Drive folder.

## Files Created

### 1. **`scripts/download_and_prepare.py`** (Main Script - Recommended)

Full-featured Python script that handles everything:

**Features**:
- ✓ Downloads from Google Drive folder automatically
- ✓ Extracts all zip files
- ✓ Shows dataset statistics (images, size, classes)
- ✓ Optional automatic encoding to latents
- ✓ Resume capability (skips existing files)
- ✓ Error handling and user-friendly messages

**Usage**:
```bash
# Download and encode everything
python scripts/download_and_prepare.py --encode

# Download only
python scripts/download_and_prepare.py

# Custom location
python scripts/download_and_prepare.py --output-dir /path/to/datasets
```

**Options**:
- `--output-dir DIR`: Where to save datasets (default: ./datasets)
- `--folder-id ID`: Google Drive folder ID (already set to your folder)
- `--keep-zips`: Keep zip files after extraction
- `--encode`: Automatically encode to latents after download
- `--model`: Model for encoding (flux-schnell/flux-dev)
- `--device`: Device (cuda/cpu)
- `--skip-download`: Only extract existing zips

### 2. **`scripts/download_from_gdrive.sh`** (Bash Alternative)

Simple bash script for quick downloads:

```bash
./scripts/download_from_gdrive.sh
# or
./scripts/download_from_gdrive.sh /path/to/datasets
```

### 3. **`scripts/download_datasets.py`** (Template)

Template for downloading individual files by ID. Useful if you need fine-grained control.

### 4. **Documentation**

- **`scripts/README.md`**: Complete documentation for all scripts
- **`DOWNLOAD_GUIDE.md`**: Quick start guide for downloading datasets

## Google Drive Folder

Your datasets are in:
- **URL**: https://drive.google.com/drive/folders/1chmH5iyCV-mEpuacPDOEG76wrS5ywHi9
- **Folder ID**: `1chmH5iyCV-mEpuacPDOEG76wrS5ywHi9` (already configured in scripts)

## Quick Start

### Fastest Way (One Command)

```bash
python scripts/download_and_prepare.py --encode
```

This single command will:
1. Install `gdown` if needed
2. Download all datasets from Google Drive
3. Extract all zip files
4. Encode datasets to latents (10-100x faster training!)
5. Show you statistics about each dataset
6. Give you next steps

### Step by Step

```bash
# 1. Download datasets
python scripts/download_and_prepare.py

# 2. Verify
ls -la datasets/

# 3. Encode (optional but recommended)
python examples/prepare_dataset.py \
    --dataset-path datasets/[dataset_name] \
    --mode encode \
    --output datasets/[dataset_name]_latents.pt

# 4. Generate images
python examples/sample_from_dataset.py \
    --dataset-path datasets/[dataset_name] \
    --prompt "your prompt" \
    --num-shots 5
```

## What Happens During Download

```
1. Install gdown (if needed)
   ↓
2. Download from Google Drive
   ├─ dataset1.zip
   ├─ dataset2.zip
   └─ dataset3.zip
   ↓
3. Extract zip files
   ├─ datasets/dataset1/
   ├─ datasets/dataset2/
   └─ datasets/dataset3/
   ↓
4. Show statistics
   - Number of images
   - Classes
   - Size
   ↓
5. Encode (if --encode flag)
   ├─ dataset1_latents.pt
   ├─ dataset2_latents.pt
   └─ dataset3_latents.pt
   ↓
6. Ready to use! 🎉
```

## Directory Structure After Download

```
datasets/
├── dataset1/
│   ├── class1/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── class2/
│       ├── img001.jpg
│       └── ...
├── dataset1_latents.pt  # Pre-encoded (if --encode used)
├── dataset2/
│   └── ...
└── dataset2_latents.pt
```

## Troubleshooting

### Issue: gdown not installed

**Auto-fix**: The script will install it automatically

**Manual**: `pip install gdown`

### Issue: Download fails

**Solutions**:
1. Check internet connection
2. Try again (script resumes automatically)
3. Manual download:
   - Visit: https://drive.google.com/drive/folders/1chmH5iyCV-mEpuacPDOEG76wrS5ywHi9
   - Download files manually
   - Place in `datasets/`
   - Run: `python scripts/download_and_prepare.py --skip-download`

### Issue: Out of disk space

**Solutions**:
- Don't use `--keep-zips` (removes zips after extraction)
- Download to different location: `--output-dir /path/with/space`
- Download datasets one at a time

### Issue: Out of memory during encoding

**Solution**:
```bash
# Use smaller batch size or skip auto-encode
python scripts/download_and_prepare.py  # without --encode

# Then encode manually with smaller batch
python examples/prepare_dataset.py \
    --dataset-path datasets/dataset1 \
    --mode encode \
    --batch-size 2  # or even 1
```

## Complete Workflow Example

```bash
# Step 1: Download and prepare everything
python scripts/download_and_prepare.py --encode

# Step 2: Verify downloads
ls -la datasets/

# Expected output:
# datasets/
#   dataset1/
#   dataset1_latents.pt
#   dataset2/
#   dataset2_latents.pt
#   ...

# Step 3: Generate your first image
python examples/sample_from_dataset.py \
    --dataset-path datasets/dataset1 \
    --prompt "a beautiful sunset landscape" \
    --num-shots 5 \
    --class-name "landscapes" \
    --output my_first_result.png

# Step 4: View result
open my_first_result.png  # or xdg-open on Linux
```

## Advanced Options

### Download to Specific Location

```bash
python scripts/download_and_prepare.py \
    --output-dir /mnt/large_drive/datasets \
    --encode
```

### Use Different Model for Encoding

```bash
python scripts/download_and_prepare.py \
    --encode \
    --model flux-dev \
    --device cuda
```

### Keep Zip Files

```bash
python scripts/download_and_prepare.py \
    --keep-zips \
    --encode
```

### Resume Interrupted Download

Just run the same command again:
```bash
python scripts/download_and_prepare.py --encode
```

The script automatically:
- Skips already downloaded files
- Skips already extracted directories
- Continues where it left off

## Integration with Existing Tools

The downloaded datasets work seamlessly with all the dataset utilities:

```python
from fewshot_guidance.dataset_utils import FewshotImageDataset

# Load downloaded dataset
dataset = FewshotImageDataset("./datasets/dataset1")

# Or load pre-encoded latents
from fewshot_guidance.dataset_utils import FewshotLatentDataset
latents = FewshotLatentDataset.load("./datasets/dataset1_latents.pt")
```

## What Each Script Does

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `download_and_prepare.py` | Complete automation | **Always use this** |
| `download_from_gdrive.sh` | Quick bash download | When you prefer bash |
| `download_datasets.py` | Individual file control | When you need custom downloads |

## Next Steps After Download

1. **Verify datasets**:
   ```bash
   python test_dataset_utils.py
   ```

2. **Explore a dataset**:
   ```bash
   python examples/prepare_dataset.py \
       --dataset-path datasets/dataset1 \
       --mode dataloader
   ```

3. **Generate images**:
   ```bash
   python examples/sample_from_dataset.py \
       --dataset-path datasets/dataset1 \
       --prompt "your creative prompt" \
       --output result.png
   ```

4. **Read documentation**:
   - Dataset utilities: `docs/DATASET_UTILS.md`
   - Quick reference: `QUICK_REFERENCE.md`
   - Examples: `examples/README.md`

## Performance Tips

1. **Always use --encode**: Pre-encoding saves 10-100x time during training
2. **Use SSD**: Download to SSD for faster extraction
3. **Batch size**: Adjust based on GPU memory (4 for most GPUs)
4. **Remove zips**: Don't use --keep-zips to save space

## Files Reference

```
scripts/
├── download_and_prepare.py    ← Main script (USE THIS)
├── download_from_gdrive.sh    ← Bash alternative
├── download_datasets.py       ← Template for custom downloads
└── README.md                  ← Complete documentation

DOWNLOAD_GUIDE.md              ← Quick start guide
```

## Support

- **Download issues**: See `scripts/README.md`
- **Dataset issues**: See `docs/DATASET_UTILS.md`
- **Generation issues**: See `examples/README.md`
- **Quick help**: See `DOWNLOAD_GUIDE.md`

---

**Ready to download?** Run this command now:

```bash
python scripts/download_and_prepare.py --encode
```

It will handle everything automatically! 🚀
