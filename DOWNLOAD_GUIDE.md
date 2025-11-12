# Quick Guide: Download Your Datasets

## 🚀 Fastest Method (One Command)

```bash
python scripts/download_and_prepare.py --encode
```

This will:
- ✓ Download all datasets from Google Drive
- ✓ Extract them automatically
- ✓ Encode to latents for fast training
- ✓ Show you statistics

## 📋 Step-by-Step

### Step 1: Install Dependencies

```bash
pip install gdown
```

### Step 2: Download Datasets

Choose one method:

**Method A: Python (Recommended)**
```bash
python scripts/download_and_prepare.py
```

**Method B: Bash Script**
```bash
./scripts/download_from_gdrive.sh
```

### Step 3: Verify Download

```bash
ls -la datasets/
```

You should see your dataset directories.

### Step 4: (Optional) Encode to Latents

This makes training 10-100x faster:

```bash
# For each dataset
python examples/prepare_dataset.py \
    --dataset-path datasets/[dataset_name] \
    --output datasets/[dataset_name]_latents.pt \
    --mode encode
```

Or do it automatically during download:
```bash
python scripts/download_and_prepare.py --encode
```

### Step 5: Generate Images!

```bash
python examples/sample_from_dataset.py \
    --dataset-path datasets/[dataset_name] \
    --prompt "a beautiful landscape" \
    --num-shots 5 \
    --output my_result.png
```

## 🔧 Common Options

### Download to Different Location

```bash
python scripts/download_and_prepare.py --output-dir /path/to/datasets
```

### Keep Zip Files

```bash
python scripts/download_and_prepare.py --keep-zips
```

### Use CPU Instead of GPU

```bash
python scripts/download_and_prepare.py --encode --device cpu
```

### Skip Download (Only Extract)

If you already have the zip files:

```bash
python scripts/download_and_prepare.py --skip-download
```

## ❓ Troubleshooting

### "gdown not found"

```bash
pip install gdown
```

### "Permission denied" or "Cannot access folder"

Try manual download:
1. Visit: https://drive.google.com/drive/folders/1chmH5iyCV-mEpuacPDOEG76wrS5ywHi9
2. Download each file manually
3. Place in `datasets/` folder
4. Run: `python scripts/download_and_prepare.py --skip-download`

### "Out of memory" during encoding

```bash
# Use smaller batch size
python examples/prepare_dataset.py \
    --dataset-path datasets/[dataset_name] \
    --mode encode \
    --batch-size 2
```

### Download is very slow

- Check your internet connection
- Try downloading during off-peak hours
- Consider manual download for large files

## 📊 After Download

Your directory structure will look like:

```
datasets/
├── dataset1/
│   ├── class1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── class2/
│       └── ...
├── dataset1_latents.pt  (if encoded)
├── dataset2/
│   └── ...
└── dataset2_latents.pt  (if encoded)
```

## 🎯 Next Steps

1. **Test your setup**:
   ```bash
   python test_dataset_utils.py
   ```

2. **Explore your datasets**:
   ```bash
   python examples/prepare_dataset.py \
       --dataset-path datasets/dataset1 \
       --mode dataloader
   ```

3. **Generate your first image**:
   ```bash
   python examples/sample_from_dataset.py \
       --dataset-path datasets/dataset1 \
       --prompt "your creative prompt here" \
       --num-shots 5 \
       --output result.png
   ```

## 📚 More Help

- **Script documentation**: `scripts/README.md`
- **Dataset utilities**: `docs/DATASET_UTILS.md`
- **Examples**: `examples/README.md`
- **Quick reference**: `QUICK_REFERENCE.md`
- **Getting started**: `GETTING_STARTED.md`

## 💡 Pro Tips

1. **Always encode first**: Pre-encoding saves tons of time
   ```bash
   python scripts/download_and_prepare.py --encode
   ```

2. **Check dataset stats**: Before training, verify your data
   ```bash
   python examples/prepare_dataset.py --dataset-path datasets/dataset1 --mode dataloader
   ```

3. **Start with small num-shots**: Test with 3-5 shots first
   ```bash
   python examples/sample_from_dataset.py ... --num-shots 3
   ```

4. **Use specific classes**: If your dataset has classes, target them
   ```bash
   python examples/sample_from_dataset.py ... --class-name "landscapes"
   ```

---

**Ready to start?** Run this now:

```bash
python scripts/download_and_prepare.py --encode
```

Then generate your first image! 🎨
