# Scripts Directory

This directory contains utility scripts for downloading and managing datasets.

## Quick Start

### Option 1: Automatic Download and Prepare (Recommended)

```bash
# Download all datasets from Google Drive and prepare them
python scripts/download_and_prepare.py

# Download and automatically encode to latents
python scripts/download_and_prepare.py --encode

# Download to specific directory
python scripts/download_and_prepare.py --output-dir /path/to/datasets
```

### Option 2: Shell Script

```bash
# Download using bash script
./scripts/download_from_gdrive.sh

# Or specify output directory
./scripts/download_from_gdrive.sh /path/to/datasets
```

## Available Scripts

### 1. `download_and_prepare.py` (Recommended)

Full-featured Python script to download, extract, and prepare datasets.

**Features**:
- Downloads from Google Drive folder
- Automatically extracts zip files
- Shows dataset statistics
- Optional automatic encoding to latents
- Resume capability

**Usage**:
```bash
# Basic usage
python scripts/download_and_prepare.py

# With automatic encoding
python scripts/download_and_prepare.py --encode

# Custom options
python scripts/download_and_prepare.py \
    --output-dir ./my_datasets \
    --encode \
    --model flux-schnell \
    --device cuda
```

**Options**:
- `--output-dir DIR`: Output directory (default: ./datasets)
- `--folder-id ID`: Google Drive folder ID (default: your shared folder)
- `--keep-zips`: Keep zip files after extraction
- `--encode`: Automatically encode datasets to latents
- `--model`: Model for encoding (flux-schnell, flux-dev)
- `--device`: Device for encoding (cuda, cpu)
- `--skip-download`: Only extract existing zips

### 2. `download_from_gdrive.sh`

Bash script for quick downloads using gdown.

**Usage**:
```bash
./scripts/download_from_gdrive.sh [output_dir]
```

### 3. `download_datasets.py`

Template script for downloading individual files by ID.

**Usage**:
1. Edit the script to add your file IDs
2. Run: `python scripts/download_datasets.py`

**Options**:
- `--list`: List available datasets
- `--dataset NAME`: Download specific dataset
- `--output-dir DIR`: Output directory
- `--keep-zip`: Keep zip files

## Google Drive Folder

The scripts download from this shared folder:
- **URL**: https://drive.google.com/drive/folders/1chmH5iyCV-mEpuacPDOEG76wrS5ywHi9
- **Folder ID**: `1chmH5iyCV-mEpuacPDOEG76wrS5ywHi9`

## Installation

All scripts will automatically install `gdown` if needed:

```bash
pip install gdown
```

Or install manually:
```bash
pip install gdown
```

## Workflow

### Complete Workflow

```bash
# 1. Download and prepare datasets
python scripts/download_and_prepare.py --encode

# 2. Verify datasets
ls -la datasets/

# 3. Generate images
python examples/sample_from_dataset.py \
    --dataset-path datasets/[dataset_name] \
    --prompt "your prompt" \
    --output result.png
```

### Manual Workflow

```bash
# 1. Download only
python scripts/download_and_prepare.py

# 2. Encode datasets later
python examples/prepare_dataset.py \
    --dataset-path datasets/dataset1 \
    --output datasets/dataset1_latents.pt \
    --mode encode

# 3. Use in generation
python examples/sample_from_dataset.py \
    --dataset-path datasets/dataset1 \
    --prompt "your prompt"
```

## Troubleshooting

### Issue: gdown not installed

**Solution**:
```bash
pip install gdown
```

### Issue: Download fails with "Permission denied"

**Solutions**:
1. Make sure the Google Drive folder is shared properly
2. Try the manual download method
3. Check your internet connection

### Issue: "Cannot download folder"

**Solution**: Download files manually:
1. Visit: https://drive.google.com/drive/folders/1chmH5iyCV-mEpuacPDOEG76wrS5ywHi9
2. Download each file
3. Place in `./datasets/`
4. Run: `python scripts/download_and_prepare.py --skip-download`

### Issue: Out of disk space

**Solution**:
- Use `--keep-zips=False` to remove zips after extraction
- Check available space: `df -h`
- Download to a different directory with more space

### Issue: Extraction fails

**Solution**:
- Verify the download completed (check file sizes)
- Try re-downloading the specific file
- Check if the file is actually a zip file: `file filename.zip`

## Manual Download

If automatic download doesn't work:

1. **Visit the Google Drive folder**:
   ```
   https://drive.google.com/drive/folders/1chmH5iyCV-mEpuacPDOEG76wrS5ywHi9
   ```

2. **Download each file**:
   - Click on each file
   - Click "Download"
   - Save to `./datasets/`

3. **Extract manually**:
   ```bash
   cd datasets
   unzip dataset1.zip -d dataset1
   unzip dataset2.zip -d dataset2
   ```

4. **Or use the script to extract**:
   ```bash
   python scripts/download_and_prepare.py --skip-download
   ```

## Advanced Usage

### Download Specific Dataset Only

If you know the file ID of a specific dataset:

```python
# Edit download_datasets.py to add file IDs
python scripts/download_datasets.py --dataset my_dataset
```

### Batch Process Multiple Folders

```bash
# Download from multiple folders
for folder_id in "folder1" "folder2" "folder3"; do
    python scripts/download_and_prepare.py \
        --folder-id "$folder_id" \
        --output-dir "datasets_${folder_id}"
done
```

### Resume Interrupted Downloads

The scripts are designed to skip existing files:

```bash
# If download was interrupted, just run again
python scripts/download_and_prepare.py
# It will skip already downloaded files
```

## File Structure After Download

```
datasets/
├── dataset1/
│   ├── class1/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── class2/
│       └── img001.jpg
├── dataset1_latents.pt  (if --encode used)
├── dataset2/
│   └── ...
└── dataset2_latents.pt  (if --encode used)
```

## Next Steps After Download

1. **Verify datasets**:
   ```bash
   ls -la datasets/*/
   ```

2. **Check dataset statistics**:
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
       --num-shots 5
   ```

## Support

For issues with:
- **Download scripts**: Check this README
- **Dataset utilities**: See `docs/DATASET_UTILS.md`
- **Image generation**: See `examples/README.md`
- **General usage**: See `GETTING_STARTED.md`

## Performance Tips

1. **Use SSD**: Download to SSD for faster extraction
2. **Parallel downloads**: Download different datasets in parallel (manually)
3. **Pre-encode**: Use `--encode` flag to prepare datasets for fast training
4. **Cleanup**: Use default settings to remove zips after extraction
