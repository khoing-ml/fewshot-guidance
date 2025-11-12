"""
Script to download and unzip datasets from Google Drive.

This script downloads datasets from a Google Drive folder and extracts them
to a local directory for use with the fewshot-guidance framework.
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path
import subprocess


def install_gdown():
    """Install gdown if not already installed."""
    try:
        import gdown
        print("gdown is already installed")
        return True
    except ImportError:
        print("Installing gdown...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            print("gdown installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("Failed to install gdown")
            print("Please install manually: pip install gdown")
            return False


def download_from_google_drive(file_id, output_path, fuzzy=False):
    """
    Download a file from Google Drive.
    
    Args:
        file_id: Google Drive file ID or sharing URL
        output_path: Where to save the file
        fuzzy: Whether to use fuzzy matching for folder downloads
    """
    try:
        import gdown
    except ImportError:
        print("✗ gdown not installed. Run: pip install gdown")
        return False
    
    try:
        print(f"\nDownloading to: {output_path}")
        if fuzzy:
            # For folder downloads
            gdown.download_folder(file_id, output=str(output_path), quiet=False, use_cookies=False)
        else:
            # For single file downloads
            gdown.download(file_id, str(output_path), quiet=False, fuzzy=True)
        print(f"Download completed: {output_path}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def unzip_file(zip_path, extract_to):
    """
    Unzip a file to a directory.
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
    """
    try:
        print(f"\nUnzipping: {zip_path}")
        print(f"Extracting to: {extract_to}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            print(f"Found {len(file_list)} files in archive")
            zip_ref.extractall(extract_to)
        
        print(f"✓ Extraction completed")
        return True
    except zipfile.BadZipFile:
        print(f"✗ Not a valid zip file: {zip_path}")
        return False
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False


def download_dataset(dataset_info, output_dir, keep_zip=False):
    """
    Download and extract a single dataset.
    
    Args:
        dataset_info: Dict with 'name', 'file_id' or 'url'
        output_dir: Base directory for datasets
        keep_zip: Whether to keep the zip file after extraction
    
    Returns:
        Path to extracted dataset or None if failed
    """
    dataset_name = dataset_info['name']
    file_id = dataset_info.get('file_id')
    url = dataset_info.get('url')
    
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine download URL
    if url:
        download_url = url
    elif file_id:
        download_url = f"https://drive.google.com/uc?id={file_id}"
    else:
        print(f"No file_id or url provided for {dataset_name}")
        return None
    
    # Download
    zip_path = output_dir / f"{dataset_name}.zip"
    
    if zip_path.exists():
        print(f"Zip file already exists: {zip_path}")
        user_input = input("Download again? (y/n): ").lower()
        if user_input != 'y':
            print("Skipping download...")
        else:
            success = download_from_google_drive(download_url, zip_path)
            if not success:
                return None
    else:
        success = download_from_google_drive(download_url, zip_path)
        if not success:
            return None
    
    # Extract
    extract_dir = output_dir / dataset_name
    
    if extract_dir.exists():
        print(f"Extract directory already exists: {extract_dir}")
        user_input = input("Extract again? (y/n): ").lower()
        if user_input != 'y':
            print("Skipping extraction...")
            if not keep_zip and zip_path.exists():
                print(f"Removing zip file: {zip_path}")
                zip_path.unlink()
            return extract_dir
    
    success = unzip_file(zip_path, extract_dir)
    
    if not success:
        return None
    if not keep_zip:
        print(f"Removing zip file: {zip_path}")
        zip_path.unlink()
    
    return extract_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract datasets from Google Drive"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./datasets",
        help="Directory to save datasets (default: ./datasets)"
    )
    
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep zip files after extraction"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Download specific dataset only (optional)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit"
    )
    
    args = parser.parse_args()
    
    # Check/install gdown
    if not install_gdown():
        print("\n Please install gdown first: pip install gdown")
        sys.exit(1)
    
    datasets = [
        {
            'name': '512x512_vending_machine',
            'file_id': '1K_g3v17s_W48tyh5BLk4PjW0QIZbEYjF',  
            'description': 'images of vending machines from various angles'
        },
        
        # Add more datasets as needed
    ]
    
    if args.list:
        print("\nAvailable datasets:")
        print("="*60)
        for i, ds in enumerate(datasets, 1):
            print(f"{i}. {ds['name']}")
            if 'description' in ds:
                print(f"   {ds['description']}")
        print("="*60)
        print(f"\nTotal: {len(datasets)} datasets")
        print("\nTo download a specific dataset:")
        print("  python scripts/download_datasets.py --dataset dataset1")
        print("\nTo download all datasets:")
        print("  python scripts/download_datasets.py")
        return
    
    # Filter datasets if specific one requested
    if args.dataset:
        datasets = [ds for ds in datasets if ds['name'] == args.dataset]
        if not datasets:
            print(f"✗ Dataset '{args.dataset}' not found")
            print("Run with --list to see available datasets")
            sys.exit(1)
    
    # Download and extract datasets
    print(f"\n{'='*60}")
    print(f"DOWNLOADING DATASETS")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")
    print(f"Keep zip files: {args.keep_zip}")
    print(f"Datasets to download: {len(datasets)}")
    
    successful = []
    failed = []
    
    for dataset_info in datasets:
        result = download_dataset(dataset_info, args.output_dir, args.keep_zip)
        if result:
            successful.append(dataset_info['name'])
        else:
            failed.append(dataset_info['name'])
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Successful: {len(successful)}")
    for name in successful:
        print(f"  - {name}")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        for name in failed:
            print(f"  - {name}")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print(f"Your datasets are in: {args.output_dir}")
    print("\nVerify dataset structure:")
    print(f"   ls -la {args.output_dir}/*/")


if __name__ == "__main__":
    main()
