"""
Simple script to sample an image from a dataset
"""

import argparse
from pathlib import Path
import random
from shutil import copyfile


def load_dataset(root_dir, extensions=['.jpg', '.jpeg', '.png', '.webp']):
    """
    Load image paths from a dataset directory.
    
    Supports two structures:
    1. Class-based: dataset_root/class1/img1.jpg, dataset_root/class2/img2.jpg
    2. Flat: dataset_root/img1.jpg, dataset_root/img2.jpg
    
    Returns:
        tuple: (image_paths, class_labels, class_to_idx)
    """
    root_dir = Path(root_dir)
    
    if not root_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {root_dir}")
    
    image_paths = []
    class_labels = []
    class_to_idx = {}
    
    # Check for class-based directory structure
    subdirs = [d for d in root_dir.iterdir() if d.is_dir()]
    
    if subdirs:
        # Class-based structure
        for class_idx, class_dir in enumerate(sorted(subdirs)):
            class_name = class_dir.name
            class_to_idx[class_name] = class_idx
            
            for ext in extensions:
                for img_path in class_dir.glob(f"*{ext}"):
                    image_paths.append(str(img_path))
                    class_labels.append(class_idx)
    else:
        # Flat structure - all images in one class
        for ext in extensions:
            for img_path in root_dir.glob(f"*{ext}"):
                image_paths.append(str(img_path))
                class_labels.append(0)
        class_to_idx = {'default': 0}
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {root_dir} with extensions {extensions}")
    
    return image_paths, class_labels, class_to_idx


def main():
    parser = argparse.ArgumentParser(
        description="Sample a single image from a dataset"
    )
    
    # Dataset arguments
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--class-name", type=str, default=None,
                       help="Sample from this class (optional)")
    parser.add_argument("--output", type=str, default="sampled_image.png",
                       help="Output path for the sampled image")
    
    args = parser.parse_args()
    
    # Load dataset
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    
    image_paths, class_labels, class_to_idx = load_dataset(args.dataset_path)
    
    print(f"Found {len(image_paths)} images in {len(class_to_idx)} classes")
    if len(class_to_idx) > 1 or 'default' not in class_to_idx:
        print(f"Classes: {list(class_to_idx.keys())}")
    
    # Sample one image
    print(f"\n" + "="*60)
    print("SAMPLING IMAGE")
    print("="*60)
    
    if args.class_name:
        if args.class_name in class_to_idx:
            class_idx = class_to_idx[args.class_name]
            # Get all indices for this class
            indices = [i for i, c in enumerate(class_labels) if c == class_idx]
            if indices:
                selected_idx = random.choice(indices)
                image_path = image_paths[selected_idx]
                print(f"Sampled from class '{args.class_name}': {Path(image_path).name}")
            else:
                print(f"ERROR: No images found for class '{args.class_name}'")
                return
        else:
            print(f"ERROR: Class '{args.class_name}' not found.")
            print(f"Available classes: {list(class_to_idx.keys())}")
            return
    else:
        idx = random.randint(0, len(image_paths) - 1)
        image_path = image_paths[idx]
        print(f"Randomly sampled: {Path(image_path).name}")
    
    # Copy the image to output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    copyfile(image_path, args.output)
    
    print(f"\n✓ Copied image to: {args.output}")
    print(f"  Source: {image_path}")
    print()


if __name__ == "__main__":
    main()
