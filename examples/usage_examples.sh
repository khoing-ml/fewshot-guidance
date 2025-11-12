#!/bin/bash
# Example commands for using the fewshot dataset utilities

echo "=================================================="
echo "Fewshot Dataset Utilities - Usage Examples"
echo "=================================================="
echo ""

# Set your dataset path
DATASET_PATH="./my_fewshot_dataset"
OUTPUT_DIR="./outputs"

echo "Prerequisites:"
echo "  1. Organize your dataset in: $DATASET_PATH"
echo "  2. Directory structure should be:"
echo "     $DATASET_PATH/"
echo "       class1/"
echo "         img1.jpg"
echo "         img2.jpg"
echo "       class2/"
echo "         img1.jpg"
echo ""
echo "=================================================="
echo ""

# Example 1: Encode dataset to latents
echo "Example 1: Encode your dataset to latents"
echo "=================================================="
echo "Command:"
echo ""
cat << 'EOF'
python examples/prepare_dataset.py \
    --dataset-path ./my_fewshot_dataset \
    --output ./dataset_latents.pt \
    --mode encode \
    --batch-size 4 \
    --height 1024 \
    --width 1024 \
    --device cuda
EOF
echo ""
echo "This will:"
echo "  - Load all images from the dataset"
echo "  - Encode them to latent space using the VAE"
echo "  - Save latents to dataset_latents.pt (much faster for training)"
echo ""
echo "=================================================="
echo ""

# Example 2: Load and inspect latents
echo "Example 2: Load and inspect pre-encoded latents"
echo "=================================================="
echo "Command:"
echo ""
cat << 'EOF'
python examples/prepare_dataset.py \
    --dataset-path ./my_fewshot_dataset \
    --latent-path ./dataset_latents.pt \
    --mode load \
    --device cuda
EOF
echo ""
echo "=================================================="
echo ""

# Example 3: Create and test dataloader
echo "Example 3: Create and test a training dataloader"
echo "=================================================="
echo "Command:"
echo ""
cat << 'EOF'
python examples/prepare_dataset.py \
    --dataset-path ./my_fewshot_dataset \
    --mode dataloader \
    --num-shots 5 \
    --same-class \
    --train-batch-size 2 \
    --height 1024 \
    --width 1024
EOF
echo ""
echo "This creates a dataloader that:"
echo "  - Samples 5 fewshot images from the same class"
echo "  - Batches 2 queries together"
echo "  - Shuffles the data"
echo ""
echo "=================================================="
echo ""

# Example 4: Sample with fewshot guidance using dataset
echo "Example 4: Generate images using fewshot guidance from dataset"
echo "=================================================="
echo "Command:"
echo ""
cat << 'EOF'
python examples/sample_from_dataset.py \
    --dataset-path ./my_fewshot_dataset \
    --prompt "a beautiful landscape" \
    --num-shots 5 \
    --class-name "landscapes" \
    --guidance-type mlp \
    --steps 4 \
    --guidance-scale 1.0 \
    --guidance-train-steps 10 \
    --output outputs/landscape_fewshot.png \
    --device cuda
EOF
echo ""
echo "This will:"
echo "  - Sample 5 fewshot images from the 'landscapes' class"
echo "  - Use them to guide the generation"
echo "  - Train the guidance controller online"
echo "  - Generate an image matching the prompt and fewshot style"
echo ""
echo "=================================================="
echo ""

# Example 5: Original manual approach
echo "Example 5: Manual fewshot guidance (original approach)"
echo "=================================================="
echo "Command:"
echo ""
cat << 'EOF'
python src/fewshot-guidance/sample_with_fewshot_guidance.py \
    --prompt "a beautiful landscape" \
    --fewshot-images \
        path/to/img1.jpg \
        path/to/img2.jpg \
        path/to/img3.jpg \
    --guidance-type mlp \
    --steps 4 \
    --output outputs/landscape_manual.png
EOF
echo ""
echo "=================================================="
echo ""

echo "Quick Start Guide:"
echo "=================================================="
echo ""
echo "1. Prepare your dataset:"
echo "   mkdir -p my_fewshot_dataset/class1"
echo "   cp /path/to/images/*.jpg my_fewshot_dataset/class1/"
echo ""
echo "2. Encode it (one-time, recommended):"
echo "   python examples/prepare_dataset.py \\"
echo "       --dataset-path ./my_fewshot_dataset \\"
echo "       --output ./dataset_latents.pt \\"
echo "       --mode encode"
echo ""
echo "3. Generate images:"
echo "   python examples/sample_from_dataset.py \\"
echo "       --dataset-path ./my_fewshot_dataset \\"
echo "       --prompt \"your prompt here\" \\"
echo "       --num-shots 5 \\"
echo "       --output outputs/result.png"
echo ""
echo "=================================================="
echo ""

echo "For more information, see docs/DATASET_UTILS.md"
