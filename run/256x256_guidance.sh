#!/bin/bash

# Training an MLP guidance model for few-shot images in the "dataset" folder

DATASET_PATH="./datasets/512x512_vending_machine"
MODEL_TYPE="mlp"
GUIDANCE_SCALE=1.0
GUIDANCE_TRAIN_STEPS=10
GUIDANCE_LR=0.001
IMAGE_SIZE=256
PROMPT="A vending machine"
# Allow DEVICE to be set via environment variable or first script argument, default to "cuda"
DEVICE="${1:-${DEVICE:-cuda}}"
OUTPUT_DIR="./output/mlp_guidance"
SEED=42
STEPS=4
LAMBDA_RECONSTRUCTION=1.0
LAMBDA_CONSISTENCY=0.1

mkdir -p $OUTPUT_DIR

# Find all fewshot reference images in the dataset directory
FEWSHOT_IMAGES=()
if [ -d "$DATASET_PATH" ]; then
    while IFS= read -r img_file; do
        FEWSHOT_IMAGES+=("$img_file")
    done < <(find "$DATASET_PATH" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | head -30)
    echo "Found ${#FEWSHOT_IMAGES[@]} fewshot reference images"
else
    echo "Error: Dataset path $DATASET_PATH does not exist"
    exit 1
fi

python src/fewshot-guidance/fewshot_guidance.py \
    --prompt "$PROMPT" \
    --fewshot-images "${FEWSHOT_IMAGES[@]}" \
    --guidance-type $MODEL_TYPE \
    --guidance-scale $GUIDANCE_SCALE \
    --guidance-train-steps $GUIDANCE_TRAIN_STEPS \
    --guidance-lr $GUIDANCE_LR \
    --height $IMAGE_SIZE \
    --width $IMAGE_SIZE \
    --steps $STEPS \
    --seed $SEED \
    --lambda-reconstruction $LAMBDA_RECONSTRUCTION \
    --lambda-consistency $LAMBDA_CONSISTENCY \
    --device $DEVICE \
    --output "$OUTPUT_DIR/guidance_output.png"