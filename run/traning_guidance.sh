#! /bin/bash

DATASET="./datasets/512x512_sleeping_yellow_cat"
OUTPUT="./trained_models/512x512_guidance_model"
MODEL_TYPE="mlp"  # options: attention, conv, mlp
NUM_EPOCHS=30
BATCH_SIZE=4
LEARNING_RATE=1e-4
NUM_FEWSHOT=5
GUIDANCE_SCALE=1.0
CHECKPOINT_DIR="./checkpoints/512x512_guidance_model"
DEVICE="cuda"



python train_guidance.py \
    --dataset="$DATASET" \
    --flux-model="flux-dev" \
    --output_dir="$OUTPUT" \
    --model_type="$MODEL_TYPE" \
    --epochs=$NUM_EPOCHS \
    --batch-size=$BATCH_SIZE \
    --learning_rate=$LEARNING_RATE \
    --num_fewshot=$NUM_FEWSHOT \
    --guidance_scale=$GUIDANCE_SCALE \
    --checkpoint_dir="$CHECKPOINT_DIR" \
    --device="$DEVICE"

